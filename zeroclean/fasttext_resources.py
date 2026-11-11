"""Prepare immutable, source-equivalent fastText reductions without full loads.

The ordinary helper loads the full C++ model and then copies its large matrices.
Here those exact unquantized matrices are read-only memory maps. PCA retains
the upstream computation; complex projections stream in bounded row blocks to
avoid promoting the complete input to complex. Each dimension starts from the
original matrices, never another reduced checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import mmap
import os
from pathlib import Path
import struct
import sys

import numpy as np


DEFAULT_DIMENSIONS = (6, 9, 11, 17, 22)
PROJECTION_BLOCK_ROWS = 8192


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _binary_layout(path: Path) -> dict:
    """Validate the nonquantized fastText v12 on-disk matrix layout.

    The C++ serialization uses 12 int32 arguments and one double, followed by
    dictionary entries (zero-terminated word, int64 count, int8 entry type),
    then each matrix's quantization flag and int64 row/column counts.
    """
    if sys.byteorder != "little":
        raise ValueError("The memory-mapped fastText reader requires a little-endian host")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        if len(view) < 108:
            raise ValueError("Truncated fastText model header")
        magic, version = struct.unpack_from("<ii", view, 0)
        if magic != 793712314 or version != 12:
            raise ValueError("Only the unquantized fastText v12 model format is supported")
        dimension = struct.unpack_from("<i", view, 8)[0]
        count, words, labels, _, pruned = struct.unpack_from("<iiiqq", view, 64)
        if dimension <= 0 or count < 0 or words < 0 or labels < 0 or words + labels != count or pruned < -1:
            raise ValueError("Invalid fastText dictionary metadata")
        position = 92
        for _ in range(count):
            end = view.find(b"\0", position)
            if end < 0 or end + 10 > len(view):
                raise ValueError("Truncated fastText dictionary entry")
            position = end + 10
        position += max(pruned, 0) * 8
        if position + 17 > len(view) or view[position] != 0:
            raise ValueError("Quantized or truncated input matrices are unsupported")
        input_header = position + 1
        input_rows, input_dim = struct.unpack_from("<qq", view, input_header)
        input_offset = input_header + 16
        output_flag = input_offset + input_rows * input_dim * 4
        if input_rows < 2 or input_dim != dimension or output_flag + 17 > len(view) or view[output_flag] != 0:
            raise ValueError("Invalid input matrix or unsupported quantized output matrix")
        output_rows, output_dim = struct.unpack_from("<qq", view, output_flag + 1)
        output_offset = output_flag + 17
        if output_rows < 1 or output_dim != dimension or output_offset + output_rows * output_dim * 4 != len(view):
            raise ValueError("Invalid or truncated fastText output matrix")
    return {"dimension": dimension, "prefix_bytes": input_header,
            "input_offset": input_offset, "input_rows": input_rows,
            "output_offset": output_offset, "output_rows": output_rows}


def prepared_fasttext_provenance(model_path: str | Path) -> dict | None:
    """Verify a prepared model against its immutable cache manifest, if present."""
    path = Path(model_path).resolve()
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or "identity" not in manifest or "dimensions" not in manifest:
        return None
    entries = [entry for entry in manifest["dimensions"].values()
               if Path(entry["path"]).resolve() == path]
    if len(entries) != 1:
        raise ValueError("Prepared fastText model is not listed uniquely in its cache manifest")
    entry = entries[0]
    if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
        raise ValueError("Prepared fastText model does not match its immutable SHA256")
    if _binary_layout(path)["dimension"] != entry["dimension"]:
        raise ValueError("Prepared fastText model dimension disagrees with its cache manifest")
    return {"model_sha256": entry["sha256"], "dimension": entry["dimension"],
            "projection_sha256": entry["projection_sha256"],
            "original_sha256": manifest["identity"]["original_sha256"],
            "reduction_identity": manifest["identity"], "manifest_path": str(manifest_path)}


@contextmanager
def _read_only_matrix(path: Path, offset: int, rows: int, dimension: int):
    matrix = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(rows, dimension))
    try:
        yield matrix
    finally:
        matrix._mmap.close()


def _discard_mapped_pages(matrix):
    """Drop this process's clean source pages after each bounded projection."""
    mapping = getattr(matrix, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        try:
            mapping.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            # This advice is optional and never changes matrix contents.
            pass


def _pca_basis(matrix: np.ndarray) -> np.ndarray:
    """The exact first-100000 covariance/eigv lines from upstream reduction."""
    values = matrix[:100000]
    values = values - values.mean(axis=0, dtype=np.float32)
    covariance = np.divide(np.matmul(values.T, values), values.shape[0] - 1, dtype=np.float32)
    _, basis = np.linalg.eig(covariance)
    return basis


def _write_projected_matrix(matrix: np.ndarray, projection: np.ndarray, destination,
                            block_rows: int = PROJECTION_BLOCK_ROWS):
    """Write upstream float32-cast projections with bounded complex promotion.

    Complex64 projection requires casting the float32 left operand; a full
    matmul promotes the entire several-GB matrix. Only an 8192-row slice is
    promoted here. The real branch retains full-shape GEMM, because splitting
    real GEMM can select a different OpenBLAS kernel and change final bits.
    It has no full input dtype promotion and allocates only its narrow result.
    """
    if np.iscomplexobj(projection):
        for start in range(0, len(matrix), block_rows):
            projected = np.matmul(matrix[start:start+block_rows], projection)
            # _FastText.set_matrices performs this exact float32 cast, including
            # discarding imaginary roundoff components in complex eigenvectors.
            converted = np.asarray(projected, dtype=np.float32)
            converted.tofile(destination)
            del projected, converted
            _discard_mapped_pages(matrix)
    else:
        projected = np.matmul(matrix, projection)
        converted = np.asarray(projected, dtype=np.float32)
        converted.tofile(destination)
        del projected, converted
        _discard_mapped_pages(matrix)


def _write_reduced(original: Path, target: Path, layout: dict, dimension: int,
                   projection: np.ndarray):
    with original.open("rb") as source, target.open("xb") as destination:
        # Prefix preserves all model arguments/dictionary bytes, including the
        # false input-quantization flag. Only args.dim is patched below.
        remaining = layout["prefix_bytes"]
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Original fastText changed while copying metadata")
            destination.write(block)
            remaining -= len(block)
        destination.write(struct.pack("<qq", layout["input_rows"], dimension))
        with _read_only_matrix(original, layout["input_offset"], layout["input_rows"], layout["dimension"]) as matrix:
            _write_projected_matrix(matrix, projection, destination)
        destination.write(b"\0")
        destination.write(struct.pack("<qq", layout["output_rows"], dimension))
        with _read_only_matrix(original, layout["output_offset"], layout["output_rows"], layout["dimension"]) as matrix:
            _write_projected_matrix(matrix, projection, destination)
        destination.seek(8)
        destination.write(struct.pack("<i", dimension))
        destination.flush()
        os.fsync(destination.fileno())


def prepare_fasttext_dimensions(original_path: str | Path, output_dir: str | Path,
                                dimensions=DEFAULT_DIMENSIONS,
                                progress=None) -> dict:
    """Sequentially cache fresh-original reductions using upstream PCA exactly.

    The original checkpoint is never loaded as a C++ model. Complex projection
    streams 8192-row blocks, immediately casts/writes the narrow result, and
    advises the OS to release mapped source pages. Real projection retains the
    original full-shape GEMM to preserve its exact kernel/numerics. PCA retains
    upstream's first-100000 mean/covariance, unsorted ``np.linalg.eig`` basis,
    float32 output casts, and absence of RNG consumption.
    Results are byte-hashed, atomically installed, and made read-only. Existing
    results must match both provenance and their immutable hashes.
    """
    from fasttext.util.util import _reduce_matrix
    from .types import write_json
    original = Path(original_path).resolve()
    root = Path(output_dir).resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    selected = tuple(sorted(set(dimensions)))
    if not selected or any(type(value) is not int or value <= 0 for value in selected):
        raise ValueError("fastText target dimensions must be positive integers")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        # A process-wide lock prevents concurrent large-matrix preparation.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        layout = _binary_layout(original)
        if any(value >= layout["dimension"] for value in selected):
            raise ValueError("Each prepared dimension must be below the original model dimension")
        original_stat = original.stat()
        if progress:
            progress({"stage": "hashing_original", "path": str(original)})
        source_hash = _sha256(original)
        function_hash = hashlib.sha256(inspect.getsource(_reduce_matrix).encode()).hexdigest()
        execution_hash = hashlib.sha256((inspect.getsource(_pca_basis) + inspect.getsource(_write_projected_matrix)).encode()).hexdigest()
        identity = {"original_sha256": source_hash, "original_bytes": original_stat.st_size,
                    "numpy_version": np.__version__, "reduction_function_sha256": function_hash,
                    "bounded_execution_sha256": execution_hash, "projection_block_rows": PROJECTION_BLOCK_ROWS,
                    "algorithm": "upstream_pca_complex_row_blocks_real_full_matmul_v2"}
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("identity") != identity:
                raise ValueError("Existing fastText cache provenance does not match the original/runtime")
        else:
            try:
                package_version = importlib.metadata.version("fasttext-wheel")
            except importlib.metadata.PackageNotFoundError:
                package_version = importlib.metadata.version("fasttext")
            manifest = {"identity": identity, "original_path": str(original),
                        "fasttext_package_version": package_version, "rng_consumed": False,
                        "uses_ground_truth": False, "dimensions": {}}
        for dimension in selected:
            key = str(dimension)
            destination = root / f"cc.en.{dimension}.bin"
            existing = manifest["dimensions"].get(key)
            if existing is not None:
                if not destination.is_file() or destination.stat().st_size != existing["bytes"] or _sha256(destination) != existing["sha256"]:
                    raise ValueError(f"Immutable fastText cache is missing or modified: {destination}")
                if progress:
                    progress({"stage": "cached_dimension", "dimension": dimension, "path": str(destination)})
                continue
            if destination.exists():
                raise ValueError(f"Untracked fastText cache file already exists: {destination}")
            if progress:
                progress({"stage": "reducing_dimension", "dimension": dimension, "path": str(destination)})
            temporary = root / f".{destination.name}.{os.getpid()}.building"
            try:
                with _read_only_matrix(original, layout["input_offset"], layout["input_rows"], layout["dimension"]) as original_input:
                    projection = _pca_basis(original_input)[:, :dimension]
                    _discard_mapped_pages(original_input)
                if progress:
                    progress({"stage": "projection_ready", "dimension": dimension,
                              "projection_dtype": str(projection.dtype),
                              "projection_execution": "bounded_row_blocks" if np.iscomplexobj(projection) else "source_full_shape_real_gemm"})
                _write_reduced(original, temporary, layout, dimension, projection)
                projection_hash = hashlib.sha256(projection.tobytes()).hexdigest()
                projection_dtype = str(projection.dtype)
                del projection
                if _binary_layout(temporary)["dimension"] != dimension:
                    raise ValueError("Prepared fastText serialization has an unexpected dimension")
                prepared_hash = _sha256(temporary)
                prepared_size = temporary.stat().st_size
                temporary.chmod(0o444)
                temporary.replace(destination)
                manifest["dimensions"][key] = {"path": str(destination), "dimension": dimension,
                                               "sha256": prepared_hash, "bytes": prepared_size,
                                               "projection_sha256": projection_hash,
                                               "projection_dtype": projection_dtype}
                write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                if progress:
                    progress({"stage": "prepared_dimension", **manifest["dimensions"][key]})
            finally:
                if temporary.exists():
                    temporary.unlink()
        current = original.stat()
        if (current.st_size, current.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
            raise ValueError("Original fastText model changed during preparation")
        return manifest
