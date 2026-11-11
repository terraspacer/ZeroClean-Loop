"""Dirty-only reconstruction of the committed ZeroEC zero-shot correction path.

The named zero-shot templates were never committed upstream. This runner uses
an explicitly selected committed-template or hash-bound author-source profile
through the public ``run_reconstructed_*`` API, with reconstruction fidelity.
No upstream entrypoint, clean-data helper, evaluation, or generated code runs.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np


PROMPT_FILES = ("prompt_templates/SystemMessage.txt", "prompt_templates/HumanMessage.txt",
                "prompt_templates/examples.txt")
MODEL_FILES = ("quantized_true.onnx", "tokenizer.json", "sentence_bert_config.json",
               "1_Pooling/config.json")
COMMITTED_PROMPT_PROFILE = "committed_zeroec"
AUTHOR_PROMPT_PROFILE = "author_zerodc_29cde9e0dbcd"
AUTHOR_PROMPT_COMMIT = "29cde9e0dbcdde04c34bbc4c185c73b357817390"
AUTHOR_PROMPT_REPOSITORY = "https://github.com/YangChen32768/ZeroDC"
AUTHOR_PROMPT_HASHES = {
    "SystemMessage_EC.txt": "aecc5b7719a0da05b01d7d544b64ebd7ab10af5c61fae6a92010747a0ce66248",
    "HumanMessage_EC.txt": "3695f49cb835a0d2d931cccd91f1a1ebdceeb91b9be5ac5e6eb6ca7ca0b8d3f4",
    "general_examples_EC.txt": "b420e6160f2a8d92c56930f1d08bb4d7d94fefd05f5cf73d9bb727d7e23b4ee6",
}
DEFAULT_RECOVERED_PROMPT_DIR = Path("artifacts/recovered_prompts/zerodc_29cde9e0dbcd")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_profile(source: Path, profile: str, recovered_dir: str | Path | None,
                     use_general_examples: bool) -> tuple[dict, list[str]]:
    """Read selected prompt bytes and bind author candidates to audited hashes."""
    if profile == COMMITTED_PROMPT_PROFILE:
        paths = [source / name for name in PROMPT_FILES]
        keys = list(PROMPT_FILES)
        provenance = {"profile": profile, "repository": "https://github.com/YangChen32768/ZeroEC",
                      "commit": "d76ef45509d7e0f9e7cc0fa98d445c06afbb157c",
                      "historical_identity_established": False,
                      "fidelity_note": "Different committed correction templates substitute for missing EC paths"}
    elif profile == AUTHOR_PROMPT_PROFILE:
        root = Path(recovered_dir) if recovered_dir is not None else DEFAULT_RECOVERED_PROMPT_DIR
        paths = [root / name for name in AUTHOR_PROMPT_HASHES]
        keys = [f"author_recovered/ZeroDC@{AUTHOR_PROMPT_COMMIT}/{path.name}" for path in paths]
        provenance = {"profile": profile, "repository": AUTHOR_PROMPT_REPOSITORY,
                      "commit": AUTHOR_PROMPT_COMMIT, "historical_identity_established": False,
                      "fidelity_note": "Author-source candidates; identity to omitted pinned ZeroEC files remains unproven"}
    else:
        return {"profile": profile, "texts": [], "sha256": {}, "provenance": {}}, [f"Unknown ZeroEC prompt profile: {profile}"]
    issues, texts, hashes = [], [], {}
    for path, key in zip(paths[:3 if use_general_examples else 2], keys):
        if not path.is_file():
            issues.append(f"Missing selected ZeroEC prompt: {path}")
            continue
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if profile == AUTHOR_PROMPT_PROFILE and actual != AUTHOR_PROMPT_HASHES[path.name]:
            issues.append(f"Author-recovered prompt SHA256 mismatch: {path.name}")
        hashes[key] = actual
        texts.append(payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
    if profile == AUTHOR_PROMPT_PROFILE:
        manifest = paths[0].parent / "provenance.json"
        try:
            payload = manifest.read_bytes()
            audit = json.loads(payload)
            if audit.get("source_repository") != AUTHOR_PROMPT_REPOSITORY or audit.get("source_commit") != AUTHOR_PROMPT_COMMIT or audit.get("target_commit") != "d76ef45509d7e0f9e7cc0fa98d445c06afbb157c":
                raise ValueError("Repository or revision differs from the audited author source")
            recorded = {item["file"]: item["sha256"] for item in audit["files"]}
            if recorded != AUTHOR_PROMPT_HASHES:
                raise ValueError("Recorded author prompt hashes differ from the pinned profile")
            provenance["audit_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"Invalid author-recovered prompt provenance: {error}")
    provenance["uses_general_examples"] = use_general_examples
    provenance["prompt_sha256"] = hashes
    return {"profile": profile, "texts": texts, "sha256": hashes, "provenance": provenance}, issues


def reconstruction_preflight(source: Path, embedding: Path,
                             use_general_examples: bool = True,
                             prompt_profile: str = COMMITTED_PROMPT_PROFILE,
                             recovered_prompt_dir: str | Path | None = None) -> dict:
    import importlib.util
    required = ("correction_zero_shot.py",)
    missing = [str(source / name) for name in required if not (source / name).is_file()]
    missing += [str(embedding / name) for name in MODEL_FILES if not (embedding / name).is_file()]
    dependencies = [name for name in ("numpy", "pandas", "sklearn", "onnxruntime", "tokenizers")
                    if importlib.util.find_spec(name) is None]
    blockers = (["Missing reconstruction resources: " + ", ".join(missing)] if missing else [])
    prompts, prompt_issues = _prompt_profile(source, prompt_profile, recovered_prompt_dir, use_general_examples)
    blockers.extend(prompt_issues)
    if dependencies:
        blockers.append("Missing reconstruction runtime modules: " + ", ".join(dependencies))
    return {"baseline": "zeroec_reconstruction", "ready": not blockers, "blockers": blockers,
            "reproduced_upstream": False, "fidelity": ("author_source_reconstruction" if prompt_profile == AUTHOR_PROMPT_PROFILE else "source_derived_reconstruction"),
            "prompt_profile": prompt_profile, "prompt_provenance": prompts["provenance"],
            "strict_zero_shot": not use_general_examples,
            "uses_ground_truth": False, "uses_general_examples": use_general_examples,
            "missing_dependencies": dependencies,
            "source_sha256": {**{name: _file_hash(source / name) for name in required
                                  if (source / name).is_file()}, **prompts["sha256"]},
            "embedding_path": str(embedding.resolve())}


class MiniLMCPUEncoder:
    """The bundled quantized ONNX graph, masked mean pooling and normalization."""

    def __init__(self, root: Path, batch_size: int = 64):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        # fast_sentence_transformers==0.4.1 encode_batch explicitly overrides
        # sentence_bert_config.json's 256-token setting with max_length=512.
        self.max_length = 512
        pooling = json.loads((root / "1_Pooling/config.json").read_text())
        if not pooling.get("pooling_mode_mean_tokens") or any(
                enabled for name, enabled in pooling.items()
                if name.startswith("pooling_mode_") and name != "pooling_mode_mean_tokens"):
            raise ValueError("Bundled MiniLM pooling contract changed")
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / "quantized_true.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        result = []
        # Like FastSentenceTransformer.encode, batch by decreasing text length.
        order = np.argsort([-len(text) for text in texts])
        for start in range(0, len(texts), self.batch_size):
            items = self.tokenizer.encode_batch([texts[i] for i in order[start:start+self.batch_size]])
            feed = {"input_ids": np.asarray([item.ids for item in items], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in items], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in items], dtype=np.int64)}
            hidden = self.session.run(["last_hidden_state"],
                                      {key: value for key, value in feed.items() if key in self.inputs})[0]
            mask = feed["attention_mask"].astype(np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            result.append(pooled.astype(np.float32))
        if not result:
            return np.empty((0, 384), dtype=np.float32)
        return np.concatenate(result, axis=0)[np.argsort(order)]


def _source_helpers(source: Path) -> dict:
    allowed = {"format_row", "format_row_2", "sort_dicts", "form_examples"}
    tree = ast.parse((source / "correction_zero_shot.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in allowed]
    if {node.name for node in nodes} != allowed:
        raise ValueError("Upstream ZeroEC helper contract changed")
    namespace = {"json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<upstream-zeroec-helpers>", "exec"), namespace)
    return namespace


def normalized_mutual_information(frame, target: str) -> np.ndarray:
    """Serial equivalent of upstream calc_mi_2, including singleton filtering."""
    from sklearn.metrics import mutual_info_score
    values = []
    for column in frame.columns:
        filtered = frame if column == target else frame[
            frame.groupby([target, column])[target].transform("size") >= 2]
        values.append(mutual_info_score(filtered[target], filtered[column]) if len(filtered) else 0.)
    maximum = max(values, default=0.)
    return np.asarray(values, dtype=float) / maximum if maximum else np.zeros(len(values))


def nearest_rows(vectors: np.ndarray, query: np.ndarray, k: int = 30,
                 squared_norms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """NumPy equivalent of pinned LangChain's DOT_PRODUCT -> IndexFlatL2 path.

    Each cell embedding is normalized; concatenated database vectors and the
    weighted/masked query are deliberately NOT normalized again upstream.
    Stable row order resolves exact ties; backend float32 differences may alter
    ties near the 30-row cutoff or the subsequent two-decimal score rounding.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    norms = np.sum(vectors * vectors, axis=1) if squared_norms is None else squared_norms
    distances = np.maximum(norms + np.dot(query, query)
                           - 2 * (vectors @ query), 0.)
    indices = np.argsort(distances, kind="stable")[:k]
    return indices, distances[indices]


def _dirty_frame(table):
    import pandas as pd
    # Match upstream pd.read_csv(dtype=str).fillna('null'), including NA tokens.
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    stream.seek(0)
    return pd.read_csv(stream, dtype=str).fillna("null")


def _embedding_matrix(table, frame, embedding: Path, out: Path) -> tuple[np.ndarray, dict]:
    from .types import digest, write_json
    cache = out / "run_cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {name: _file_hash(embedding / name) for name in MODEL_FILES}
    key = digest({"dirty_fingerprint": table.fingerprint, "model_sha256": manifest,
                  "encoder": "cpu_onnx_masked_mean_l2_v1", "max_length": 512,
                  "batch_size": 64})
    metadata_file = cache / "embedding_key.json"
    # Keyed immutable files make metadata publication a single commit point.
    # Existing mapped readers retain their inode across a different cache key,
    # and a crash between matrix publication and metadata commit is harmless.
    matrix_file = cache / f"embeddings.{key}.npy"
    with (cache / ".embedding.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
        selected = matrix_file
        if metadata.get("key") == key:
            filename = metadata.get("matrix_file", "embeddings.npy")
            if filename not in ("embeddings.npy", matrix_file.name):
                raise ValueError("Invalid embedding cache filename")
            selected = cache / filename
        if selected.is_file():
            matrix = np.load(selected, mmap_mode="r", allow_pickle=False)
            if matrix.ndim != 3 or matrix.shape[:2] != frame.shape or matrix.shape[2] <= 0 or matrix.dtype != np.dtype(np.float32) or not np.isfinite(matrix).all():
                raise ValueError("Invalid cached dirty embedding matrix")
            if metadata.get("key") != key:
                # Recover a complete matrix published before an interrupted
                # metadata commit, without running the encoder a second time.
                write_json(metadata_file, {"key": key, "model_sha256": manifest,
                                           "matrix_file": selected.name})
            return matrix, manifest
        values = frame.values.flatten().tolist()
        encoded = MiniLMCPUEncoder(embedding).encode(values)
        if encoded.ndim != 2 or encoded.shape[0] != len(values) or encoded.shape[1] <= 0 or encoded.dtype != np.dtype(np.float32) or not np.isfinite(encoded).all():
            raise ValueError("Invalid generated dirty embedding matrix")
        matrix = encoded.reshape(*frame.shape, encoded.shape[1])
        descriptor, name = tempfile.mkstemp(prefix=f".embeddings.{key}.", suffix=".npy.tmp", dir=cache)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o444)
            temporary.replace(matrix_file)
            write_json(metadata_file, {"key": key, "model_sha256": manifest,
                                       "matrix_file": matrix_file.name})
        finally:
            temporary.unlink(missing_ok=True)
        return np.load(matrix_file, mmap_mode="r", allow_pickle=False), manifest


def _detector_arrays(table, detection) -> tuple[np.ndarray, np.ndarray]:
    """Accept source Predictions, explicit bool masks, or probability matrices."""
    if hasattr(detection, "probabilities"):
        detection.validate(table.shape)
        probabilities = np.asarray(detection.probabilities, dtype=float).copy()
        metadata = detection.metadata
        fingerprint = metadata.get("input_fingerprint")
        if fingerprint is not None and fingerprint != table.fingerprint:
            raise ValueError("CTS detector belongs to a different dirty input")
    else:
        probabilities = np.asarray(detection, dtype=float).copy()
        metadata = {}
    if probabilities.shape != table.shape or not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("CTS detection must be a shape-matched boolean or probability matrix")
    upstream_mask = metadata.get("upstream_binary_decisions")
    if upstream_mask is not None:
        mask = np.asarray(upstream_mask)
        if mask.shape != table.shape or mask.dtype != np.dtype(bool):
            raise ValueError("Source ZeroED decisions must be a shape-matched boolean matrix")
        if not np.array_equal(mask, probabilities > .5):
            raise ValueError("Source ZeroED probabilities disagree with its preserved prediction mask")
        detected = mask.astype(np.int8)
    else:
        detected = np.asarray(probabilities > .5, dtype=np.int8)
    probabilities.setflags(write=False)
    detected.setflags(write=False)
    return probabilities, detected


class SourceCTS:
    """Reusable source ZeroEC context-tuple selection on immutable dirty data.

    ``detection`` may be a source Prediction, boolean mask or probability matrix.
    The baseline's default includes the query row. Loop proposal generation can
    set ``exclude_query_row=True`` to prevent using the row being reconsidered
    as its own evidence. Embeddings share the existing dirty/model-keyed cache;
    MI, concatenated vectors and filtered documents are prepared once per used
    column, including columns whose scores are uncertain rather than positive.
    ``max_cached_columns`` optionally bounds these per-column states using LRU
    eviction. It only changes resource use; returned context and provenance
    remain identical to the default unbounded cache.
    """

    def __init__(self, table, detection, output_dir: str | Path,
                 source_dir: str | Path = "vendor/ZeroEC",
                 embedding_path: str | Path | None = None,
                 exclude_query_row: bool = False,
                 max_cached_columns: int | None = None):
        from .types import digest
        if max_cached_columns is not None and (type(max_cached_columns) is not int or max_cached_columns < 1):
            raise ValueError("CTS max_cached_columns must be a positive integer or None")
        self.table = table
        self.probabilities, self.detected = _detector_arrays(table, detection)
        self.source = Path(source_dir).resolve()
        self.embedding = Path(embedding_path).resolve() if embedding_path else self.source / "all-MiniLM-L6-v2"
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exclude_query_row = bool(exclude_query_row)
        self.max_cached_columns = max_cached_columns
        self.dirty = _dirty_frame(table)
        if self.dirty.shape != table.shape:
            raise ValueError("Source CSV normalization changed the dirty table shape")
        self.helpers = _source_helpers(self.source)
        self.matrix, self.model_sha256 = _embedding_matrix(table, self.dirty, self.embedding, self.output_dir)
        self.matrix.setflags(write=False)
        self.detection_sha256 = digest({"probabilities": self.probabilities.tolist(),
                                        "dirty_fingerprint": table.fingerprint})
        self.provenance = {"method": "zeroec_source_cts", "fidelity": "source_derived_reconstruction",
                           "uses_ground_truth": False, "uses_answer_examples": False,
                           "dirty_fingerprint": table.fingerprint,
                           "detection_sha256": self.detection_sha256,
                           "source_sha256": {"correction_zero_shot.py": _file_hash(self.source / "correction_zero_shot.py")},
                           "model_sha256": self.model_sha256,
                           "backend": "numpy_squared_l2_langchain_0.0.28_dot_product_fallback",
                           "retrieval_pool_size": 30, "related_column_threshold": .5,
                           "exclude_query_row": self.exclude_query_row}
        self.column_states = OrderedDict()

    def _column_index(self, column) -> int:
        if isinstance(column, str):
            try:
                return self.table.columns.index(column)
            except ValueError:
                raise ValueError(f"Unknown CTS column: {column}") from None
        if isinstance(column, (int, np.integer)) and not isinstance(column, (bool, np.bool_)) and 0 <= column < self.table.shape[1]:
            return int(column)
        raise ValueError("CTS column index is out of bounds")

    def column_state(self, column) -> dict:
        """Return cached source MI/vector/document state for one target column."""
        j = self._column_index(column)
        if j in self.column_states:
            if self.max_cached_columns is not None:
                self.column_states.move_to_end(j)
        else:
            # Release the least recently used state before allocating another
            # concatenated matrix. The shared cell embedding map stays intact.
            if self.max_cached_columns is not None and len(self.column_states) >= self.max_cached_columns:
                self.column_states.popitem(last=False)
            related = np.round(normalized_mutual_information(self.dirty, self.table.columns[j]), decimals=1)
            indices = np.flatnonzero(related >= .5)
            weights = related[indices]
            header = [self.table.columns[idx] for idx in indices]
            vectors = self.matrix[:, indices, :].reshape(len(self.dirty), len(indices) * self.matrix.shape[2])
            documents = [self.helpers["format_row_2"](row, header, mask)
                         for row, mask in zip(self.dirty.iloc[:, indices].values.tolist(), self.detected[:, indices])]
            norms = np.sum(vectors * vectors, axis=1)
            column_sums = self.detected[:, indices].sum(axis=1)
            for array in (indices, weights, vectors, norms, column_sums):
                array.setflags(write=False)
            self.column_states[j] = {"indices": indices, "weights": weights, "header": header,
                                      "vectors": vectors, "squared_norms": norms,
                                      "documents": documents, "column_sums": column_sums}
        return self.column_states[j]

    def retrieve(self, row: int, column, k: int = 3, *,
                 exclude_query_row: bool | None = None,
                 mask_query_target: bool = False) -> dict:
        """Return source-ranked tuples, row indices, MI weights and prompt text.

        The nearest-30 pool is selected before the source's rounded-distance /
        target-detection / error-count sort. Exclusion removes the query from
        that pool before its 30-row cap; the standalone baseline never excludes
        it. Only predicted-clean fields appear in each retrieved tuple.

        ``mask_query_target=True`` also zeros this query's target embedding,
        even when its detector score is uncertain. Other query fields still
        follow the global detector mask, and peer document masks never change.
        False preserves the prior complete returned JSON; only True adds the
        corresponding provenance flag for an explicitly versioned loop policy.
        """
        if not isinstance(row, (int, np.integer)) or isinstance(row, (bool, np.bool_)) or not 0 <= row < self.table.shape[0]:
            raise ValueError("CTS row index is out of bounds")
        if type(k) is not int or k < 0:
            raise ValueError("CTS context size must be a nonnegative integer")
        i, j = int(row), self._column_index(column)
        state = self.column_state(j)
        indices, weights = state["indices"], state["weights"]
        exclude = self.exclude_query_row if exclude_query_row is None else bool(exclude_query_row)
        selected, context = [], " "
        if self.table.name != "imdb":
            query = self.matrix[i, indices, :].copy()
            masked = self.detected[i, indices] == 1
            if mask_query_target:
                masked |= indices == j
            query[masked] = 0.
            query *= weights[:, None]
            nearest, distances = nearest_rows(state["vectors"], query.flatten(), 31 if exclude else 30,
                                               state["squared_norms"])
            candidates = [{"index": int(idx), "score": round(score, 2),
                           "target_column": int(self.detected[idx, j]), "sum": int(state["column_sums"][idx]),
                           "page_content": state["documents"][idx]}
                          for idx, score in zip(nearest, distances) if not (exclude and idx == i)][:30]
            selected = self.helpers["sort_dicts"](candidates, "score", "target_column", "sum")[:k]
            context = "".join(item["page_content"] + "\n" for item in selected)
        rows = [dict(item, score=float(item["score"]), tuple=json.loads(item["page_content"]))
                for item in selected]
        provenance = dict(self.provenance, exclude_query_row=exclude)
        if mask_query_target:
            provenance["mask_query_target"] = True
        return {"row": i, "column": self.table.columns[j], "column_index": j,
                "query_tuple": self.dirty.iloc[i, indices].to_dict(),
                "query_tuple_text": self.helpers["format_row"](self.dirty.iloc[i, indices], state["header"]),
                "erroneous_value": self.dirty.iat[i, j], "context": context,
                "selected": rows, "tuples": [item["tuple"] for item in rows],
                "indices": [item["index"] for item in rows],
                "related_column_indices": indices.tolist(), "related_columns": list(state["header"]),
                "weights": weights.tolist(), "excluded_query_row": exclude,
                "provenance": provenance}


def _prompt_templates(source: Path, helpers: dict, use_general_examples: bool,
                       prompt_profile: str = COMMITTED_PROMPT_PROFILE,
                       recovered_prompt_dir: str | Path | None = None) -> tuple[str, str, str]:
    selected, issues = _prompt_profile(source, prompt_profile, recovered_prompt_dir, use_general_examples)
    if issues:
        raise ValueError("; ".join(issues))
    system, human = selected["texts"][:2]
    examples = ""
    if use_general_examples:
        examples = helpers["form_examples"](ast.literal_eval(selected["texts"][2]))
    # SystemMessage.txt uses specific_examples; the missing EC entrypoint
    # provides dataset_specific_examples=' '. Both are empty in this path.
    return system, human, examples


FORMAT_INSTRUCTIONS = ('Return a JSON object with "chain_of_thought_for_correction" (string) '
                       'and "correction" (object mapping the erroneous column name to one corrected value).')


def _parse_correction(response: str, column: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else text
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("correction"), dict):
        raise ValueError("ZeroEC response must contain a correction object")
    if column not in value["correction"]:
        raise ValueError("ZeroEC response omitted the requested column")
    corrected = value["correction"][column]
    if corrected is not None and (isinstance(corrected,(dict,list,bool)) or not isinstance(corrected,(str,int,float))):
        raise ValueError("ZeroEC replacement must be a scalar string, number, or null")
    if isinstance(corrected,float) and not np.isfinite(corrected):
        raise ValueError("ZeroEC replacement must be finite")
    return "null" if corrected is None else str(corrected)


def external_repair_value(parsed,raw_original,source_original):
    """Invert the adapter's internal missing-value representation, without labels."""
    if parsed==source_original:
        return raw_original
    return "" if parsed=="null" else parsed


def run_source_zeroec(table, detection, request: Callable[[str, str, dict], str], out: Path,
                      source: Path, embedding: Path, status: dict,
                      use_general_examples: bool = True,
                      prompt_profile: str = COMMITTED_PROMPT_PROFILE,
                      recovered_prompt_dir: str | Path | None = None):
    from .llm import ResponsePending
    from .types import Prediction, digest, write_json
    probabilities, detected = _detector_arrays(table, detection)
    detection_hash = digest({"probabilities": detection.probabilities.tolist(),
                             "dirty_fingerprint": table.fingerprint})
    cts = SourceCTS(table, detection, out, source_dir=source, embedding_path=embedding,
                    exclude_query_row=False, max_cached_columns=2) if detected.any() else None
    helpers = cts.helpers if cts is not None else _source_helpers(source)
    system, human, examples = _prompt_templates(source, helpers, use_general_examples,
                                               prompt_profile, recovered_prompt_dir)
    model_hashes = cts.model_sha256 if cts is not None else {}
    repairs, pending, invalid = {}, [], []
    retrieval_manifest = {}
    completed = 0
    for j, column in enumerate(table.columns):
        targets = np.flatnonzero(detected[:, j])
        if not len(targets):
            continue
        state = cts.column_state(j)
        retrieval_manifest[column] = {"indices": state["indices"].tolist(), "weights": state["weights"].tolist()}
        for i in targets:
            i = int(i)
            retrieved = cts.retrieve(i, j, k=3)
            inputs = {"general_examples": examples, "specific_examples": " ",
                      "dataset_specific_examples": " ", "format_instructions": FORMAT_INSTRUCTIONS,
                      "Dirty_Tuple": retrieved["query_tuple_text"],
                      "Erroneous_value": '{' + f'"{column}": "{retrieved["erroneous_value"]}"' + '}',
                      "Relevant_clean_tuples": retrieved["context"]}
            prompt = "System: " + system.format(**inputs) + "\nHuman: " + human.format(**inputs)
            metadata = {"dataset": table.name, "dirty_fingerprint": table.fingerprint,
                        "baseline": "zeroed_zeroec_reconstruction", "fidelity": status["fidelity"],
                        "reproduced_upstream": False, "row": i, "column": column,
                        "column_index": j, "detection_sha256": detection_hash,
                        "source_sha256": status["source_sha256"], "model_sha256": model_hashes,
                        "prompt_profile": prompt_profile, "prompt_provenance": status["prompt_provenance"],
                        "use_general_examples": use_general_examples,
                        "retrieved_row_indices": retrieved["indices"]}
            try:
                # Keep each attempt separately content-addressed. Replaying an
                # invalid first response must not retry that same cached text.
                for attempt in range(3):
                    response = request("zeroec_cell_correction", prompt, dict(metadata, attempt=attempt))
                    try:
                        value = _parse_correction(response, column)
                        value = external_repair_value(value,table.rows[i][j],retrieved["erroneous_value"])
                        if value != table.rows[i][j]:
                            repairs[(i, j)] = value
                        completed += 1
                        break
                    except (ValueError, TypeError) as error:
                        invalid.append({"row": i, "column": column, "attempt": attempt,
                                        "error": str(error)})
            except ResponsePending as error:
                pending.append(str(error))
    report = {"baseline": "zeroed_zeroec_reconstruction", "fidelity": status["fidelity"],
              "dataset": table.name, "dirty_fingerprint": table.fingerprint,
              "detection_sha256": detection_hash, "detected_cells": int(detected.sum()),
              "completed_cells": completed, "pending_cells": len(pending),
              "response_errors": invalid, "related_columns": retrieval_manifest}
    write_json(out / "queue_status.json", report)
    if pending:
        raise ResponsePending(f"{len(pending)} ZeroEC correction cells pending; all detected cells were enqueued. "
                              f"First: {pending[0]}")
    metadata = dict(report, name="zeroed_zeroec_reconstruction", reproduced_upstream=False,
                    strict_zero_shot=not use_general_examples, uses_ground_truth=False,
                    cached_results_reused=False, source_sha256=status["source_sha256"],
                    prompt_profile=prompt_profile, prompt_provenance=status["prompt_provenance"],
                    model_sha256=model_hashes, detection_metadata=detection.metadata,
                    upstream_binary_decisions=detected.astype(bool).tolist(),
                    decision_threshold=.5, decision_threshold_operator=">",
                    probabilities_semantics=detection.metadata.get("probabilities_semantics", "Uncalibrated source ZeroED classifier estimates"),
                    retrieval_provenance=cts.provenance if cts is not None else None,
                    adaptation_warnings=[
                        ("Author ZeroDC EC prompt candidates selected explicitly; historical identity to pinned ZeroEC is unproven"
                         if prompt_profile == AUTHOR_PROMPT_PROFILE else "Missing EC templates reconstructed from committed SystemMessage.txt and HumanMessage.txt"),
                        ("Selected profile's fixed general correction examples used" if use_general_examples
                         else "General correction examples omitted to satisfy zero-example policy"),
                        "Manual JSON schema instructions replace missing LangChain parser formatting",
                        "CPU ONNX execution of bundled quantized_true.onnx; source 512-token limit, batch size 64, masked mean pooling and L2 normalization",
                        "Quantized CPU numerics can differ from CUDA execution",
                        "NumPy squared L2 matches pinned LangChain DOT_PRODUCT fallback to IndexFlatL2",
                        "Stable row order resolves exact retrieval ties; FAISS float32 rounding may differ",
                        "Logging uses row position; no index data column is required",
                        "Clean/evaluation/human repair/code/FD stages omitted; no correction feedback"])
    metadata["null_sentinel_policy"]="Unchanged source sentinel retains original raw text; newly introduced null maps to an empty CSV cell"
    write_json(out / "metadata.json", metadata)
    prediction = Prediction(probabilities.copy(), repairs, metadata).validate(table.shape)
    prediction.save(out / "prediction.json")
    return prediction
