"""Safe precomputation recommendation; the active assembler is not modified.

After the active assembly finishes, hoist the immutable V2 script SHA and both
mask digests outside its cell loop. Reuse one V2 metadata-check dictionary and
the same constants in the final identity. Bind the script SHA into watched input
files and recheck it before publication, preserving mutation detection.
"""
from pathlib import Path

import pytest

from scripts import assemble_indexed_cts as indexed
from scripts import queue_cts_v2 as v2
from test_indexed_cts import frozen
from test_cts_v2 import tiny_source_cts
from zeroclean.types import Prediction, digest


def test_precomputed_script_and_mask_hashes_are_equivalent_to_current_assembly(tmp_path, tiny_source_cts, monkeypatch):
    table, seed, detection, index, queue, _ = frozen(tmp_path)
    original = indexed.assemble(table, seed, detection, index, tmp_path / "original.json", queue)
    retrieval, source, _ = v2.retrieval_only_mask(table, Prediction.load(detection))
    cached_masks = [(retrieval.tolist(), digest(retrieval.tolist())), (source.tolist(), digest(source.tolist()))]
    original_digest, original_sha = indexed.digest, indexed.sha256
    script_path = Path(v2.__file__).resolve()
    script_hash = original_sha(script_path)
    hits = []
    def cached_digest(value):
        if isinstance(value, list):
            for matrix, expected in cached_masks:
                if value == matrix:
                    hits.append("mask")
                    return expected
        return original_digest(value)
    def cached_sha(path):
        if Path(path).resolve() == script_path:
            hits.append("script")
            return script_hash
        return original_sha(path)
    monkeypatch.setattr(indexed, "digest", cached_digest)
    monkeypatch.setattr(indexed, "sha256", cached_sha)
    cached = indexed.assemble(table, seed, detection, index, tmp_path / "cached.json", queue)
    assert "mask" in hits and "script" in hits
    assert cached == original
    assert (tmp_path / "cached.json").read_bytes() == (tmp_path / "original.json").read_bytes()


def test_precomputed_script_identity_must_remain_a_watched_input(tmp_path):
    script = tmp_path / "synthetic_v2.py"
    script.write_text("frozen implementation\n")
    watched = {str(script): indexed.sha256(script)}
    v2._check_inputs(watched)
    script.write_text("changed implementation\n")
    with pytest.raises(ValueError, match="input changed"):
        v2._check_inputs(watched)
