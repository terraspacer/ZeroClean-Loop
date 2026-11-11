"""Additional synthetic integrity regressions for frozen-prompt CTS assembly."""
import json
from pathlib import Path

import pytest

from scripts import assemble_indexed_cts as indexed
from scripts import queue_cts_v2 as v2
from test_indexed_cts import frozen
from test_cts_v2 import tiny_source_cts
from zeroclean.data import sha256
from zeroclean.types import write_json


def test_indexed_response_effort_provenance_is_already_enforced(tmp_path, tiny_source_cts):
    table, seed, detector, index, queue, _ = frozen(tmp_path)
    first = json.loads(index.read_text())[0]["request_hash"]
    path = queue / "responses" / f"{first}.json"
    reply = json.loads(path.read_text())
    reply["reasoning_effort"] = "low"
    write_json(path, reply)
    with pytest.raises(ValueError, match="provenance"):
        indexed.assemble(table, seed, detector, index, tmp_path / "assembled.json", queue)


def test_request_mutation_between_validation_and_file_hash_capture_is_rejected(tmp_path, tiny_source_cts, monkeypatch):
    table, seed, detector, index, queue, _ = frozen(tmp_path)
    entries = json.loads(index.read_text())
    key = next(entry["request_hash"] for entry in entries if entry["row"] == 0 and entry["column"] == "target")
    path = queue / "requests" / f"{key}.json"
    original = v2.ConditionalProfiles.profiles
    edited = []
    def mutate(self, row):
        result = original(self, row)
        if not edited:
            body = json.loads(path.read_text())
            body["prompt"] += "\nChanged after the assembler read this request."
            write_json(path, body)
            edited.append(True)
        return result
    monkeypatch.setattr(v2.ConditionalProfiles, "profiles", mutate)
    output = tmp_path / "assembled.json"
    with pytest.raises(ValueError, match="changed|request|Request"):
        indexed.assemble(table, seed, detector, index, output, queue)
    assert edited and not output.exists()


def test_future_assembly_exposes_each_bound_input_file_hash(tmp_path, tiny_source_cts):
    table, seed, detector, index, queue, _ = frozen(tmp_path)
    result = indexed.assemble(table, seed, detector, index, tmp_path / "assembled.json", queue)
    recorded = result["indexed_assembly"]["input_file_hashes"]
    expected_paths = [seed, detector, index, Path(indexed.__file__), Path(indexed.v2.__file__)]
    for entry in json.loads(index.read_text()):
        key = entry["request_hash"]
        expected_paths.extend(queue / kind / f"{key}.json" for kind in ("requests", "responses"))
    assert recorded == {str(Path(path).resolve()): sha256(path) for path in expected_paths}


@pytest.mark.parametrize("alias", [0., False])
def test_index_coordinates_cannot_alias_integer_rows_with_other_types(tmp_path, tiny_source_cts, alias):
    table, seed, detector, index, queue, _ = frozen(tmp_path)
    entries = json.loads(index.read_text())
    next(entry for entry in entries if entry["row"] == 0)["row"] = alias
    write_json(index, entries)
    with pytest.raises(ValueError, match="coordinate|integer|row"):
        indexed.assemble(table, seed, detector, index, tmp_path / "assembled.json", queue)
