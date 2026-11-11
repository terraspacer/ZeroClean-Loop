"""Frozen evidence must survive retrieval drift and reject evidence substitution."""
import json

import pytest

from scripts import assemble_indexed_cts as indexed
from scripts import queue_cts_v2 as v2
from test_cts_v2 import fixture_files, response, run, tiny_source_cts
from zeroclean.types import digest, write_json


def frozen(tmp_path):
    table, seed, seed_path, detector_path = fixture_files(tmp_path)
    queue = tmp_path / "queue"
    entries = []
    responder = response([])
    def request(stage, prompt, metadata):
        body = dict(stage=stage, prompt=prompt, metadata=metadata, model=v2.MODEL, reasoning_effort=v2.EFFORT)
        key = digest(body)
        write_json(queue / "requests" / f"{key}.json", body)
        write_json(queue / "responses" / f"{key}.json", dict(request_hash=key, model=v2.MODEL,
                   reasoning_effort=v2.EFFORT, response=responder(stage, prompt, metadata)))
        entries.append(dict(request_hash=key, row=metadata["row"], column=metadata["column"]))
        return json.loads((queue / "responses" / f"{key}.json").read_text())["response"]
    original = run(tmp_path, table, seed_path, detector_path, request)
    index_path = tmp_path / "index.json"
    write_json(index_path, entries)
    return table, seed_path, detector_path, index_path, queue, original


def test_assembly_uses_original_evidence_without_retrieval(tmp_path, tiny_source_cts, monkeypatch):
    table, seed, detector, index, queue, original = frozen(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Retrieval must not rerun during assembly")
    monkeypatch.setattr(v2, "SourceCTS", forbidden)
    output = tmp_path / "assembled.json"
    result = indexed.assemble(table, seed, detector, index, output, queue)
    assert result["cells"] == original["cells"]
    assert result["request_hashes"] == original["request_hashes"]
    assert result["experiment_identity"] == original["experiment_identity"]
    assert result["indexed_assembly"]["retrieval_regenerated"] is False
    assert indexed.assemble(table, seed, detector, index, output, queue) == result


def test_original_cts_stage_preserves_exact_request_order(tmp_path, tiny_source_cts):
    table, seed, detector, index, queue, original = frozen(tmp_path)
    entries = json.loads(index.read_text())
    for entry in entries:
        key = entry["request_hash"]
        body = json.loads((queue / "requests" / f"{key}.json").read_text())
        prefix, serialized = body["prompt"].split("\n", 1)
        evidence = json.loads(serialized)
        evidence["retrieved_context"] = evidence.pop("masked_source_context")
        del evidence["leave_one_out_conditional_profiles"]
        for peer in evidence["observed_dirty_peers"]:
            del peer["trust"]
        body["prompt"] = prefix + "\n" + json.dumps(evidence)
        body["stage"] = "loop_cts_top3"
        body["metadata"]["stage_contract"] = "cts_top3"
        new_key = digest(body)
        write_json(queue / "requests" / f"{new_key}.json", body)
        reply = json.loads((queue / "responses" / f"{key}.json").read_text())
        reply["request_hash"] = new_key
        write_json(queue / "responses" / f"{new_key}.json", reply)
        entry["request_hash"] = new_key
    write_json(index, dict(dataset=table.name, requests=entries))
    result = indexed.assemble(table, seed, detector, index, tmp_path / "assembled.json", queue)
    expected = {(entry["row"], entry["column"]): entry["request_hash"] for entry in entries}
    assert result["request_hashes"] == [expected[c["row"], c["column"]] for c in result["cells"]]
    assert [(c["row"], c["column"]) for c in result["cells"]] == [(c["row"], c["column"]) for c in original["cells"]]
    assert result["mask_policy"] == "all_candidates"


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "request_tamper", "wrong_response", "missing_response", "false_peer", "false_profile"])
def test_assembly_rejects_substituted_or_incomplete_evidence(tmp_path, tiny_source_cts, mutation):
    table, seed, detector, index, queue, _ = frozen(tmp_path)
    entries = json.loads(index.read_text())
    key = entries[0]["request_hash"]
    request_path = queue / "requests" / f"{key}.json"
    response_path = queue / "responses" / f"{key}.json"
    if mutation == "incomplete":
        write_json(index, entries[1:])
    elif mutation == "duplicate":
        write_json(index, entries + [entries[0]])
    elif mutation == "request_tamper":
        body = json.loads(request_path.read_text())
        body["prompt"] += "tampered"
        write_json(request_path, body)
    elif mutation == "wrong_response":
        body = json.loads(response_path.read_text())
        body["request_hash"] = "0" * 64
        write_json(response_path, body)
    elif mutation == "missing_response":
        response_path.unlink()
    else:
        # Re-hashing a fabricated prompt cannot bypass dirty-evidence validation.
        body = json.loads(request_path.read_text())
        prefix, serialized = body["prompt"].split("\n", 1)
        evidence = json.loads(serialized)
        if mutation == "false_peer":
            evidence["observed_dirty_peers"][0]["tuple"]["target"] = "fabricated"
        else:
            evidence["leave_one_out_conditional_profiles"][0]["peer_row_count"] += 1
        body["prompt"] = prefix + "\n" + json.dumps(evidence)
        replacement_key = digest(body)
        write_json(queue / "requests" / f"{replacement_key}.json", body)
        reply = json.loads(response_path.read_text())
        reply["request_hash"] = replacement_key
        write_json(queue / "responses" / f"{replacement_key}.json", reply)
        entries[0]["request_hash"] = replacement_key
        write_json(index, entries)
    output = tmp_path / "assembled.json"
    with pytest.raises(ValueError):
        indexed.assemble(table, seed, detector, index, output, queue)
    assert not output.exists()
