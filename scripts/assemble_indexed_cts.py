#!/usr/bin/env python3
"""Assemble completed CTS judgments from frozen requests, without reretrieval.

Floating-point nearest-neighbor ties can change across numerical environments.
This transport consumes exactly the indexed, content-addressed prompts the model
answered. It does not substitute responses between different prompt hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import queue_cts_v2 as v2
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.llm import FileLLMQueue
from zeroclean.proposals import parse_proposals
from zeroclean.types import Prediction, digest, write_json


def assemble(table, payload_path, detection_path, index_path, output_path, queue_root="artifacts/queue"):
    payload_path, detection_path, index_path = map(lambda p: Path(p).resolve(),
                                                  (payload_path, detection_path, index_path))
    output_path, queue_root = Path(output_path), Path(queue_root).resolve()
    inputs = {str(p): sha256(p) for p in (payload_path, detection_path, index_path)}
    assembly_script = Path(__file__).resolve()
    v2_script = Path(v2.__file__).resolve()
    assembly_script_hash, v2_script_hash = sha256(assembly_script), sha256(v2_script)
    inputs.update({str(assembly_script): assembly_script_hash, str(v2_script): v2_script_hash})
    payload = json.loads(payload_path.read_text())
    detection = Prediction.load(detection_path)
    v2._validate_seed(table, payload, detection)
    index = json.loads(index_path.read_text())
    if isinstance(index, dict):
        if index.get("dataset") not in (None, table.name):
            raise ValueError("Indexed CTS dataset mismatch")
        if index.get("seed_request_hash") not in (None, payload["request_hash"]):
            raise ValueError("Indexed CTS seed mismatch")
        index = index.get("requests", index.get("attempted_requests"))
    if not isinstance(index, list):
        raise ValueError("CTS index must contain an explicit request list")
    expected = {(c["row"], c["column"]): c for c in payload["cells"]}
    indexed, file_hashes = {}, {}
    for entry in index:
        if (not isinstance(entry, dict) or type(entry.get("row")) is not int
                or not isinstance(entry.get("column"), str) or entry["column"] not in table.columns):
            raise ValueError("CTS index requires integer row and known column name")
        coordinate = (entry["row"], entry["column"])
        key = entry.get("request_hash", entry.get("hash"))
        if coordinate in indexed or coordinate not in expected:
            raise ValueError("CTS index must contain exactly one request per seed cell")
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid CTS request hash")
        indexed[coordinate] = key
    if set(indexed) != set(expected):
        raise ValueError("Incomplete CTS index coverage")
    queue = FileLLMQueue(queue_root)
    completed, request_ids, common = [], {}, None
    profiles, current_column = None, None
    retrieval_mask, source_mask, relaxed = v2.retrieval_only_mask(table, detection)
    retrieval_mask_hash = digest(retrieval_mask.tolist())
    source_mask_hash = digest(source_mask.tolist())
    v2_checks = {"experiment_policy": v2.POLICY, "seed_payload_sha256": inputs[str(payload_path)],
                 "source_detection_sha256": inputs[str(detection_path)],
                 "experiment_script_sha256": v2_script_hash,
                 "retrieval_only_mask_hash": retrieval_mask_hash, "mask_query_target": True}
    for coordinate, cell in sorted(expected.items(), key=lambda item: (item[1]["column_index"], item[0][0])):
        key = indexed[coordinate]
        request_path = queue_root / "requests" / f"{key}.json"
        response_path = queue_root / "responses" / f"{key}.json"
        request_bytes = request_path.read_bytes()
        request_file_hash = hashlib.sha256(request_bytes).hexdigest()
        if not response_path.is_file():
            raise ValueError(f"Missing response for indexed CTS request {key}")
        before = {str(request_path): request_file_hash, str(response_path): sha256(response_path)}
        body = json.loads(request_bytes)
        canonical = {k: body[k] for k in ("stage", "prompt", "metadata", "model", "reasoning_effort")}
        if digest(canonical) != key or body["model"] != v2.MODEL or body["reasoning_effort"] != v2.EFFORT:
            raise ValueError("Indexed CTS request digest/model mismatch")
        metadata = body["metadata"]
        row, column = coordinate
        j = cell["column_index"]
        required = {"dataset": table.name, "row": row, "column": column,
                    "dirty_fingerprint": table.fingerprint, "seed_request_hash": payload["request_hash"],
                    "system": "zeroclean_loop", "excludes_query_row": True, "k": 3}
        if any(metadata.get(k) != value for k, value in required.items()):
            raise ValueError("Indexed CTS metadata mismatch")
        if body["stage"] not in ("loop_cts_top3", v2.STAGE):
            raise ValueError("Unsupported indexed CTS stage")
        is_v2 = body["stage"] == v2.STAGE
        contract = "cts_top3_v2" if is_v2 else "cts_top3"
        if metadata.get("stage_contract") != contract:
            raise ValueError("Indexed CTS contract mismatch")
        evidence = json.loads(body["prompt"].split("\n", 1)[1])
        if (evidence.get("dataset") != table.name or evidence.get("target_column") != column
                or evidence.get("dirty_tuple") != cell["tuple"]
                or evidence.get("seed_alternatives") != [c["value"] for c in cell["candidates"]]):
            raise ValueError("Indexed CTS original evidence mismatch")
        context = evidence["masked_source_context" if is_v2 else "retrieved_context"]
        indices = context["indices"]
        if (context.get("row") != row or context.get("column") != column or context.get("column_index") != j
                or context.get("excluded_query_row") is not True or len(indices) > 3
                or any(type(i) is not int or not 0 <= i < len(table.rows) or i == row for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Indexed CTS retrieval coordinates mismatch")
        raw_peers = [{"row": i, "tuple": dict(zip(table.columns, table.rows[i]))} for i in indices]
        if is_v2:
            raw_peers = [dict(p, trust="untrusted_dirty_observation") for p in raw_peers]
        if evidence.get("observed_dirty_peers") != raw_peers:
            raise ValueError("Indexed CTS raw peers do not match dirty observations")
        provenance = context["provenance"]
        if provenance.get("dirty_fingerprint") != table.fingerprint or provenance.get("uses_ground_truth") is not False:
            raise ValueError("Indexed CTS retrieval provenance mismatch")
        version = {"stage": body["stage"], "mask_policy": metadata.get("mask_policy", "all_candidates"),
                   "provenance": provenance}
        if common is None:
            common = version
        elif common != version:
            raise ValueError("Mixed CTS experiment provenance")
        if is_v2:
            if any(metadata.get(k) != value for k, value in v2_checks.items()):
                raise ValueError("Indexed CTS-v2 identity mismatch")
            related, weights = context["related_column_indices"], context["weights"]
            if (len(related) != len(weights) or len(set(related)) != len(related)
                    or any(type(i) is not int or not 0 <= i < len(table.columns) for i in related)):
                raise ValueError("Invalid indexed CTS-v2 related columns")
            column_identity = (j, tuple(related), tuple(weights))
            if column_identity != current_column:
                profiles = v2.ConditionalProfiles(table, j, zip(related, weights))
                current_column = column_identity
            if profiles.profiles(row) != evidence.get("leave_one_out_conditional_profiles"):
                raise ValueError("Indexed CTS-v2 profiles do not match dirty counts")
        # Never create a replacement request when the original response is absent.
        parsed = parse_proposals(queue.request(body["stage"], body["prompt"], metadata), cell["original"], 3)
        v2._check_inputs(before)
        file_hashes.update(before)
        result = dict(row=row, column=column, **parsed)
        if is_v2:
            for candidate in result["candidates"]:
                candidate["count_evidence"] = profiles.candidate_evidence(row, candidate["value"], indices)
            result.update(conditional_profiles=evidence["leave_one_out_conditional_profiles"],
                          retrieved_row_indices=indices, retrieval_provenance=provenance)
        completed.append(result)
        request_ids[coordinate] = key
    if common is None:
        raise ValueError("Empty indexed CTS experiment needs an explicit stage")
    order = {key: i for i, key in enumerate(expected)}
    completed.sort(key=lambda c: order[c["row"], c["column"]])
    artifact = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": v2.MODEL,
                "reasoning_effort": v2.EFFORT, "cells": completed,
                "request_hashes": [request_ids[c["row"], c["column"]] for c in completed],
                "retrieval_method": "zeroec_source_cts", "seed_request_hash": payload["request_hash"],
                "retrieval_provenance": common["provenance"],
                "indexed_assembly": {"script_sha256": assembly_script_hash, "index_sha256": inputs[str(index_path)],
                    "original_request_and_response_files_hash": digest(file_hashes),
                    "input_file_hashes": dict(inputs, **file_hashes),
                    "retrieval_regenerated": False, "response_hash_substitution": False}}
    if common["stage"] == v2.STAGE:
        source_inputs = {str(p): inputs[str(p)] for p in (payload_path, detection_path)}
        identity = {"experiment_policy": v2.POLICY, "dirty_sha256": table.fingerprint,
                    "seed_request_hash": payload["request_hash"], "source_input_hashes": source_inputs,
                    "script_sha256": v2_script_hash, "source_decision_mask_hash": source_mask_hash,
                    "retrieval_only_mask_hash": retrieval_mask_hash}
        artifact.update(experiment_policy=v2.POLICY, experiment_identity=identity,
            script_sha256=v2_script_hash, seed_payload_sha256=inputs[str(payload_path)],
            source_detection_sha256=inputs[str(detection_path)], source_input_hashes=source_inputs,
            source_probabilities_hash=digest(detection.probabilities.tolist()),
            policy_disclosure={"same_policy_all_datasets": True, "source_predictions_modified": False,
                "candidate_eligibility": "exact_initial_seed_coordinates_and_order", "mask_query_target": True,
                "relaxed_columns": relaxed, "source_decision_mask_hash": identity["source_decision_mask_hash"],
                "retrieval_only_mask_hash": identity["retrieval_only_mask_hash"],
                "profiles": "Original dirty observations only; query excluded; counts are not labels",
                "top_observed_values_presentation_cap": v2.TOP_OBSERVED_VALUES,
                "claim": "Experimental retrieval hypothesis; no demonstrated improvement or calibrated confidence"})
    else:
        artifact.update(mask_policy=common["mask_policy"],
                        provenance="Original indexed source CTS prompts and exact model replies; no reretrieval or clean labels")
    artifact["artifact_hash"] = digest(artifact)
    v2._check_inputs(inputs)
    v2._check_inputs(file_hashes)
    if output_path.exists():
        if json.loads(output_path.read_text()) != artifact:
            raise ValueError("Refusing to overwrite a different assembled CTS artifact")
    else:
        write_json(output_path, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, default=Path("artifacts/queue"))
    args = parser.parse_args()
    artifact = assemble(load_dirty(args.dataset), args.payload, args.detection, args.index, args.output, args.queue_root)
    print(json.dumps({"dataset": args.dataset, "cells": len(artifact["cells"]), "artifact_hash": artifact["artifact_hash"]}), flush=True)


if __name__ == "__main__":
    main()
