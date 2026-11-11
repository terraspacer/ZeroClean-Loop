#!/usr/bin/env python3
"""Experimental dirty-only CTS-v2 proposals; no demonstrated accuracy claim.

The retrieval-only mask relaxes declared one-class source columns. Baseline
probabilities/decisions and initial candidate eligibility remain unchanged.
Conditional frequencies describe fallible observations, never reference labels.
"""
from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.llm import FileLLMQueue, ResponsePending
from zeroclean.proposals import parse_proposals
from zeroclean.types import Prediction, digest, write_json
from zeroclean.upstream_zeroec import SourceCTS, _detector_arrays


STAGE = "loop_cts_top3_v2"
POLICY = "degenerate_retrieval_mask_conditional_profiles_v2"
MODEL, EFFORT = "gpt-6.1-sol", "xhigh"
TOP_OBSERVED_VALUES = 5  # Presentation cap, not an acceptance/support threshold.


def retrieval_only_mask(table, detection):
    """Keep nondegenerate source decisions; unmask only declared one-class columns."""
    _, source = _detector_arrays(table, detection)
    mask = source.astype(bool).copy()
    changed = []
    classifiers = detection.metadata.get("classifier_columns", {})
    for j, column in enumerate(table.columns):
        classes = classifiers.get(column, {}).get("classes")
        if isinstance(classes, list) and len(classes) == 1 and type(classes[0]) in (int, float) and classes[0] in (0, 1):
            mask[:, j] = False
            changed.append({"column": column, "classes": classes,
                            "reason": "one_class_classifier_is_uninformative_for_retrieval_masking"})
    return mask, source.astype(bool).copy(), changed


class ConditionalProfiles:
    """One target column's exact dirty counts; subtract the query observation."""

    def __init__(self, table, target, related):
        self.table, self.target = table, target
        self.related = [(int(k), float(weight)) for k, weight in related if int(k) != target]
        self.groups = {}
        for k, _ in self.related:
            groups = defaultdict(Counter)
            for row in table.rows:
                groups[row[k]][row] += 1
            self.groups[k] = groups
        self.global_rows = Counter(table.rows)

    def _counts(self, rows, query):
        peers = rows.copy()
        if peers.get(query, 0):
            peers[query] -= 1
            if not peers[query]: del peers[query]
        counts = Counter()
        distinct = Counter()
        for row, count in peers.items():
            counts[row[self.target]] += count
            distinct[row[self.target]] += 1
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.
        return {"peer_row_count": total, "distinct_peer_tuple_count": len(peers),
                "duplicate_peer_rows": total - len(peers), "distinct_target_values": len(counts),
                "entropy_bits": float(entropy)}, counts, distinct

    def profiles(self, row_index):
        query = self.table.rows[row_index]
        result = []
        for k, weight in self.related:
            groups = self.groups[k]
            summary, counts, distinct = self._counts(groups.get(query[k], Counter()), query)
            total = summary["peer_row_count"]
            unique = summary["distinct_peer_tuple_count"]
            population = max(len(self.table.rows) - 1, 0)
            cardinality = len(groups) - int(sum(groups[query[k]].values()) == 1)
            status = ("unsupported_no_peer_observations" if not total else
                      "weak_singleton_peer_tuple" if unique == 1 else "untrusted_repeated_dirty_observations")
            if not query[k]: status = "unsupported_empty_conditioning_value"
            result.append({
                "conditioning_column": self.table.columns[k], "conditioning_value": query[k],
                "nmi_weight": weight, "nmi_weight_source": "original_source_full_dirty_table",
                "conditioning_cardinality": cardinality, "cardinality_population_rows": population,
                "conditioning_cardinality_fraction": cardinality / max(population, 1),
                "unique_conditioning_value_per_row": bool(population and cardinality == population),
                **summary, "original_value_count": counts.get(query[self.target], 0),
                "original_distinct_peer_tuple_count": distinct.get(query[self.target], 0),
                "competing_target_values": sum(value != query[self.target] for value in counts),
                "top_observed_values": [{"value": value, "count": count,
                                         "distinct_peer_tuple_count": distinct[value]}
                                        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_OBSERVED_VALUES]],
                "evidence_status": status, "query_observation_excluded": True,
                "trust": "untrusted_dirty_observations",
                "cardinality_caution": "A cardinality fraction near one can reflect row identity rather than useful semantic support; no cutoff or purity acceptance rule is applied.",
            })
        return result

    def candidate_evidence(self, row_index, value, retrieved_indices):
        query = self.table.rows[row_index]
        summary, counts, distinct = self._counts(self.global_rows, query)
        conditional = []
        for profile, (k, _) in zip(self.profiles(row_index), self.related):
            _, group_counts, group_distinct = self._counts(self.groups[k].get(query[k], Counter()), query)
            conditional.append({
                key: profile[key] for key in ("conditioning_column", "conditioning_value", "peer_row_count",
                                             "distinct_peer_tuple_count", "original_value_count", "entropy_bits",
                                             "conditioning_cardinality_fraction", "evidence_status")
            } | {"candidate_value_count": group_counts.get(value, 0),
                 "candidate_distinct_peer_tuple_count": group_distinct.get(value, 0)})
        retrieved_rows = [self.table.rows[index] for index in retrieved_indices]
        matching = [row for row in retrieved_rows if row[self.target] == value]
        return {"trust": "untrusted_dirty_observations", "query_observation_excluded": True,
                "global_target": {**summary, "candidate_value_count": counts.get(value, 0),
                                  "candidate_distinct_peer_tuple_count": distinct.get(value, 0),
                                  "original_value_count": counts.get(query[self.target], 0)},
                "conditional_profiles": conditional,
                "retrieved_peers": {"row_count": len(retrieved_rows), "candidate_value_count": len(matching),
                                    "candidate_distinct_peer_tuple_count": len(set(matching))}}


def _validate_seed(table, payload, detection):
    if payload.get("dataset") != table.name or payload.get("dirty_sha256") != table.fingerprint:
        raise ValueError("CTS-v2 seed dirty input mismatch")
    if payload.get("request_hash") != digest({key: value for key, value in payload.items() if key != "request_hash"}):
        raise ValueError("CTS-v2 seed content hash mismatch")
    if payload.get("model") != MODEL or payload.get("reasoning_effort") != EFFORT:
        raise ValueError("CTS-v2 seed model provenance mismatch")
    detection.validate(table.shape)
    if detection.metadata.get("input_fingerprint") != table.fingerprint or detection.metadata.get("reproduced_upstream") is not True:
        raise ValueError("CTS-v2 requires the original-source detector for this dirty input")
    seen = set()
    for cell in payload["cells"]:
        i, j = cell.get("row"), cell.get("column_index")
        if type(i) is not int or type(j) is not int or not 0 <= i < table.shape[0] or not 0 <= j < table.shape[1]:
            raise ValueError("CTS-v2 invalid seed coordinate")
        if (i, j) in seen: raise ValueError("CTS-v2 duplicate seed coordinate")
        seen.add((i, j))
        if (cell.get("column") != table.columns[j] or cell.get("original") != table.rows[i][j]
                or cell.get("tuple") != dict(zip(table.columns, table.rows[i]))):
            raise ValueError("CTS-v2 seed must retain the original dirty tuple")


def _check_inputs(inputs):
    for filename, expected in inputs.items():
        if sha256(filename) != expected: raise ValueError(f"CTS-v2 source input changed: {filename}")


def queue_cts_v2(table, payload_path, detection_path, request, *, output_path=None,
                  status_root="artifacts/cts_loop_v2", cache_root="artifacts/upstream_zeroec"):
    payload_path, detection_path = Path(payload_path).resolve(), Path(detection_path).resolve()
    inputs = {str(path): sha256(path) for path in (payload_path, detection_path)}
    payload = json.loads(payload_path.read_text())
    detection = Prediction.load(detection_path)
    _validate_seed(table, payload, detection)
    retrieval_mask, source_mask, relaxed = retrieval_only_mask(table, detection)
    identity = {"experiment_policy": POLICY, "dirty_sha256": table.fingerprint,
                "seed_request_hash": payload["request_hash"], "source_input_hashes": inputs,
                "script_sha256": sha256(__file__),
                "source_decision_mask_hash": digest(source_mask.tolist()),
                "retrieval_only_mask_hash": digest(retrieval_mask.tolist())}
    output_path = Path(output_path) if output_path is not None else Path("artifacts/llm") / table.name / "proposals_cts_v2.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if (existing.get("experiment_identity") != identity
                or existing.get("artifact_hash") != digest({key: value for key, value in existing.items() if key != "artifact_hash"})):
            raise ValueError("Refusing to overwrite a different completed CTS-v2 artifact")
        _check_inputs(inputs)
        return existing
    status_directory = Path(status_root) / table.name / digest(identity)
    cts = SourceCTS(table, retrieval_mask, Path(cache_root) / table.name,
                    exclude_query_row=True, max_cached_columns=2) if payload["cells"] else None
    completed, pending, errors, attempted = [], [], [], []
    ids = {}
    order = {(cell["row"], cell["column"]): position for position, cell in enumerate(payload["cells"])}
    profiles, current_column = None, None
    for cell in sorted(payload["cells"], key=lambda item: (item["column_index"], item["row"])):
        i, j = cell["row"], cell["column_index"]
        if j != current_column:
            state = cts.column_state(j)
            profiles = ConditionalProfiles(table, j, zip(state["indices"], state["weights"]))
            current_column = j
        context = cts.retrieve(i, j, k=3, mask_query_target=True)
        indices = context["indices"]
        if any(type(index) is not int or not 0 <= index < table.shape[0] or index == i for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("CTS-v2 retrieved peers must be distinct non-query dirty rows")
        conditional = profiles.profiles(i)
        evidence = {"dataset": table.name, "target_column": table.columns[j], "dirty_tuple": cell["tuple"],
                    "masked_source_context": context,
                    "observed_dirty_peers": [{"row": index, "tuple": dict(zip(table.columns, table.rows[index])),
                                               "trust": "untrusted_dirty_observation"} for index in indices],
                    "leave_one_out_conditional_profiles": conditional,
                    "seed_alternatives": [candidate["value"] for candidate in cell["candidates"]]}
        prompt = (
            "Propose at most three distinct exact-string tabular repair alternatives using only the supplied dirty evidence. "
            "Infer supported entity/context relationships and functional dependencies from combined schema and repeated tuple patterns. "
            "Compare the original and every alternative against those relationships and the supported exact-string representation. "
            "Rank the best-supported exact repairs and state concise corroborating evidence. "
            "Masked source CTS contexts and raw retrieved peers are complementary views of the same fallible observations, not clean references. "
            "The retrieval-only mask unmasks declared one-class classifier columns; it does not change source detector predictions or establish validity. "
            "Conditional counts exclude this query observation. Exact duplicate rows can inflate raw counts; inspect distinct peer tuple counts. "
            "High conditioning cardinality, singleton groups, empty conditions, and absent peers provide weak or unsupported evidence. "
            "Entropy and majority frequency are descriptions, never proof that a value is correct; no purity threshold establishes truth. "
            "Infer row grain, entity identity, source authority, units, and supported representations from combined schema and observed relationships; "
            "when ambiguity remains, lower confidence or abstain. Preserve valid outliers and legitimate missing values. You may return fewer than three alternatives. "
            "Scores are uncalibrated judgments, not benchmark accuracy. Give concise evidence explanations, not private reasoning. "
            "Return JSON {\"original_valid_score\":number,\"candidates\":[{\"value\":string,\"score\":number,\"explanation\":string}]}; "
            "finite scores in [0,1], alternatives differ from the original, no duplicate alternatives.\n"
            + json.dumps(evidence, ensure_ascii=False, allow_nan=False))
        metadata = {"dataset": table.name, "row": i, "column": table.columns[j], "dirty_fingerprint": table.fingerprint,
                    "system": "zeroclean_loop", "stage_contract": "cts_top3_v2", "experiment_policy": POLICY,
                    "seed_request_hash": payload["request_hash"], "seed_payload_sha256": inputs[str(payload_path)],
                    "source_detection_sha256": inputs[str(detection_path)], "retrieval_only_mask_hash": identity["retrieval_only_mask_hash"],
                    "experiment_script_sha256": identity["script_sha256"],
                    "excludes_query_row": True, "mask_query_target": True, "k": 3}
        for attempt in range(3):
            body = dict(metadata, attempt=attempt)
            request_hash = digest({"stage": STAGE, "prompt": prompt, "metadata": body, "model": MODEL, "reasoning_effort": EFFORT})
            attempted.append({"row": i, "column": table.columns[j], "attempt": attempt, "request_hash": request_hash})
            try:
                parsed = parse_proposals(request(STAGE, prompt, body), cell["original"], 3)
                for candidate in parsed["candidates"]:
                    candidate["count_evidence"] = profiles.candidate_evidence(i, candidate["value"], indices)
                completed.append({"row": i, "column": table.columns[j], **parsed,
                                  "conditional_profiles": conditional,
                                  "retrieved_row_indices": indices,
                                  "retrieval_provenance": context.get("provenance", {})})
                ids[i, table.columns[j]] = request_hash
                break
            except ResponsePending as exc:
                pending.append(str(exc)); break
            except (ValueError, TypeError, KeyError) as exc:
                errors.append({"row": i, "column": table.columns[j], "attempt": attempt, "reason": str(exc)})
    _check_inputs(inputs)
    write_json(status_directory / "status.json", {"dataset": table.name, "experiment_policy": POLICY,
               "experiment_identity": identity, "requested_cells": len(payload["cells"]), "completed_cells": len(completed),
               "pending_cells": len(pending), "response_errors": errors, "attempted_requests": attempted})
    if pending: raise ResponsePending(f"{len(pending)} CTS-v2 requests await independent experiment responses")
    if len(completed) != len(payload["cells"]): raise ValueError("CTS-v2 responses exhausted validation retries")
    completed.sort(key=lambda item: order[item["row"], item["column"]])
    artifact = {"dataset": table.name, "dirty_sha256": table.fingerprint, "model": MODEL, "reasoning_effort": EFFORT,
                "experiment_policy": POLICY, "experiment_identity": identity,
                "script_sha256": identity["script_sha256"],
                "seed_request_hash": payload["request_hash"], "seed_payload_sha256": inputs[str(payload_path)],
                "source_detection_sha256": inputs[str(detection_path)], "source_input_hashes": inputs,
                "source_probabilities_hash": digest(detection.probabilities.tolist()),
                "retrieval_method": "zeroec_source_cts", "cells": completed,
                "request_hashes": [ids[cell["row"], cell["column"]] for cell in completed],
                "retrieval_provenance": cts.provenance if cts is not None else {},
                "policy_disclosure": {"same_policy_all_datasets": True, "source_predictions_modified": False,
                    "candidate_eligibility": "exact_initial_seed_coordinates_and_order", "mask_query_target": True,
                    "relaxed_columns": relaxed, "source_decision_mask_hash": identity["source_decision_mask_hash"],
                    "retrieval_only_mask_hash": identity["retrieval_only_mask_hash"],
                    "profiles": "Original dirty observations only; query observation excluded; duplicate counts and cardinality exposed; no majority/purity acceptance threshold",
                    "top_observed_values_presentation_cap": TOP_OBSERVED_VALUES,
                    "claim": "Next-iteration retrieval hypothesis; no demonstrated improvement or calibrated confidence"}}
    artifact["artifact_hash"] = digest(artifact)
    _check_inputs(inputs)
    if output_path.exists():
        if json.loads(output_path.read_text()) != artifact:
            raise ValueError("Refusing to overwrite a different completed CTS-v2 artifact")
    else: write_json(output_path, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--payload", type=Path, required=True, help="Immutable original-source candidate seed")
    parser.add_argument("--detection", type=Path, required=True, help="Exact source detector prediction")
    parser.add_argument("--output", type=Path, help="Default: artifacts/llm/<dataset>/proposals_cts_v2.json")
    parser.add_argument("--status-root", type=Path, default=Path("artifacts/cts_loop_v2"))
    args = parser.parse_args()
    table = load_dirty(args.dataset)
    try:
        artifact = queue_cts_v2(table, args.payload, args.detection, FileLLMQueue().request,
                                output_path=args.output, status_root=args.status_root)
        print(json.dumps({"dataset": table.name, "completed_cells": len(artifact["cells"]),
                          "artifact_hash": artifact["artifact_hash"], "experiment_policy": POLICY}), flush=True)
    except ResponsePending as exc:
        print(str(exc), flush=True)


if __name__ == "__main__":
    main()
