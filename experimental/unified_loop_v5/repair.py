"""Dirty-only repair selection after immutable, empty-state error detection.

Selection is provisional until the caller checks all selected edits in a row
together.  No replacement is synthesized, and no input object is modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from decimal import Decimal, InvalidOperation
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class RepairConfig:
    candidate_cap: int = 3
    feature_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
    tolerance: float = 1e-12
    fd_min_peers: int = 2
    cts_min_peers: int = 3
    cts_min_agreement: float = 0.9
    cts_min_margin: float = 0.5
    unit_min_numeric_peers: int = 2
    unit_min_numeric_fraction: float = 0.6
    consensus_min_sources: int = 2
    consensus_min_winning_sources: int = 2
    consensus_min_agreement: float = 2 / 3
    consensus_min_margin: float = 0.2
    catalog_min_exact_peers: int = 2
    numeric_min_exact_peers: int = 3
    numeric_min_canonical_agreement: float = 0.95


DEFAULT_CONFIG = RepairConfig()
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_BARE_NUMBER = re.compile(_NUMBER + r"\Z")


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _config(config):
    if config is None:
        result = DEFAULT_CONFIG
    elif isinstance(config, RepairConfig):
        result = config
    elif isinstance(config, Mapping):
        unknown = set(config) - {f.name for f in fields(RepairConfig)}
        if unknown:
            raise ValueError(f"Unknown repair configuration fields: {sorted(unknown)}")
        result = replace(DEFAULT_CONFIG, **dict(config))
    else:
        raise TypeError("config must be RepairConfig, a mapping, or None")
    limits = (("candidate_cap", 1, 3), ("fd_min_peers", 2, None),
              ("cts_min_peers", 3, None), ("unit_min_numeric_peers", 2, None),
              ("consensus_min_sources", 2, None), ("consensus_min_winning_sources", 2, None))
    limits += (("catalog_min_exact_peers", 2, None), ("numeric_min_exact_peers", 3, None))
    for name, lower, upper in limits:
        value = getattr(result, name)
        if (not isinstance(value, Integral) or isinstance(value, bool) or value < lower
                or (upper is not None and value > upper)):
            raise ValueError(f"Invalid {name}")
    if not _finite(result.tolerance) or result.tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    for name in ("cts_min_agreement", "cts_min_margin", "unit_min_numeric_fraction",
                 "consensus_min_agreement", "consensus_min_margin", "numeric_min_canonical_agreement"):
        value = getattr(result, name)
        if not _finite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {name}")
    weights = result.feature_weights
    if (len(weights) != 4 or not all(_finite(v) and v >= 0 for v in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("feature_weights must be four nonnegative finite weights summing to one")
    return replace(result, feature_weights=tuple(float(v) for v in weights))


def _column_key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _catalog_roles(columns):
    # v5 deliberately performs no benchmark-schema dispatch. Functional
    # dependencies are discovered from repeated dirty tuples and model rules.
    return None


def _mask_snapshot(mask, n, m):
    raw = mask.tolist() if hasattr(mask, "tolist") else mask
    if len(raw) != n or any(len(row) != m for row in raw):
        raise ValueError("Detection mask shape mismatch")
    if any(value not in (False, True) for row in raw for value in row):
        raise ValueError("Detection mask must contain only boolean/binary values")
    return tuple(tuple(bool(value) for value in row) for row in raw)


def _unique_winner(counts):
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or (len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], abs_tol=1e-12)):
        return None
    return ranked[0][0]


def _consensus_evidence(raw, candidate_value):
    """Recheck the original consensus module's declared source records.

    Keep its actionability decision and exact thresholds. Distinct sources,
    unique raw/weighted winners, and chronology cannot be manufactured by
    repeating candidate records or source observations at this boundary.
    """
    if not isinstance(raw, Mapping):
        return None
    numeric = ("score", "peer_source_count", "winning_source_count", "agreement", "margin")
    if not all(_finite(raw.get(key)) for key in numeric):
        return None
    sources, winning = raw["peer_source_count"], raw["winning_source_count"]
    if any(count != int(count) or count < 0 for count in (sources, winning)):
        return None
    reasons = []
    if raw.get("value", candidate_value) != candidate_value:
        reasons.append("candidate_consensus_value_mismatch")
    evidence = raw.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    query = evidence.get("query_source")
    if not isinstance(query, str) or not query.strip() or evidence.get("query_source_vote_excluded") is not True:
        reasons.append("query_source_exclusion_not_declared")
    observations = evidence.get("peer_observations")
    reliability = evidence.get("source_reliability")
    peers, weights = {}, {}
    if not isinstance(observations, list) or not isinstance(reliability, Mapping):
        reasons.append("missing_source_vote_evidence")
    else:
        for observation in observations:
            if not isinstance(observation, Mapping):
                reasons.append("invalid_source_observation")
                continue
            source, vote = observation.get("source"), observation.get("usable_vote")
            if (not isinstance(source, str) or not source.strip() or source == query
                    or (vote is not None and not isinstance(vote, str))):
                reasons.append("invalid_or_query_source_observation")
                continue
            source_evidence = reliability.get(source)
            score = source_evidence.get("score") if isinstance(source_evidence, Mapping) else None
            if not _finite(score) or not 0 < score <= 1:
                reasons.append("invalid_source_weight")
                continue
            exact_values = observation.get("exact_values")
            if vote is not None and (not isinstance(exact_values, list)
                                     or not all(isinstance(v, str) for v in exact_values)
                                     or set(exact_values) != {vote}):
                reasons.append("vote_not_an_unambiguous_exact_observation")
            if source in peers and (peers[source] != vote or weights[source] != float(score)):
                reasons.append("conflicting_duplicate_source_observation")
            peers[source] = vote
            weights[source] = float(score)
    raw_counts = Counter(value for value in peers.values() if value is not None)
    weighted = defaultdict(float)
    for source, vote in peers.items():
        if vote is not None:
            weighted[vote] += weights[source]
    raw_winner, weighted_winner = _unique_winner(raw_counts), _unique_winner(weighted)
    if raw_winner != candidate_value or weighted_winner != candidate_value:
        reasons.append("no_unique_matching_raw_and_weighted_winner")
    actual_supporters = {source for source, vote in peers.items() if vote == candidate_value}
    declared_supporters = evidence.get("supporting_sources")
    if (not isinstance(declared_supporters, list)
            or not all(isinstance(source, str) for source in declared_supporters)
            or len(declared_supporters) != len(set(declared_supporters))
            or set(declared_supporters) != actual_supporters):
        reasons.append("winning_source_declaration_mismatch")
    if int(sources) != len(peers) or int(winning) != len(actual_supporters):
        reasons.append("distinct_source_count_mismatch")
    total_weight = sum(weights.values())
    runner = max((weight for value, weight in weighted.items() if value != candidate_value), default=0.0)
    agreement = weighted.get(candidate_value, 0.0) / total_weight if total_weight else 0.0
    margin = (weighted.get(candidate_value, 0.0) - runner) / total_weight if total_weight else 0.0
    if (not math.isclose(float(raw["agreement"]), agreement, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(float(raw["margin"]), margin, rel_tol=1e-9, abs_tol=1e-12)):
        reasons.append("weighted_support_declaration_mismatch")
    intrinsic = evidence.get("intrinsic_checks")
    intrinsic = intrinsic if isinstance(intrinsic, Mapping) else {}
    relations = intrinsic.get("temporal_relations")
    if (intrinsic.get("format_valid") is not True
            or intrinsic.get("temporal_checks_passed") is not True
            or not isinstance(relations, list)
            or any(not isinstance(relation, Mapping) or relation.get("passed") is not True
                   for relation in relations)
            or evidence.get("abstention_reasons", [])):
        reasons.append("intrinsic_format_or_chronology_conflict")
    return {"actionable": raw.get("actionable") is True,
            "score": float(raw["score"]), "peer_source_count": int(sources),
            "winning_source_count": int(winning),
            "agreement": float(raw["agreement"]), "margin": float(raw["margin"]),
            "unique_raw_winner": raw_winner, "unique_weighted_winner": weighted_winner,
            "recomputed_peer_source_count": len(peers),
            "recomputed_winning_source_count": len(actual_supporters),
            "structural_checks_passed": not reasons,
            "structural_rejection_reasons": sorted(set(reasons))}


def _actionable(evidence, config):
    return bool(evidence and evidence["actionable"]
                and evidence["structural_checks_passed"]
                and evidence["peer_source_count"] >= config.consensus_min_sources
                and evidence["winning_source_count"] >= config.consensus_min_winning_sources
                and evidence["agreement"] + config.tolerance >= config.consensus_min_agreement
                and evidence["margin"] + config.tolerance >= config.consensus_min_margin
                and 0 <= evidence["score"] <= 1
                and 0 <= evidence["agreement"] <= 1
                and 0 <= evidence["margin"] <= 1)


def _valid_peers(values, query, n):
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {int(r) for r in values if isinstance(r, Integral) and not isinstance(r, bool)
            and 0 <= r < n and r != query}


def _merge_candidates(raw_candidates, original, query, n, config):
    merged = {}
    cts_peers = set()
    omitted = Counter()
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise TypeError("Each candidate must be a mapping")
        # This explicit source-specific field is the only CTS vote boundary.
        # Generic peer_rows may also contain source-consensus provenance after
        # caller merges. Gather the union before null/no-op/budget filtering.
        cts_peers.update(_valid_peers(raw.get("cts_peer_rows", []), query, n))
        value = raw.get("value")
        if value is None:
            omitted["null_abstention"] += 1
            continue
        if not isinstance(value, str):
            raise TypeError("Repair values must be exact strings or None for abstention")
        origins_raw = raw.get("origins", [])
        if not isinstance(origins_raw, (list, tuple, set)) or not all(isinstance(o, str) for o in origins_raw):
            raise TypeError("Candidate origins must be a sequence of strings")
        origins = set(origins_raw)
        peers = _valid_peers(raw.get("peer_rows", []), query, n)
        if value == original:
            omitted["original_noop"] += 1
            continue
        feature_raw = raw.get("delta_features")
        features = (tuple(float(v) for v in feature_raw)
                    if isinstance(feature_raw, (list, tuple)) and len(feature_raw) == 4
                    and all(_finite(v) for v in feature_raw) else None)
        gain_raw = raw.get("rule_gain")
        gain = float(gain_raw) if _finite(gain_raw) else None
        consensus = _consensus_evidence(raw.get("consensus"), value)
        errors = bool(raw.get("counterfactual_error"))
        criteria = all(raw.get(key, True) is not False for key in
                       ("criterion_valid", "rule_criterion_valid", "counterfactual_valid"))
        item = {"value": value, "origins": origins, "peer_rows": peers,
                "features": features, "rule_gain": gain,
                "new_rule_violation": raw.get("new_rule_violation") is not False,
                "format_valid": raw.get("candidate_format_valid") is True,
                "counterfactual_error": errors, "criteria_valid": criteria,
                "consensus_records": [consensus] if consensus else [], "records": 1}
        if value not in merged:
            merged[value] = item
            continue
        previous = merged[value]
        previous["records"] += 1
        previous["origins"].update(origins)
        previous["peer_rows"].update(peers)
        previous["features"] = (tuple(min(a, b) for a, b in zip(previous["features"], features))
                                if previous["features"] is not None and features is not None else None)
        previous["rule_gain"] = (min(previous["rule_gain"], gain)
                                 if previous["rule_gain"] is not None and gain is not None else None)
        for key in ("new_rule_violation", "counterfactual_error"):
            previous[key] = previous[key] or item[key]
        for key in ("format_valid", "criteria_valid"):
            previous[key] = previous[key] and item[key]
        if consensus:
            previous["consensus_records"].append(consensus)
    for item in merged.values():
        actionable_records = [e for e in item["consensus_records"] if _actionable(e, config)]
        item["consensus"] = (max(actionable_records,
                                  key=lambda e: (e["score"], e["agreement"], e["margin"], e["peer_source_count"]))
                             if actionable_records else None)
        item["actionable"] = "key_source_consensus" in item["origins"] and bool(item["consensus"])
        item["priority"] = (0 if item["actionable"] else 1 if "seed" in item["origins"]
                            else 2 if "cts" in item["origins"] else 3)
    ordered = sorted(merged.values(), key=lambda item: (
        item["priority"], -(item["consensus"]["score"] if item["consensus"] else 0.0), item["value"]))
    return ordered[:config.candidate_cap], ordered[config.candidate_cap:], cts_peers, dict(omitted)


class _DirtyEvidence:
    def __init__(self, columns, rows, roles, config):
        self.columns, self.rows, self.roles, self.config = columns, rows, roles, config
        self.groups = {}
        self.target_keys = {}
        self.numeric_counts = {}
        self.exact_counts = [Counter(row[j] for row in rows) for j in range(len(columns))]
        self.numeric_forms = {}
        if roles is None:
            return
        product = roles["product_id"]
        name = roles["product_name"]
        org_id, org_name = roles["organization_id"], roles["organization_name"]
        organization_targets = {org_id, org_name, roles["city"], roles["state"]}
        for j in range(len(columns)):
            keys = [(product,)]
            if j in organization_targets:
                keys.extend([(org_id,), (org_name,)])
            keys.extend([(name, org_id), (name, org_name)])
            keys = tuple(dict.fromkeys(key for key in keys if j not in key))
            self.target_keys[j] = keys
            for key in keys:
                if key in self.groups:
                    continue
                groups = defaultdict(list)
                for r, row in enumerate(rows):
                    values = tuple(row[k] for k in key)
                    if all(values):
                        groups[values].append(r)
                self.groups[key] = groups
        for j in roles["measures"]:
            self.numeric_counts[j] = (sum(bool(_BARE_NUMBER.fullmatch(row[j])) for row in rows),
                                      sum(bool(row[j]) for row in rows))
            forms = defaultdict(Counter)
            for row in rows:
                if _BARE_NUMBER.fullmatch(row[j]):
                    forms[Decimal(row[j])][row[j]] += 1
            self.numeric_forms[j] = forms

    def exact_support(self, i, j, candidate):
        return max(self.exact_counts[j].get(candidate, 0) - int(self.rows[i][j] == candidate), 0)

    def canonical_numeric(self, i, j, candidate):
        """A unit edit must preserve amount and match an observed exact mode.

        Numeric equivalence is a witness only. The replacement remains the
        exact proposal string; this method never formats or synthesizes it.
        """
        result = {"valid": False, "candidate_exact_peers": self.exact_support(i, j, candidate),
                  "canonical_exact_value": None, "canonical_agreement": 0.0,
                  "numeric_family_peer_count": 0, "amount_preserved": False}
        if self.roles is None or j not in self.numeric_forms:
            return result
        unit = _UNITS.get(_column_key(self.columns[j]))
        if unit is None or not _BARE_NUMBER.fullmatch(candidate):
            return result
        match = re.fullmatch(r"(?P<number>" + _NUMBER + r")[ \t]*" + unit,
                             self.rows[i][j], flags=re.IGNORECASE)
        if not match:
            return result
        try:
            amount, proposed = Decimal(match.group("number")), Decimal(candidate)
        except InvalidOperation:
            return result
        if not amount.is_finite() or not proposed.is_finite() or amount != proposed:
            return result
        result["amount_preserved"] = True
        forms = self.numeric_forms[j].get(amount, Counter()).copy()
        if _BARE_NUMBER.fullmatch(self.rows[i][j]):
            forms[self.rows[i][j]] -= 1
        forms += Counter()
        ranked = forms.most_common(2)
        total = sum(forms.values())
        if not ranked:
            return result
        mode, support = ranked[0]
        agreement = support / total
        result.update(canonical_exact_value=mode, canonical_agreement=agreement,
                      numeric_family_peer_count=total)
        result["valid"] = (mode == candidate and support >= self.config.numeric_min_exact_peers
                           and (len(ranked) == 1 or support > ranked[1][1])
                           and agreement >= self.config.numeric_min_canonical_agreement)
        return result

    def fd(self, i, j):
        records = []
        if self.roles is not None:
            row = self.rows[i]
            for keys in self.target_keys[j]:
                key = tuple(row[k] for k in keys)
                if not all(key):
                    continue
                peers = [r for r in self.groups[keys].get(key, ()) if r != i]
                if len(peers) < self.config.fd_min_peers:
                    continue
                counts = Counter(self.rows[r][j] for r in peers)
                records.append({"determinants": [self.columns[k] for k in keys],
                                "peer_rows": peers, "peer_count": len(peers),
                                "values": [{"value": value, "count": count}
                                           for value, count in sorted(counts.items())],
                                "unanimous": len(counts) == 1,
                                "value": next(iter(counts)) if len(counts) == 1 else None})
        unanimous = {record["value"] for record in records if record["unanimous"]}
        conflict = any(not record["unanimous"] for record in records) or len(unanimous) > 1
        return {"conflict": conflict, "value": next(iter(unanimous)) if len(unanimous) == 1 and not conflict else None,
                "available": bool(records), "evidence": records}

    def unit(self, i, j, candidate):
        result = {"valid": False, "bare_numeric_peers": 0, "nonempty_peers": 0,
                  "bare_numeric_fraction": 0.0}
        if self.roles is None or j not in self.numeric_counts:
            return result
        unit = _UNITS.get(_column_key(self.columns[j]))
        if unit is None:
            return result
        original = self.rows[i][j]
        match = re.fullmatch(r"(?P<number>" + _NUMBER + r")[ \t]*" + unit, original, flags=re.IGNORECASE)
        numeric, nonempty = self.numeric_counts[j]
        numeric -= int(bool(_BARE_NUMBER.fullmatch(original)))
        nonempty -= int(bool(original))
        fraction = numeric / nonempty if nonempty else 0.0
        result.update(bare_numeric_peers=numeric, nonempty_peers=nonempty,
                      bare_numeric_fraction=fraction)
        result["valid"] = bool(match and candidate == match.group("number")
                               and numeric >= self.config.unit_min_numeric_peers
                               and fraction >= self.config.unit_min_numeric_fraction)
        return result

    def cts(self, i, j, candidate, peers):
        # Peers are already deduplicated, range checked, and query excluded.
        counts = Counter(self.rows[r][j] for r in peers)
        total = len(peers)
        support = counts.get(candidate, 0)
        other = max((count for value, count in counts.items() if value != candidate), default=0)
        agreement = support / total if total else 0.0
        margin = (support - other) / total if total else 0.0
        return {"valid": (total >= self.config.cts_min_peers and support > other
                          and agreement >= self.config.cts_min_agreement
                          and margin >= self.config.cts_min_margin),
                "peer_rows": sorted(peers), "peer_count": total,
                "support": support, "agreement": agreement, "margin": margin}


def _review(item, i, j, fd, evidence, cts_peers, config):
    reasons = []
    features, gain = item["features"], item["rule_gain"]
    score = sum(w * v for w, v in zip(config.feature_weights, features)) if features is not None else None
    if features is None or gain is None:
        reasons.append("invalid_counterfactual_evidence")
    if item["counterfactual_error"]:
        reasons.append("counterfactual_error")
    if not item["criteria_valid"]:
        reasons.append("invalid_rule_criterion")
    if not item["format_valid"]:
        reasons.append("invalid_format")
    if features is not None:
        if score < -0.50:
            reasons.append("weighted_consistency_loss")
    if fd["conflict"]:
        reasons.append("conflicting_dirty_fd_evidence")
    elif fd["available"] and fd["value"] != item["value"]:
        reasons.append("contradicts_dirty_fd_consensus")
    unit = evidence.unit(i, j, item["value"])
    cts = evidence.cts(i, j, item["value"], cts_peers)
    fd_valid = bool(fd["available"] and not fd["conflict"] and fd["value"] == item["value"])
    strict_rule = gain is not None and gain > config.tolerance
    witnesses = (["dirty_fd_consensus"] if fd_valid else [])
    witnesses += [name for name, yes in (("literal_unit_stripping", unit["valid"]),
                                       ("distinct_cts_agreement", cts["valid"]),
                                       ("strict_rule_improvement", strict_rule)) if yes]
    exact_peers = evidence.exact_support(i, j, item["value"])
    numeric = evidence.canonical_numeric(i, j, item["value"])
    if numeric["valid"]:
        witnesses.append("observed_numeric_canonical_form")
    if evidence.roles is not None and not witnesses:
        reasons.append("catalog_requires_strict_witness")
    if evidence.roles is not None:
        # No model-rule gain, generic retrieved majority, seed provenance or
        # unchanged grammar can replace deterministic entity/amount evidence.
        if not (fd_valid or numeric["valid"]):
            reasons.append("catalog_requires_entity_fd_or_observed_numeric_canonical_form")
        if exact_peers < config.catalog_min_exact_peers:
            reasons.append("catalog_requires_exact_dirty_peer_support")
    if not ("seed" in item["origins"] or item["actionable"] or witnesses
            or (score is not None and score > config.tolerance)):
        reasons.append("no_actionable_repair_evidence")
    if item["priority"] == 3:
        reasons.append("unsupported_origin")
    return {"value": item["value"], "origins": sorted(item["origins"]),
            "input_record_count": item["records"], "peer_rows": sorted(item["peer_rows"]),
            "delta_features": list(features) if features is not None else None,
            "weighted_delta": score, "rule_gain": gain,
            "actionable_consensus": item["actionable"], "consensus": item["consensus"],
            "consensus_evidence_records": item["consensus_records"],
            "strict_witnesses": witnesses, "unit_evidence": unit, "cts_evidence": cts,
            "catalog_gate": {"applied": evidence.roles is not None, "candidate_exact_peer_count": exact_peers,
                             "entity_fd_valid": fd_valid, "numeric_canonical_evidence": numeric},
            "eligible": not reasons, "rejection_reasons": reasons}


def select_repairs(table, detection_mask, candidates, config=None) -> dict[str, Any]:
    """Select exact-string edits exclusively within a fixed detection mask.

    ``candidates`` maps ``(row, column)`` to candidate dictionary sequences.
    The returned decisions and diagnostics are JSON serializable. The caller
    must apply its combined whole-row guard before committing returned edits.
    """
    config = _config(config)
    columns = tuple(table.columns)
    rows = tuple(tuple(row) for row in table.rows)
    n, m = len(rows), len(columns)
    if (not all(isinstance(c, str) for c in columns)
            or any(len(row) != m or not all(isinstance(v, str) for v in row) for row in rows)):
        raise ValueError("Dirty table must have rectangular exact-string rows and column names")
    mask = _mask_snapshot(detection_mask, n, m)
    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a coordinate mapping")
    for coordinate in candidates:
        if (not isinstance(coordinate, tuple) or len(coordinate) != 2
                or any(not isinstance(v, Integral) or isinstance(v, bool) for v in coordinate)
                or not 0 <= coordinate[0] < n or not 0 <= coordinate[1] < m):
            raise ValueError(f"Invalid candidate coordinate: {coordinate!r}")
    roles = _catalog_roles(columns)
    evidence = _DirtyEvidence(columns, rows, roles, config)
    repairs, decisions = {}, []
    counts = Counter()
    counts["candidate_cells_outside_detection"] = sum(not mask[i][j] for i, j in candidates)
    for i in range(n):
        for j in range(m):
            if not mask[i][j]:
                continue
            counts["detection_positive_cells"] += 1
            items, pruned, cts_peers, omitted = _merge_candidates(
                candidates.get((i, j), []), rows[i][j], i, n, config)
            fd = evidence.fd(i, j)
            reviews = [_review(item, i, j, fd, evidence, cts_peers, config) for item in items]
            eligible = [(item, review) for item, review in zip(items, reviews) if review["eligible"]]
            eligible.sort(key=lambda pair: (-pair[1]["weighted_delta"], pair[0]["priority"],
                                           -(pair[0]["consensus"]["score"] if pair[0]["consensus"] else 0.0),
                                           pair[0]["value"]))
            chosen = eligible[0] if eligible else None
            decision = {"row": i, "column": j, "original": rows[i][j],
                        "detection_positive": True, "selected_value": None,
                        "status": "abstained", "selection_reason": "no_eligible_candidate",
                        "candidate_reviews": reviews,
                        "budget_pruned_values": [item["value"] for item in pruned],
                        "omitted_candidates": omitted, "fd_evidence": fd}
            if chosen:
                item, review = chosen
                repairs[i, j] = item["value"]
                decision.update(selected_value=item["value"], status="selected_pending_row_guard",
                                selection_reason=("noisy_seeded_carry" if "seed" in item["origins"]
                                                  and not item["actionable"] and not review["strict_witnesses"]
                                                  else "guarded_candidate_selection"))
                counts["selected_cells"] += 1
                counts[decision["selection_reason"]] += 1
            else:
                counts["abstained_cells"] += 1
            counts["budget_pruned_strings"] += len(pruned)
            counts["null_abstention_records"] += omitted.get("null_abstention", 0)
            decisions.append(decision)
    role_names = ({key: [columns[j] for j in value] if isinstance(value, list) else columns[value]
                   for key, value in roles.items()} if roles is not None else None)
    diagnostics = {"method": "unified_loop_v5", "config": asdict(config),
                   "catalog_safeguard": roles is not None, "catalog_roles": role_names,
                   "requires_combined_row_guard": True, "detection_mask_modified": False,
                   "counts": dict(counts)}
    return {"repairs": repairs, "decisions": decisions, "diagnostics": diagnostics}
