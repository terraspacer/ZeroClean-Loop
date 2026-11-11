"""Intrinsic, dirty-only multi-source proposals; detection is a separate step.

No file access, external metadata, learned verifier, or dataset-name dispatch is
used here. Scores describe evidence strength, not calibrated correctness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
import math
import re
from types import MappingProxyType


DEFAULT_CONFIG = MappingProxyType({
    "min_peer_sources": 2,
    "min_winning_sources": 2,
    "min_agreement": 2 / 3,
    "min_margin": 0.20,
    "min_repeated_entities": 2,
    "min_multisource_coverage": 0.50,
    "min_slot_uniqueness": 0.80,
    "max_key_columns": 2,
    "reliability_prior_strength": 3.0,
    "format_weight": 0.35,
    "peer_weight": 0.50,
    "consistency_weight": 0.15,
    "reliability_floor": 0.20,
})
POLICY_VERSION = "intrinsic-key-source-consensus-v1"

_SOURCE_NAMES = {"src", "source", "sourceid", "datasource", "datasourceid",
                 "provider", "providerid", "reporter", "reporterid", "feed",
                 "feedid", "origin", "system", "systemid"}
_SOURCE_PRIORITY = {"src": 2, "source": 2, "sourceid": 2,
                    "datasource": 2, "datasourceid": 2}
_CLOCK = re.compile(
    r"^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?\s*"
    r"(a\.?m\.?|p\.?m\.?)?\s*(Z|UTC|GMT|[+-]\d{2}:\d{2})?$", re.I)
_ISO_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d{1,6})?)?(?:[Zz]|[+-]\d{2}:\d{2})?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class _Temporal:
    kind: str
    absolute: datetime | None = None


def _tokens(name):
    expanded = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return tuple(re.findall(r"[a-z0-9]+", expanded.lower()))


def _parse_temporal(raw):
    """Parse for validation only. Never return a replacement string."""
    value = raw.strip()
    if _ISO_DATETIME.fullmatch(value):
        offset = re.search(r"[+-]\d{2}:\d{2}$", value)
        if offset is not None:
            hours, minutes = map(int, offset.group()[1:].split(":"))
            # fromisoformat normalizes malformed offsets such as +01:60.
            # Validate the literal fields before parsing or using chronology.
            if hours > 23 or minutes > 59:
                return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError:
            return None
        # RFC 3339 uses -00:00 for an unknown local offset. It supplies no
        # affirmative timezone evidence for a chronology comparison.
        absolute = parsed if parsed.tzinfo is not None and not value.endswith("-00:00") else None
        return _Temporal("dated_timestamp", absolute)
    if _ISO_DATE.fullmatch(value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None
        return _Temporal("date")
    match = _CLOCK.fullmatch(value)
    if not match:
        return None
    hour, minute, second = (int(match.group(i) or 0) for i in (1, 2, 3))
    meridiem, zone = match.group(5), match.group(6)
    if minute > 59 or second > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23):
        return None
    if zone and zone[0] in "+-":
        hours, minutes = map(int, zone[1:].split(":"))
        if hours > 23 or minutes > 59:
            return None
    return _Temporal("clock")


def _time_hint(name):
    tokens = _tokens(name)
    return bool(set(tokens) & {"time", "timestamp", "datetime", "date"}) or tokens[-1:] == ("at",)


def _key_hint(name):
    return int(bool(set(_tokens(name)) & {"id", "identifier", "key", "code", "number"}))


def _temporal_pairs(columns, targets):
    starts, ends = defaultdict(list), defaultdict(list)
    begin = {"start", "begin", "began", "departure", "depart", "dep"}
    end = {"end", "finish", "finished", "arrival", "arrive", "arr"}
    for j in targets:
        tokens = _tokens(columns[j])
        role = set(tokens) & (begin | end)
        if len(role) != 1:
            continue
        marker = next(iter(role))
        stem = tuple(token for token in tokens if token != marker)
        (starts if marker in begin else ends)[stem].append(j)
    # Ambiguous many-to-many schema relations supply no chronology constraint.
    return tuple((starts[stem][0], ends[stem][0]) for stem in sorted(starts.keys() & ends.keys())
                 if len(starts[stem]) == len(ends[stem]) == 1)


def _intrinsic(rows, row_index, target, parsed, pairs):
    checked, skipped = [], []
    row = rows[row_index]
    for start, end in pairs:
        if target not in (start, end):
            continue
        other = end if target == start else start
        companion = _parse_temporal(row[other])
        if companion is None:
            skipped.append("companion_missing_or_malformed")
            continue
        if parsed.absolute is None or companion.absolute is None:
            skipped.append("date_and_explicit_timezone_required_for_ordering")
            continue
        first, last = (parsed.absolute, companion.absolute) if target == start else (companion.absolute, parsed.absolute)
        checked.append({"start_column_index": start, "end_column_index": end,
                        "passed": first <= last,
                        "basis": "dated_timestamps_compared_with_explicit_UTC_offsets"})
    return {"format_valid": True, "parsed_kind": parsed.kind,
            "temporal_relations": checked, "skipped_temporal_relations": sorted(set(skipped)),
            "temporal_checks_passed": all(item["passed"] for item in checked),
            "ordering_assumed_for_clock_only_values": False}


def _config(config):
    result = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("Unknown consensus config keys: " + ", ".join(sorted(unknown)))
        result.update(config)
    integer_keys = {"min_peer_sources", "min_winning_sources", "min_repeated_entities", "max_key_columns"}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid consensus config value: {key}")
        if key in integer_keys and (int(value) != value or value < 1):
            raise ValueError(f"Expected positive integer for {key}")
    if result["max_key_columns"] > 2:
        raise ValueError("At most two key columns are supported")
    for key in ("min_agreement", "min_margin", "min_multisource_coverage", "min_slot_uniqueness", "reliability_floor"):
        if not 0 <= result[key] <= 1:
            raise ValueError(f"Expected [0, 1] for {key}")
    if result["reliability_prior_strength"] <= 0:
        raise ValueError("reliability_prior_strength must be positive")
    weights = [result[key] for key in ("format_weight", "peer_weight", "consistency_weight")]
    if min(weights) < 0 or not math.isclose(sum(weights), 1.0):
        raise ValueError("Reliability component weights must be nonnegative and sum to one")
    return result


def _key_profile(rows, source, keys, targets, cfg):
    groups = defaultdict(list)
    slots = set()
    observations = set()
    for i, row in enumerate(rows):
        if not row[source].strip() or any(not row[k].strip() for k in keys):
            continue
        key = tuple(row[k] for k in keys)
        groups[key].append(i)
        slots.add((key, row[source]))
        observations.add((key, row[source], tuple(row[j] for j in targets)))
    repeated = {key for key, indices in groups.items() if len({rows[i][source] for i in indices}) >= 2}
    repeated_slots = sum(key in repeated for key, _ in slots)
    coverage = repeated_slots / max(len(slots), 1)
    uniqueness = len(slots) / max(len(observations), 1)
    if (len(repeated) < cfg["min_repeated_entities"] or coverage < cfg["min_multisource_coverage"]
            or uniqueness < cfg["min_slot_uniqueness"]):
        return None
    return {"groups": dict(groups), "repeated_entity_count": len(repeated),
            "multisource_coverage": coverage, "slot_uniqueness": uniqueness,
            "entity_count": len(groups)}


def _discover_roles(columns, rows, cfg):
    # Discovery precedes key/source selection. Count complete exact dirty
    # observations once, preserving identical values from distinct contexts.
    discovery_rows = tuple(dict.fromkeys(rows))
    target_profiles = []
    for j, column in enumerate(columns):
        observed = [row[j] for row in discovery_rows if row[j].strip()]
        valid = sum(_parse_temporal(value) is not None for value in observed)
        if valid and (_time_hint(column) or (len(observed) >= 3 and valid / len(observed) >= .8)):
            target_profiles.append({"index": j, "name": column, "nonempty_count": len(observed),
                                    "parseable_count": valid, "schema_hint": _time_hint(column)})
    targets = tuple(profile["index"] for profile in target_profiles)
    if not targets:
        return None, {"reason": "no_temporal_columns", "temporal_columns": []}
    sources = [j for j, name in enumerate(columns)
               if "".join(_tokens(name)) in _SOURCE_NAMES and j not in targets
               and len({row[j] for row in rows if row[j].strip()}) >= cfg["min_peer_sources"] + 1]
    if not sources:
        return None, {"reason": "no_schema_supported_multisource_column", "temporal_columns": target_profiles}
    choices = []
    for source in sources:
        possible = [j for j in range(len(columns)) if j != source and j not in targets and j not in sources]
        for width in range(1, int(cfg["max_key_columns"]) + 1):
            found = []
            for keys in combinations(possible, width):
                profile = _key_profile(rows, source, keys, targets, cfg)
                if profile is None:
                    continue
                rank = (_SOURCE_PRIORITY.get("".join(_tokens(columns[source])), 1),
                        sum(_key_hint(columns[k]) for k in keys), profile["slot_uniqueness"],
                        profile["multisource_coverage"], profile["repeated_entity_count"], -width)
                found.append((rank, source, keys, profile))
            choices.extend(found)
            # A supported simple entity key takes precedence over refinements.
            if found:
                break
    if not choices:
        return None, {"reason": "no_repeated_multisource_entity_key", "temporal_columns": target_profiles,
                      "source_columns_considered": [columns[j] for j in sources]}
    choices.sort(key=lambda item: (tuple(-number for number in item[0]),
                                  columns[item[1]], tuple(columns[j] for j in item[2])))
    _, source, keys, profile = choices[0]
    selected = {"source": source, "keys": keys, "targets": targets, "groups": profile["groups"]}
    diagnostic = {"source_column": {"index": source, "name": columns[source]},
                  "key_columns": [{"index": k, "name": columns[k]} for k in keys],
                  "temporal_columns": target_profiles,
                  "key_discovery": {key: value for key, value in profile.items() if key != "groups"},
                  "supported_role_choices": len(choices), "dataset_name_used": False}
    return selected, diagnostic


def _entity_observations(rows, groups, source, target, pairs):
    entities = {}
    for key, indices in groups.items():
        by_source = defaultdict(list)
        for i in indices:
            by_source[rows[i][source]].append(i)
        observations = {}
        for label, source_rows in by_source.items():
            values = sorted({rows[i][target] for i in source_rows if rows[i][target].strip()})
            if not values:
                continue
            parsed = {value: _parse_temporal(value) for value in values}
            checks = [_intrinsic(rows, i, target, parsed[rows[i][target]], pairs)
                      for i in source_rows if rows[i][target].strip() and parsed[rows[i][target]] is not None]
            relations = [relation for check in checks for relation in check["temporal_relations"]]
            consistent = None if not relations else float(all(relation["passed"] for relation in relations))
            value = values[0] if len(values) == 1 and parsed[values[0]] is not None and consistent != 0 else None
            observations[label] = {"values": values, "vote": value,
                                   "format": sum(item is not None for item in parsed.values()) / len(values),
                                   "consistency": consistent}
        entities[key] = observations
    return entities


def _unique_winner(counts):
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or (len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], abs_tol=1e-12)):
        return None
    return ranked[0][0]


def _training_statistics(entities, excluded_source, cfg):
    """Cap every training signal at one contribution per source/entity."""
    totals = defaultdict(lambda: [0.0] * 6)
    entity_stats = {}
    for key, observations in entities.items():
        stats = {}
        for label, observation in observations.items():
            if label == excluded_source:
                continue
            component = [observation["format"], 1.0, 0.0, 0.0, 0.0, 0.0]
            if observation["consistency"] is not None:
                component[4:6] = [observation["consistency"], 1.0]
            if observation["vote"] is not None:
                peer = {other: item for other, item in observations.items()
                        if other not in (label, excluded_source)}
                counts = Counter(item["vote"] for item in peer.values() if item["vote"] is not None)
                mode = _unique_winner(counts)
                # Invalid or ambiguous nonempty peers remain in the denominator.
                if mode is not None and counts[mode] >= cfg["min_winning_sources"] and counts[mode] / max(len(peer), 1) >= cfg["min_agreement"]:
                    component[2:4] = [float(observation["vote"] == mode), 1.0]
            stats[label] = component
            totals[label] = [a + b for a, b in zip(totals[label], component)]
        entity_stats[key] = stats
    return dict(totals), entity_stats


def _reliability(totals, query_entity_stats, label, cfg):
    # Subtract every observation of this entity, including other temporal context.
    aggregate = totals.get(label, [0.0] * 6)
    held_out = query_entity_stats.get(label, [0.0] * 6)
    remaining = [max(0.0, total - held) for total, held in zip(aggregate, held_out)]
    prior = cfg["reliability_prior_strength"]
    components = [(remaining[i] + .5 * prior) / (remaining[i + 1] + prior) for i in (0, 2, 4)]
    value = max(cfg["reliability_floor"], sum(a * b for a, b in zip(
        components, (cfg["format_weight"], cfg["peer_weight"], cfg["consistency_weight"]))))
    return {"score": min(1.0, value), "format_score": components[0], "peer_agreement_score": components[1],
            "cross_column_consistency_score": components[2],
            "training_entity_counts": {"format": int(remaining[1]), "peer_agreement": int(remaining[3]),
                                       "cross_column_consistency": int(remaining[5])},
            "query_entity_excluded": True, "query_source_excluded_from_peer_references": True,
            "calibrated_correctness": False}


def discover_consensus(table, config=None):
    """Return exact observed peer alternatives, conditional on separate detection.

    ``table`` needs only ``columns`` and ``rows``. Coordinates are zero based.
    The ``candidates`` mapping has tuple keys; ``diagnostics`` is JSON serializable.
    A consumer must require ``actionable`` and an independently detected cell.
    """
    cfg = _config(config)
    columns, rows = tuple(table.columns), tuple(tuple(row) for row in table.rows)
    if (len(set(columns)) != len(columns) or any(not isinstance(name, str) for name in columns)
            or any(len(row) != len(columns) or any(not isinstance(value, str) for value in row) for row in rows)):
        raise ValueError("Consensus requires unique string columns and rectangular string rows")
    diagnostics = {"policy_version": POLICY_VERSION, "config": cfg,
                   "score_semantics": "heuristic_evidence_strength_not_calibrated_correctness",
                   "proposals_are_conditional_on_separate_detection": True,
                   "exact_peer_strings_only": True, "source_votes_capped": True,
                   "temporal_role_discovery_observation_unit": "distinct_exact_dirty_row"}
    roles, discovery = _discover_roles(columns, rows, cfg)
    diagnostics.update(discovery)
    if roles is None:
        diagnostics["status"] = "not_applicable"
        return {"candidates": {}, "diagnostics": diagnostics}
    source, keys, targets, groups = (roles[key] for key in ("source", "keys", "targets", "groups"))
    pairs = _temporal_pairs(columns, targets)
    diagnostics["safe_temporal_schema_pairs"] = [[columns[a], columns[b]] for a, b in pairs]
    diagnostics["status"] = "applicable"
    candidates, reason_counts = {}, Counter()
    for target in targets:
        entities = _entity_observations(rows, groups, source, target, pairs)
        training_cache = {}
        for key, indices in groups.items():
            observations = entities[key]
            for query_source in sorted({rows[i][source] for i in indices}):
                if query_source not in training_cache:
                    training_cache[query_source] = _training_statistics(entities, query_source, cfg)
                totals, entity_stats = training_cache[query_source]
                peers = {label: item for label, item in observations.items() if label != query_source}
                reliability = {label: _reliability(totals, entity_stats[key], label, cfg) for label in peers}
                raw_counts = Counter(item["vote"] for item in peers.values() if item["vote"] is not None)
                weighted = defaultdict(float)
                for label, item in peers.items():
                    if item["vote"] is not None:
                        weighted[item["vote"]] += reliability[label]["score"]
                winner = _unique_winner(raw_counts)
                if winner is None or winner != _unique_winner(weighted):
                    reason_counts["no_unique_source_count_and_weight_winner"] += sum(rows[i][source] == query_source for i in indices)
                    continue
                total_weight = sum(item["score"] for item in reliability.values())
                agreement = weighted[winner] / total_weight
                runner_weight = max((weight for value, weight in weighted.items() if value != winner), default=0.0)
                margin = (weighted[winner] - runner_weight) / total_weight
                supporting = sorted(label for label, item in peers.items() if item["vote"] == winner)
                support = raw_counts[winner]
                score = min(1.0, .7 * agreement + .2 * margin + .1 * min(support / 3, 1.0))
                for i in indices:
                    if rows[i][source] != query_source or winner == rows[i][target]:
                        continue
                    intrinsic = _intrinsic(rows, i, target, _parse_temporal(winner), pairs)
                    failures = []
                    if len(peers) < cfg["min_peer_sources"]:
                        failures.append("insufficient_peer_sources")
                    if support < cfg["min_winning_sources"]:
                        failures.append("insufficient_winning_sources")
                    if agreement + 1e-12 < cfg["min_agreement"]:
                        failures.append("insufficient_weighted_agreement")
                    if margin + 1e-12 < cfg["min_margin"]:
                        failures.append("insufficient_weighted_margin")
                    if not intrinsic["temporal_checks_passed"]:
                        failures.append("explicit_timestamp_chronology_conflict")
                    why = ["Exact observed peer string passes a strict clock, ISO date, or ISO timestamp parser."]
                    if intrinsic["temporal_relations"]:
                        why.append("Every evaluated dated start/end relation passes after explicit timezone offsets are applied."
                                   if intrinsic["temporal_checks_passed"] else "An evaluated dated start/end relation fails.")
                    else:
                        why.append("No safe chronology comparison was available; clock ordering, timezone, and overnight direction were not assumed.")
                    candidate = {"value": winner, "score": score, "origin": "key_source_consensus",
                                 "actionable": not failures, "peer_source_count": len(peers),
                                 "winning_source_count": support, "agreement": agreement, "margin": margin,
                                 "source_column": columns[source], "key_columns": [columns[k] for k in keys],
                                 "target_column": columns[target],
                                 "evidence": {"policy_version": POLICY_VERSION, "original_value": rows[i][target],
                                              "query_source": query_source, "entity_key": list(key),
                                              "source_column_index": source, "key_column_indices": list(keys),
                                              "target_column_index": target, "supporting_sources": supporting,
                                              "source_reliability": {label: reliability[label] for label in sorted(reliability)},
                                              "peer_observations": [{"source": label, "exact_values": peers[label]["values"],
                                                                     "usable_vote": peers[label]["vote"]} for label in sorted(peers)],
                                              "valid_peer_source_count": sum(item["vote"] is not None for item in peers.values()),
                                              "query_source_vote_excluded": True, "query_entity_excluded_from_reliability": True,
                                              "intrinsic_checks": intrinsic, "why_intrinsic_checks_passed": why,
                                              "abstention_reasons": failures,
                                              "score_is_calibrated_correctness": False}}
                    candidates[(i, target)] = [candidate]
                    reason_counts["actionable_proposal" if not failures else "conditional_proposal_below_policy"] += 1
    diagnostics["proposal_reason_counts"] = dict(sorted(reason_counts.items()))
    diagnostics["candidate_cell_count"] = len(candidates)
    diagnostics["max_distinct_nonoriginal_candidates_per_cell"] = 1
    return {"candidates": candidates, "diagnostics": diagnostics}
