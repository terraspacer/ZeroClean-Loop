"""Dirty-only inherited gating and independent negative promotions.

The caller attests that ``rule_violations`` contains only validated, nonempty,
target-specific rule violations. See protocol.md for the frozen policy.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re
import unicodedata
from datetime import datetime
from collections import defaultdict

import numpy as np

from zeroclean.data import DirtyTable


@dataclass(frozen=True)
class DetectionConfig:
    source_positive_fraction_threshold: float = 0.40
    demoted_score: float = 0.49
    min_comparison_peers: int = 8
    min_domain_observations: int = 30
    max_domain_distinct_fraction: float = 0.20
    min_core_value_count: int = 3
    min_core_mass_fraction: float = 0.90
    max_rare_peer_count: int = 1
    max_rare_peer_fraction: float = 0.02
    min_structured_fraction: float = 0.90
    min_dominant_format_fraction: float = 0.90
    identifier_unique_fraction: float = 0.60
    promoted_score: float = 0.75
    missing_min_observations: int = 30
    max_missing_fraction: float = 0.02
    type_min_fraction: float = 0.95
    canonical_min_support: int = 5
    canonical_min_agreement: float = 0.90
    context_min_peers: int = 3
    context_min_agreement: float = 0.95
    context_global_min_agreement: float = 0.97
    context_min_coverage: float = 0.15
    context_min_groups: int = 2
    context_min_gain_over_marginal: float = 0.10

    def __post_init__(self):
        for name in (
            "source_positive_fraction_threshold", "max_domain_distinct_fraction",
            "min_core_mass_fraction", "max_rare_peer_fraction",
            "min_structured_fraction", "min_dominant_format_fraction",
            "identifier_unique_fraction",
            "max_missing_fraction", "type_min_fraction", "canonical_min_agreement",
            "context_min_agreement", "context_global_min_agreement", "context_min_coverage",
            "context_min_gain_over_marginal",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not isinstance(self.demoted_score, (int, float)) or isinstance(self.demoted_score, bool) or not np.isfinite(self.demoted_score) or not 0 <= self.demoted_score <= 0.5:
            raise ValueError("demoted_score must be finite and in [0, 0.5]")
        if not isinstance(self.promoted_score, (int, float)) or isinstance(self.promoted_score, bool) or not np.isfinite(self.promoted_score) or not 0.5 < self.promoted_score <= 1:
            raise ValueError("promoted_score must be finite and in (0.5, 1]")
        for name in ("min_comparison_peers", "min_domain_observations", "min_core_value_count",
                     "missing_min_observations", "canonical_min_support", "context_min_peers", "context_min_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_rare_peer_count, bool) or not isinstance(self.max_rare_peer_count, int) or self.max_rare_peer_count < 0:
            raise ValueError("max_rare_peer_count must be a nonnegative integer")


_TEXT_TOKENS = frozenset({
    "name", "title", "description", "address", "author", "authors", "actor",
    "actors", "cast", "director", "creator", "text", "comment", "comments",
    "note", "notes", "abstract", "summary", "caption", "biography", "content",
    "message",
})
_IDENTIFIER_TOKENS = frozenset({"id", "identifier", "uuid", "guid", "key", "code"})
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_EMPTY_EDGE_CASE = re.compile(
    r"\b(?:zero[ -]predicate|empty[ -]predicate\s+fallback|no\s+conditioned\s+predicates)\b",
    re.IGNORECASE,
)


def _schema(column):
    # Split acronym boundaries too: UUIDCode -> UUID Code.
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", column)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
    tokens = set(re.findall(r"[a-z0-9]+", spaced.lower()))
    return {
        "tokens": sorted(tokens),
        "free_text": bool(tokens & _TEXT_TOKENS),
        "identifier": bool(tokens & _IDENTIFIER_TOKENS),
    }


def _format(value):
    value = value.strip()
    if not value:
        return ""
    if _NUMBER.fullmatch(value):
        return "numeric"
    parts = []
    previous = None
    for character in value:
        token = "A" if character.isalpha() else "9" if character.isdigit() else " " if character.isspace() else character
        if token != previous or token not in ("A", "9", " "):
            parts.append(token)
        previous = token
    return "".join(parts)


def _structured(value, identifier):
    value = value.strip()
    return bool(value) and (
        any(character.isdigit() for character in value)
        or "@" in value or "://" in value
        or (identifier and not any(character.isspace() for character in value))
    )


def _fallback_metadata(metadata, column):
    """Read only documented per-column source classifier diagnostics."""
    columns = metadata.get("classifier_columns") if isinstance(metadata, Mapping) else None
    if not isinstance(columns, Mapping):
        return {"status": "missing_classifier_columns", "empty_predicate_fallback": None,
                "evidence": [], "malformed_fields": [], "conflicting_evidence": False,
                "constant_positive_classifier": None}
    info = columns.get(column)
    if not isinstance(info, Mapping):
        return {"status": "missing_column_diagnostics" if info is None else "malformed_column_diagnostics",
                "empty_predicate_fallback": None, "evidence": [], "malformed_fields": [],
                "conflicting_evidence": False, "constant_positive_classifier": None}
    positive, negative, malformed = [], [], []
    for key in ("empty_predicate_fallback", "source_empty_predicate_fallback"):
        if key in info:
            if type(info[key]) is bool:
                (positive if info[key] else negative).append(key)
            else:
                malformed.append(key)
    key = "clean_predicate_count_after_filter"
    if key in info:
        count = info[key]
        if type(count) is int and count >= 0:
            (positive if count == 0 else negative).append(f"{key}={'zero' if count == 0 else 'positive'}")
        else:
            malformed.append(key)
    if "source_edge_case" in info:
        edge = info["source_edge_case"]
        if isinstance(edge, str):
            if _EMPTY_EDGE_CASE.search(edge):
                positive.append("source_edge_case=empty_predicate_description")
        else:
            malformed.append("source_edge_case")
    known = True if positive else False if negative else None
    classes = info.get("classes")
    constant_positive = (
        info.get("score_kind") == "constant_pseudo_label_class"
        and isinstance(classes, (list, tuple)) and len(classes) == 1
        and type(classes[0]) is int and classes[0] == 1
    )
    return {
        "status": "known" if known is not None else "unknown_incomplete_column_diagnostics",
        "empty_predicate_fallback": known,
        "evidence": positive + negative,
        "malformed_fields": malformed,
        "conflicting_evidence": bool(positive and negative),
        "constant_positive_classifier": bool(constant_positive),
    }


class _DirtySupport:
    """One original dirty column; every query excludes its own observation."""

    def __init__(self, values, column, config):
        self.values = values
        self.config = config
        self.schema = _schema(column)
        self.signatures = [_format(value) for value in values]
        self.structured = [_structured(value, self.schema["identifier"]) for value in values]
        self.counts = Counter(value for value in values if value.strip())
        self.formats = Counter(signature for signature in self.signatures if signature)
        self.nonempty = sum(self.counts.values())
        self.structured_count = sum(self.structured)
        self.core_mass = sum(count for count in self.counts.values() if count >= config.min_core_value_count)
        self.top_formats = self.formats.most_common(2)

    def describe(self):
        return {
            "schema": self.schema,
            "observations": len(self.values),
            "nonempty_observations": self.nonempty,
            "distinct_nonempty_values": len(self.counts),
            "distinct_nonempty_formats": len(self.formats),
            "structured_fraction": self.structured_count / self.nonempty if self.nonempty else 0.0,
            "distinct_fraction": len(self.counts) / self.nonempty if self.nonempty else 0.0,
            "dominant_format": self.top_formats[0][0] if self.top_formats else None,
            "dominant_format_fraction": self.top_formats[0][1] / self.nonempty if self.nonempty else 0.0,
        }

    def cell(self, row):
        config = self.config
        value, signature = self.values[row], self.signatures[row]
        nonempty = bool(signature)
        peers = self.nonempty - int(nonempty)
        count = self.counts.get(value, 0) if nonempty else 0
        value_peers = max(count - int(nonempty), 0)
        format_peers = self.formats.get(signature, 0) - int(nonempty) if nonempty else 0
        dominant_format, dominant_count = None, 0
        for candidate, original_count in self.top_formats:
            candidate_count = original_count - int(nonempty and candidate == signature)
            if candidate_count > dominant_count:
                dominant_format, dominant_count = candidate, candidate_count
        structured_peers = self.structured_count - int(self.structured[row])
        structured_fraction = structured_peers / peers if peers else 0.0
        dominant_fraction = dominant_count / peers if peers else 0.0
        distinct_peers = len(self.counts) - int(nonempty and count == 1)
        distinct_fraction = distinct_peers / peers if peers else 0.0
        core_mass = self.core_mass
        if nonempty and count >= config.min_core_value_count:
            core_mass -= count if count == config.min_core_value_count else 1
        core_fraction = core_mass / peers if peers else 0.0

        if self.schema["free_text"]:
            format_status, value_status = "abstain_free_text", "abstain_free_text"
        else:
            if peers < config.min_comparison_peers:
                format_status = "abstain_insufficient_peers"
            elif structured_fraction < config.min_structured_fraction:
                format_status = "abstain_unstructured_or_mixed_syntax"
            elif dominant_fraction < config.min_dominant_format_fraction:
                format_status = "abstain_heterogeneous_formats"
            else:
                format_status = "comparable"
            if self.schema["identifier"]:
                value_status = "abstain_identifier"
            elif peers < config.min_domain_observations:
                value_status = "abstain_insufficient_peers"
            elif distinct_fraction >= config.identifier_unique_fraction:
                value_status = "abstain_high_distinctness"
            elif distinct_fraction > config.max_domain_distinct_fraction:
                value_status = "abstain_open_domain"
            elif core_fraction < config.min_core_mass_fraction:
                value_status = "abstain_weak_repeated_core"
            else:
                value_status = "comparable"
        format_fraction = format_peers / peers if peers else 0.0
        value_fraction = value_peers / peers if peers else 0.0
        low_format = (
            format_status == "comparable" and format_peers <= config.max_rare_peer_count
            and format_fraction <= config.max_rare_peer_fraction
        )
        low_value = (
            value_status == "comparable" and value_peers <= config.max_rare_peer_count
            and value_fraction <= config.max_rare_peer_fraction
        )
        return {
            "nonempty_peers": peers,
            "value_peer_count": value_peers,
            "value_peer_fraction": value_fraction,
            "value_comparison": value_status,
            "peer_distinct_fraction": distinct_fraction,
            "peer_repeated_core_fraction": core_fraction,
            "format_signature": signature or None,
            "format_peer_count": format_peers,
            "format_peer_fraction": format_fraction,
            "format_comparison": format_status,
            "peer_structured_fraction": structured_fraction,
            "dominant_peer_format": dominant_format,
            "dominant_peer_format_fraction": dominant_fraction,
            "low_format_support": bool(low_format),
            "low_value_support": bool(low_value),
            "low_support": bool(low_format or low_value),
        }


def _probabilities(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match table shape {shape}")
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return array


def _gate_inherited_detection(table: DirtyTable, source_probabilities, empty_probabilities,
                   rule_violations, source_metadata, config=None, *,
                   contextual_low_support=None) -> dict:
    """Only retain gated inherited positives with independently usable evidence.

    No repair availability is consulted. ``evidence`` records changed cells;
    diagnostics also preserve evidence for supported gated retained detections.
    ``contextual_low_support``, when supplied, attests a contradiction from an
    actionable independent-source/entity peer alternative in original dirty data.
    """
    if config is None:
        config = DetectionConfig()
    elif isinstance(config, Mapping):
        config = DetectionConfig(**dict(config))
    elif not isinstance(config, DetectionConfig):
        raise TypeError("config must be DetectionConfig, a mapping, or None")
    if source_metadata is not None and not isinstance(source_metadata, Mapping):
        raise TypeError("source_metadata must be a mapping or None")
    n, m = table.shape
    shape = (n, m)
    if any(not isinstance(column, str) for column in table.columns) or len(set(table.columns)) != m:
        raise ValueError("table columns must be unique strings")
    if any(len(row) != m or any(not isinstance(value, str) for value in row) for row in table.rows):
        raise ValueError("table rows must be rectangular strings")
    source = _probabilities(source_probabilities, shape, "source_probabilities")
    inherited = _probabilities(empty_probabilities, shape, "empty_probabilities")
    rules = np.asarray(rule_violations)
    if rules.shape != shape or rules.dtype != np.dtype(bool):
        raise ValueError("rule_violations must be a Boolean matrix matching the table")
    if contextual_low_support is None:
        contextual = np.zeros(shape, dtype=bool)
    else:
        contextual = np.asarray(contextual_low_support)
        if contextual.shape != shape or contextual.dtype != np.dtype(bool):
            raise ValueError("contextual_low_support must be a Boolean matrix matching the table")
    probabilities = inherited.copy()
    evidence, columns, retained_evidence = [], [], []
    for j, column in enumerate(table.columns):
        values = [row[j] for row in table.rows]
        profile = _DirtySupport(values, column, config)
        fallback = _fallback_metadata(source_metadata, column)
        source_positive_count = int(np.count_nonzero(source[:, j] > 0.5))
        source_fraction = source_positive_count / n if n else 0.0
        triggers = []
        if fallback["empty_predicate_fallback"] is True:
            triggers.append("source_empty_predicate_fallback")
        if source_fraction > config.source_positive_fraction_threshold:
            triggers.append("source_positive_fraction_above_threshold")
        positives = np.flatnonzero(inherited[:, j] > 0.5)
        retained_rule = retained_intrinsic = retained_contextual = demoted = 0
        if triggers:
            for i0 in positives:
                i = int(i0)
                support = profile.cell(i)
                rule_violation = bool(rules[i, j])
                contextual_support_deficit = bool(contextual[i, j])
                retain = rule_violation or support["low_support"] or contextual_support_deficit
                evidence_types = []
                if rule_violation:
                    evidence_types.append("validated_target_specific_rule_violation")
                if support["low_format_support"]:
                    evidence_types.append("low_dirty_leave_one_out_format_support")
                if support["low_value_support"]:
                    evidence_types.append("low_dirty_leave_one_out_value_support")
                if contextual_support_deficit:
                    evidence_types.append("low_dirty_entity_peer_support")
                record = {
                    "row": i, "column": j, "column_name": column,
                    "source_score": float(source[i, j]),
                    "inherited_score": float(inherited[i, j]),
                    "gate_triggers": list(triggers),
                    "validated_rule_violation": rule_violation,
                    "contextual_low_support": contextual_support_deficit,
                    "evidence_types": evidence_types,
                    "dirty_support": support,
                }
                if retain:
                    retained_rule += int(rule_violation)
                    retained_intrinsic += int(support["low_support"])
                    retained_contextual += int(contextual_support_deficit)
                    record.update(action="retain_inherited_positive", final_score=float(probabilities[i, j]))
                    retained_evidence.append(record)
                else:
                    probabilities[i, j] = min(float(inherited[i, j]), config.demoted_score)
                    record.update(action="demote_unsupported_positive", final_score=float(probabilities[i, j]))
                    evidence.append(record)
                    demoted += 1
        columns.append({
            "column": j, "column_name": column,
            "source_metadata": fallback,
            "source_positive_count": source_positive_count,
            "source_positive_fraction": source_fraction,
            "gate_triggers": triggers, "gated": bool(triggers),
            "inherited_positive_count": int(len(positives)),
            "validated_rule_violation_count": int(np.count_nonzero(rules[:, j])),
            "contextual_low_support_count": int(np.count_nonzero(contextual[:, j])),
            "demoted_positive_count": demoted,
            "retained_positive_count": int(len(positives)) - demoted,
            "retained_by_rule_count": retained_rule,
            "retained_by_intrinsic_low_support_count": retained_intrinsic,
            "retained_by_contextual_low_support_count": retained_contextual,
            "dirty_profile": profile.describe(),
        })
    mask = probabilities > 0.5
    diagnostics = {
        "protocol": "unified_loop_v3_intrinsic_detection_gate_v1",
        "score_semantics": "policy_scores_not_calibrated_probabilities",
        "starting_scores": "completed_empty_carry",
        "observation_source": "immutable_original_dirty_strings",
        "decision_threshold": 0.5, "decision_threshold_operator": ">",
        "config": asdict(config),
        "input_fingerprint": table.fingerprint,
        "shape": [n, m],
        "columns": columns,
        "aggregate": {
            "gated_column_count": sum(column["gated"] for column in columns),
            "unknown_fallback_column_count": sum(column["source_metadata"]["empty_predicate_fallback"] is None for column in columns),
            "inherited_positive_count": int(np.count_nonzero(inherited > 0.5)),
            "final_positive_count": int(np.count_nonzero(mask)),
            "changed_cell_count": len(evidence),
            "promoted_cell_count": 0,
        },
        "retained_positive_evidence": retained_evidence,
    }
    return {"probabilities": probabilities, "mask": mask, "evidence": evidence, "diagnostics": diagnostics}


def _canonical_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _date_valid(value):
    """Accept several common literal date families without assuming their order."""
    value = value.strip()
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        pass
    for grammar in ("%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
                    "%Y/%m/%d", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            datetime.strptime(value, grammar)
            return True
        except ValueError:
            pass
    return False


def _list_valid(value):
    # A generic serialized string-list grammar, not an author/name dictionary.
    return bool(re.fullmatch(r'\s*\{\s*(?:"(?:[^"\\]|\\.)*"\s*(?:,\s*"(?:[^"\\]|\\.)*"\s*)*)?\}\s*', value)
                or re.fullmatch(r'\s*\[\s*(?:"(?:[^"\\]|\\.)*"\s*(?:,\s*"(?:[^"\\]|\\.)*"\s*)*)?\]\s*', value))


class _PromotionSupport:
    """Additional immutable-column evidence, always excluding the query cell."""

    def __init__(self, values, column, config):
        self.values, self.config = values, config
        self.profile = _DirtySupport(values, column, config)
        self.keys = [_canonical_key(value) for value in values]
        self.families = defaultdict(Counter)
        for key, value in zip(self.keys, values):
            if key:
                self.families[key][value] += 1
        tokens = set(self.profile.schema["tokens"])
        self.parsers = {}
        if not self.profile.schema["free_text"]:
            self.parsers["numeric"] = lambda value: bool(_NUMBER.fullmatch(value.strip()))
        if tokens & {"date", "datetime", "created", "released", "jcreated"}:
            self.parsers["calendar_date"] = _date_valid
        if tokens & {"list", "array"}:
            self.parsers["serialized_string_list"] = _list_valid
        self.valid = {name: [bool(value.strip()) and parser(value) for value in values]
                      for name, parser in self.parsers.items()}
        self.valid_counts = {name: sum(values) for name, values in self.valid.items()}

    def cell(self, i):
        cfg, profile = self.config, self.profile
        value = self.values[i]
        nonempty = bool(value.strip())
        peers = len(self.values) - 1
        nonempty_peers = profile.nonempty - int(nonempty)
        evidence = []
        inferred_types = []
        if any((ord(character) < 32 and character not in "\t\n\r") or ord(character) == 127
               or character == "\ufffd" for character in value):
            evidence.append({"kind": "embedded_control_or_replacement_character"})
        missing_fraction = (peers - nonempty_peers) / peers if peers else 1.0
        if (not nonempty and nonempty_peers >= cfg.missing_min_observations
                and missing_fraction <= cfg.max_missing_fraction):
            evidence.append({"kind": "missing_in_nearly_complete_column", "nonempty_peers": nonempty_peers,
                             "peer_missing_fraction": missing_fraction})
        if nonempty and nonempty_peers >= cfg.min_comparison_peers:
            for name in self.parsers:
                valid_peers = self.valid_counts[name] - int(self.valid[name][i])
                fraction = valid_peers / nonempty_peers
                if fraction >= cfg.type_min_fraction:
                    inferred_types.append(name)
                if not self.valid[name][i] and fraction >= cfg.type_min_fraction:
                    evidence.append({"kind": "inferred_type_contradiction", "inferred_type": name,
                                     "valid_peers": valid_peers, "nonempty_peers": nonempty_peers,
                                     "valid_peer_fraction": fraction})
        # Case/space variants require a compact nontext domain. Open-vocabulary
        # titles, names and identifiers never become errors merely by rarity.
        distinct_fraction = (len(profile.counts) - int(nonempty and profile.counts[value] == 1)) / max(nonempty_peers, 1)
        if (nonempty and not profile.schema["free_text"] and not profile.schema["identifier"]
                and nonempty_peers >= cfg.min_domain_observations
                and distinct_fraction <= cfg.max_domain_distinct_fraction):
            counts = self.families[self.keys[i]].copy()
            counts[value] -= 1
            counts += Counter()  # Discard zero counts after leave-one-out.
            ranked = counts.most_common(2)
            total = sum(counts.values())
            if (ranked and ranked[0][0] != value and ranked[0][1] >= cfg.canonical_min_support
                    and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
                    and ranked[0][1] / total >= cfg.canonical_min_agreement):
                evidence.append({"kind": "dominant_observed_case_space_variant", "canonical_value": ranked[0][0],
                                 "canonical_peer_count": ranked[0][1], "family_peer_count": total,
                                 "canonical_peer_fraction": ranked[0][1] / total})
        support = profile.cell(i)
        typed_and_valid = any(self.valid[name][i] for name in inferred_types)
        # Rare but valid numbers/dates are ordinary open-domain observations.
        # An established parser also prevents separator diversity from becoming
        # an error merely through the coarse format signature.
        low_format = support["low_format_support"] and not typed_and_valid
        low_value = support["low_value_support"] and not inferred_types
        if low_format or low_value:
            evidence.append({"kind": "dirty_leave_one_out_low_support", "support": support})
        return evidence


def intrinsic_context(table, config):
    """Discover reliable single-column dirty FDs without candidate availability.

    A relation first needs repeated-group coverage and global purity. A query is
    promoted only by a strong leave-one-out peer majority; contradictory modes
    from independently qualified relations cause abstention.
    """
    n, m = table.shape
    columns = [[row[j] for row in table.rows] for j in range(m)]
    witnesses = defaultdict(list)
    relations = []
    for k in range(m):
        groups = defaultdict(list)
        for i, value in enumerate(columns[k]):
            if value.strip():
                groups[value].append(i)
        # Unique/open identifiers cannot establish a reliable repeated FD.
        if len(groups) > config.identifier_unique_fraction * max(n, 1):
            continue
        repeated = [indices for indices in groups.values() if len(indices) >= config.context_min_peers + 1]
        for j in range(m):
            if j == k:
                continue
            eligible = [(indices, Counter(columns[j][i] for i in indices if columns[j][i].strip()))
                        for indices in repeated]
            eligible = [(indices, counts) for indices, counts in eligible if sum(counts.values()) >= config.context_min_peers + 1]
            observations = sum(sum(counts.values()) for _, counts in eligible)
            purity = sum(max(counts.values()) for _, counts in eligible) / max(observations, 1)
            marginal = Counter()
            for _, counts in eligible:
                marginal.update(counts)
            baseline = max(marginal.values(), default=0) / max(observations, 1)
            modes = {counts.most_common(1)[0][0] for _, counts in eligible}
            if (len(eligible) < config.context_min_groups or observations < config.min_comparison_peers
                    or observations / max(n, 1) < config.context_min_coverage
                    or purity < config.context_global_min_agreement or len(modes) < 2
                    or purity - baseline < config.context_min_gain_over_marginal):
                continue
            relation = {"determinants": [table.columns[k]], "dependent": table.columns[j],
                        "repeated_groups": len(eligible), "observations": observations,
                        "coverage": observations / max(n, 1), "global_purity": purity,
                        "marginal_purity": baseline, "gain_over_marginal": purity - baseline,
                        "distinct_group_modes": len(modes)}
            relations.append(relation)
            for indices, counts in eligible:
                for i in indices:
                    original = columns[j][i]
                    peers = counts.copy()
                    peers[original] -= int(bool(original.strip()))
                    peers += Counter()
                    total = sum(peers.values())
                    ranked = peers.most_common(2)
                    if (total < config.context_min_peers or not ranked
                            or (len(ranked) > 1 and ranked[0][1] == ranked[1][1])
                            or ranked[0][1] / total < config.context_min_agreement):
                        continue
                    witnesses[i, j].append({"kind": "intrinsic_dirty_fd", "determinants": [table.columns[k]],
                                             "peer_count": total, "mode": ranked[0][0],
                                             "mode_count": ranked[0][1], "agreement": ranked[0][1] / total})
    accepted, conflicts = {}, 0
    for (i, j), records in witnesses.items():
        modes = {record["mode"] for record in records}
        if len(modes) != 1:
            conflicts += 1
        elif next(iter(modes)) != table.rows[i][j]:
            accepted[i, j] = records
    return accepted, {"qualified_relations": relations, "conflicting_cell_count": conflicts,
                      "contradicting_cell_count": len(accepted)}


def gate_detection(table: DirtyTable, source_probabilities, empty_probabilities,
                   rule_violations, source_metadata, config=None, *, contextual_low_support=None):
    """Run inherited gating, then promote independent intrinsic/rule evidence.

    Every input concerns the immutable dirty table. No repair, seed, CTS query,
    proposal pool or replacement feasibility can influence these decisions.
    """
    cfg = DetectionConfig() if config is None else DetectionConfig(**dict(config)) if isinstance(config, Mapping) else config
    result = _gate_inherited_detection(table, source_probabilities, empty_probabilities,
                                       rule_violations, source_metadata, cfg,
                                       contextual_low_support=contextual_low_support)
    probabilities = result["probabilities"]
    original_mask = probabilities > 0.5
    inherited = np.asarray(empty_probabilities, dtype=float)
    rules = np.asarray(rule_violations, dtype=bool)
    contextual = np.zeros(table.shape, dtype=bool) if contextual_low_support is None else np.asarray(contextual_low_support, dtype=bool)
    context, context_diagnostics = intrinsic_context(table, cfg)
    promotions, retained = [], []
    n, m = table.shape
    for j, column in enumerate(table.columns):
        profile = _PromotionSupport([row[j] for row in table.rows], column, cfg)
        counts = Counter()
        for i in range(n):
            evidence = profile.cell(i)
            if rules[i, j]:
                evidence.append({"kind": "validated_target_specific_rule_violation"})
            if contextual[i, j]:
                evidence.append({"kind": "independent_source_consensus_contradiction"})
            evidence.extend(context.get((i, j), []))
            if not evidence:
                continue
            record = {"row": i, "column": j, "column_name": column,
                      "inherited_score": float(inherited[i, j]), "evidence_types": sorted({item["kind"] for item in evidence}),
                      "intrinsic_evidence": evidence, "original_dirty_value": table.rows[i][j]}
            if not original_mask[i, j]:
                probabilities[i, j] = max(float(probabilities[i, j]), cfg.promoted_score)
                record.update(action="promote_independent_dirty_evidence", final_score=float(probabilities[i, j]))
                promotions.append(record)
                counts["promoted_cells"] += 1
                counts["promoted_inherited_negatives"] += int(inherited[i, j] <= 0.5)
            else:
                record.update(action="retain_independent_dirty_evidence", final_score=float(probabilities[i, j]))
                retained.append(record)
            for item in evidence:
                counts[item["kind"]] += 1
        result["diagnostics"]["columns"][j]["v3_evidence_counts"] = dict(counts)
    result["mask"] = probabilities > 0.5
    result["evidence"].extend(promotions)
    result["diagnostics"]["retained_positive_evidence"].extend(retained)
    result["diagnostics"].update(protocol="unified_loop_v3_dirty_intrinsic_promotions_v1", intrinsic_context=context_diagnostics)
    result["diagnostics"]["aggregate"].update(final_positive_count=int(np.count_nonzero(result["mask"])),
                                              changed_cell_count=int(np.count_nonzero(probabilities != inherited)),
                                              promoted_cell_count=len(promotions),
                                              promoted_inherited_negative_count=sum(record["inherited_score"] <= 0.5 for record in promotions))
    return result
