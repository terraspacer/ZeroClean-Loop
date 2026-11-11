"""Dirty-only gating of inherited detection scores; no repair or file access.

The caller attests that ``rule_violations`` contains only validated, nonempty,
target-specific rule violations. See detection_design.md for the frozen policy.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re

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

    def __post_init__(self):
        for name in (
            "source_positive_fraction_threshold", "max_domain_distinct_fraction",
            "min_core_mass_fraction", "max_rare_peer_fraction",
            "min_structured_fraction", "min_dominant_format_fraction",
            "identifier_unique_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not isinstance(self.demoted_score, (int, float)) or isinstance(self.demoted_score, bool) or not np.isfinite(self.demoted_score) or not 0 <= self.demoted_score <= 0.5:
            raise ValueError("demoted_score must be finite and in [0, 0.5]")
        for name in ("min_comparison_peers", "min_domain_observations", "min_core_value_count"):
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


def gate_detection(table: DirtyTable, source_probabilities, empty_probabilities,
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
        "protocol": "unified_loop_v2_intrinsic_detection_gate_v1",
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
