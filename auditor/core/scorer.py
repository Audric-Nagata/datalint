"""Data Quality Score (DQS) calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_WEIGHTS = {
    "missing_values": 0.25,
    "duplicates": 0.20,
    "leakage": 0.20,
    "type_mismatch": 0.10,
    "outliers": 0.10,
    "skewness": 0.05,
    "feature_noise": 0.05,
    "label_issues": 0.05,
}


@dataclass
class ScoreResult:
    quality_score: float
    score_band: str
    breakdown: dict


def penalize(finding: dict) -> float:
    """Convert an issue finding to a normalized penalty (0.0-1.0)."""
    severity = finding.get("severity", "low")
    severity_map = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
    return severity_map.get(severity, 0.25)


def apply_weights(penalties: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Compute weighted penalties to get final DQS."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_penalty = 0.0
    for check_type, penalty in penalties.items():
        weight = weights.get(check_type, 0.1)
        total_penalty += penalty * weight

    dqs = 100.0 * (1.0 - total_penalty)
    return max(0.0, min(100.0, dqs))


def get_band(score: float) -> str:
    """Map DQS score to band."""
    if score >= 81:
        return "Good"
    elif score >= 66:
        return "Fair"
    elif score >= 41:
        return "Poor"
    else:
        return "Critical"


def compute(module_findings: dict[str, list], weights: dict[str, float] | None = None) -> ScoreResult:
    """Compute final DQS from all module findings."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    penalties = {}
    breakdown = {}

    for module, findings in module_findings.items():
        if not findings:
            penalties[module] = 0.0
            breakdown[module] = {"penalty": 0.0, "weighted": 0.0}
            continue

        module_penalty = sum(penalize(f) for f in findings) / len(findings)
        module_penalty = min(1.0, module_penalty)
        penalties[module] = module_penalty

        weighted = module_penalty * weights.get(module, 0.1)
        breakdown[module] = {"penalty": module_penalty, "weighted": weighted}

    final_score = apply_weights(penalties, weights)
    band = get_band(final_score)

    return ScoreResult(
        quality_score=round(final_score, 1),
        score_band=band,
        breakdown=breakdown,
    )