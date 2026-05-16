from __future__ import annotations

from ..core.models import EvaluationReport


def build_ab_report(report: EvaluationReport) -> dict[str, float]:
    return report.model_dump()
