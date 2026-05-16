from __future__ import annotations

from ..core.models import EvaluationReport


class BenchmarkRunner:
    def run(self, report: EvaluationReport) -> dict[str, float]:
        return report.model_dump()
