from __future__ import annotations

from ..core.models import BenchmarkConfidence, ContextPack, EvaluationReport, QueryPlan
from ..llm.token_usage import TokenUsage


class TokenSavingsEvaluator:
    def evaluate(self, plan: QueryPlan, baseline_text: str, context: ContextPack) -> EvaluationReport:
        usage = TokenUsage(
            baseline_tokens=max(1, len(baseline_text) // 4),
            chronicle_tokens=context.estimated_tokens,
        )
        candidate_total = len(plan.candidate_symbols) + len(plan.candidate_files)
        matched = 0
        selected_symbol_names = {symbol.name for symbol in context.selected_symbols}
        selected_files = {symbol.file_path for symbol in context.selected_symbols}
        for symbol in plan.candidate_symbols:
            if any(name.endswith(symbol) for name in selected_symbol_names):
                matched += 1
        for file_path in plan.candidate_files:
            if file_path in selected_files:
                matched += 1
        retrieval_hit_rate = 1.0 if candidate_total == 0 and context.selected_symbols else (
            matched / candidate_total if candidate_total else 0.0
        )
        unused_context_percent = 0.0
        if context.selected_symbols:
            unused = len(context.excluded_symbols)
            unused_context_percent = round((unused / max(len(context.selected_symbols) + unused, 1)) * 100, 2)

        benchmark_confidence = self._benchmark_confidence(
            retrieval_hit_rate=retrieval_hit_rate,
            grounding_score=context.confidence,
            reduction_percent=usage.reduction_percent,
        )
        recommendation = self._recommendation(benchmark_confidence, usage.reduction_percent, retrieval_hit_rate)
        return EvaluationReport(
            baseline_tokens=usage.baseline_tokens,
            chronicle_tokens=usage.chronicle_tokens,
            token_reduction_percent=usage.reduction_percent,
            retrieval_hit_rate=round(retrieval_hit_rate, 2),
            unused_context_percent=unused_context_percent,
            answer_grounding_score=context.confidence,
            estimated_cost_saved=round((usage.saved_tokens / 1000) * 0.003, 4),
            benchmark_confidence=benchmark_confidence,
            recommendation=recommendation,
        )

    def _benchmark_confidence(
        self,
        retrieval_hit_rate: float,
        grounding_score: float,
        reduction_percent: float,
    ) -> BenchmarkConfidence:
        if retrieval_hit_rate >= 0.75 and grounding_score >= 0.8 and reduction_percent >= 50:
            return "high"
        if retrieval_hit_rate >= 0.4 and grounding_score >= 0.55 and reduction_percent >= 25:
            return "medium"
        return "low"

    def _recommendation(self, confidence: BenchmarkConfidence, reduction_percent: float, retrieval_hit_rate: float) -> str:
        if confidence == "high":
            return "Chronicle context is benchmark-ready for this query class."
        if confidence == "medium":
            return (
                "Chronicle shows promising savings, but this query class still needs manual review "
                "before claiming production-grade parity."
            )
        if reduction_percent <= 0 or retrieval_hit_rate == 0:
            return "Do not trust this benchmark yet; Chronicle needs better retrieval or indexing for this query."
        return "Chronicle saves tokens here, but grounding confidence is still too weak for a production claim."
