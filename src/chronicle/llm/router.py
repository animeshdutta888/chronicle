from __future__ import annotations

from ..core.models import ContextPack, LLMDecision, QueryPlan


class LLMRouter:
    def route(self, plan: QueryPlan, context: ContextPack) -> LLMDecision:
        if not context.selected_symbols:
            return LLMDecision(
                call_llm=False,
                reason="Chronicle found no grounded symbols, so an LLM call would be speculation.",
                model_class="local",
                max_input_tokens=context.estimated_tokens,
                max_output_tokens=0,
                expected_value="block low-signal model call",
            )
        if context.confidence < 0.45:
            return LLMDecision(
                call_llm=False,
                reason="Grounded retrieval confidence is too low; refine the query or increase budget before calling an LLM.",
                model_class="local",
                max_input_tokens=context.estimated_tokens,
                max_output_tokens=0,
                expected_value="avoid low-confidence spend",
            )
        if plan.intent == "search" and context.confidence >= 0.65 and context.selected_symbols:
            return LLMDecision(
                call_llm=False,
                reason="Deterministic symbol retrieval already answers the lookup.",
                model_class="local",
                max_input_tokens=context.estimated_tokens,
                max_output_tokens=0,
                expected_value="avoid unnecessary model cost",
            )
        if plan.intent == "debug":
            return LLMDecision(
                call_llm=True,
                reason=(
                    "Debugging often benefits from causal reasoning over grounded context."
                    if not context.patch_context
                    else "Debugging changed code benefits from patch-aware grounded reasoning."
                ),
                model_class="large",
                max_input_tokens=min(context.token_budget, 6000),
                max_output_tokens=1200,
                expected_value="deeper failure analysis",
            )
        if plan.intent == "architecture":
            return LLMDecision(
                call_llm=True,
                reason="Architecture questions usually require synthesis across multiple symbols.",
                model_class="large",
                max_input_tokens=min(context.token_budget, 8000),
                max_output_tokens=1400,
                expected_value="cross-cutting explanation",
            )
        if plan.intent in {"explain", "edit"}:
            return LLMDecision(
                call_llm=True,
                reason=(
                    "Grounded context is ready for summarization or edit planning."
                    if not context.patch_context
                    else "Patch-aware grounded context is ready for change planning with less prompt bloat."
                ),
                model_class="medium" if context.patch_context or context.call_chain else "small",
                max_input_tokens=min(context.token_budget, 4000),
                max_output_tokens=900,
                expected_value="low-cost synthesis",
            )
        return LLMDecision(
            call_llm=False,
            reason="No additional reasoning lift is expected from an LLM.",
            model_class="local",
            max_input_tokens=context.estimated_tokens,
            max_output_tokens=0,
            expected_value="deterministic response only",
        )
