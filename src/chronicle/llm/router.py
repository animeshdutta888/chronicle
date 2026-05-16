from __future__ import annotations

from ..core.models import ContextPack, LLMDecision, QueryPlan


class LLMRouter:
    def route(self, plan: QueryPlan, context: ContextPack) -> LLMDecision:
        evidence = self._evidence(plan=plan, context=context)
        confidence_floor = 0.35 if plan.intent in {"performance", "dataflow", "architecture"} else 0.45
        if not context.selected_symbols:
            return LLMDecision(
                call_llm=False,
                reason="Chronicle found no grounded symbols, so an LLM call would be speculation.",
                model_class="local",
                max_input_tokens=context.estimated_tokens,
                max_output_tokens=0,
                expected_value="block low-signal model call",
            )
        if context.confidence < confidence_floor:
            return LLMDecision(
                call_llm=False,
                reason="Grounded retrieval confidence is too low; refine the query or increase budget before calling an LLM.",
                model_class="local",
                max_input_tokens=context.estimated_tokens,
                max_output_tokens=0,
                expected_value="avoid low-confidence spend",
            )
        if plan.intent in {"search", "locator"} and evidence["exact_enough"]:
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
        if plan.intent in {"explain", "edit", "refactor", "performance", "dataflow"}:
            model_class = "medium" if evidence["needs_synthesis"] else "small"
            max_output_tokens = 900 if plan.intent in {"edit", "refactor"} else 700
            if plan.intent == "performance":
                reason = (
                    "Performance questions benefit from grounded synthesis across orchestration, I/O, and batching boundaries."
                    if evidence["needs_synthesis"]
                    else "Chronicle found a narrow performance slice; use a compact grounded summary first."
                )
            elif plan.intent == "dataflow":
                reason = "Dataflow questions benefit from summarizing the grounded path across multiple symbols."
            elif plan.intent == "refactor":
                reason = "Refactor questions benefit from grounded change planning across impacted code paths."
            else:
                reason = (
                    "Grounded context is ready for summarization or edit planning."
                    if not context.patch_context
                    else "Patch-aware grounded context is ready for change planning with less prompt bloat."
                )
            return LLMDecision(
                call_llm=True,
                reason=reason,
                model_class=model_class if not (context.patch_context or context.call_chain) else "medium",
                max_input_tokens=min(context.token_budget, 4000),
                max_output_tokens=max_output_tokens,
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

    def _evidence(self, *, plan: QueryPlan, context: ContextPack) -> dict[str, bool]:
        exact_candidates = {candidate.split(".")[-1].lower() for candidate in plan.candidate_symbols}
        selected_names = {symbol.name.split(".")[-1].lower() for symbol in context.selected_symbols[:4]}
        exact_enough = bool(selected_names and exact_candidates and (selected_names & exact_candidates)) or (
            context.confidence >= 0.7 and len(context.selected_symbols) <= 2
        )
        needs_synthesis = (
            len(context.selected_symbols) >= 2
            or bool(context.call_chain and context.call_chain.entry_symbol)
            or bool(context.patch_context and context.patch_context.changed_symbol_names)
        )
        return {
            "exact_enough": exact_enough,
            "needs_synthesis": needs_synthesis,
        }
