from __future__ import annotations

from ..core.models import (
    CallChainReport,
    CommitChange,
    ContextPack,
    LLMContextBrief,
    LLMDecision,
    PatchContextHints,
    ProvenanceRecord,
    SessionMemorySummary,
    Symbol,
)


class ContextBuilder:
    def build(
        self,
        query: str,
        token_budget: int,
        symbols: list[Symbol],
        commits: list[CommitChange],
        compressed_context: str,
        estimated_tokens: int,
        provenance: list[ProvenanceRecord],
        ranking_scores: dict[str, float],
        excluded_symbols: list[str],
        llm_decision: LLMDecision | None = None,
        session_id: str | None = None,
        memory_summary: SessionMemorySummary | None = None,
        call_chain: CallChainReport | None = None,
        patch_context: PatchContextHints | None = None,
        llm_brief: LLMContextBrief | None = None,
    ) -> ContextPack:
        confidence = self._confidence(symbols=symbols, commits=commits, token_budget=token_budget, estimated_tokens=estimated_tokens)
        return ContextPack(
            query=query,
            token_budget=token_budget,
            selected_symbols=symbols,
            selected_commits=commits,
            compressed_context=compressed_context,
            estimated_tokens=estimated_tokens,
            provenance=provenance,
            confidence=confidence,
            ranking_scores=ranking_scores,
            excluded_symbols=excluded_symbols,
            llm_decision=llm_decision,
            session_id=session_id,
            memory_summary=memory_summary,
            call_chain=call_chain,
            patch_context=patch_context,
            llm_brief=llm_brief,
        )

    def _confidence(
        self,
        symbols: list[Symbol],
        commits: list[CommitChange],
        token_budget: int,
        estimated_tokens: int,
    ) -> float:
        if not symbols:
            return 0.0
        headroom = max(0.0, min(1.0, 1 - (estimated_tokens / max(token_budget, 1))))
        return round(min(0.95, 0.45 + min(len(symbols), 5) * 0.08 + min(len(commits), 3) * 0.05 + headroom * 0.2), 2)
