from __future__ import annotations

from collections import defaultdict

from ..core.config import ChronicleConfig
from ..core.models import (
    CommitChange,
    IndexSnapshot,
    LLMContextBrief,
    PatchContextHints,
    QueryPlan,
    SessionMemoryHints,
    SessionMemorySummary,
    Symbol,
)
from .call_chain import CallChainBuilder
from .context_builder import ContextBuilder
from .context_compressor import ContextCompressor
from .graph_ranker import GraphRanker
from .patch_context import PatchContextAnalyzer
from .provenance import commit_provenance, symbol_provenance
from .symbol_ranker import SymbolRanker
from .token_budget import TokenBudgetManager


class RetrievalOrchestrator:
    def __init__(self, config: ChronicleConfig) -> None:
        self.config = config
        self.symbol_ranker = SymbolRanker()
        self.graph_ranker = GraphRanker()
        self.budget_manager = TokenBudgetManager(config.default_token_budgets)
        self.compressor = ContextCompressor(self.budget_manager)
        self.builder = ContextBuilder()
        self.call_chain_builder = CallChainBuilder()
        self.patch_context = PatchContextAnalyzer(config)

    def build_context(
        self,
        query: str,
        plan: QueryPlan,
        snapshot: IndexSnapshot,
        token_budget: int | None = None,
        memory_hints: SessionMemoryHints | None = None,
    ):
        budget = self.budget_manager.budget_for_intent(plan.intent, token_budget)
        exact_seed_ids = self._seed_ids(plan, snapshot.symbols)
        recent_touch_counts = self._recent_touch_counts(snapshot.commit_changes)
        patch_hints = self.patch_context.analyze(snapshot) if plan.needs_patch_context or plan.intent in {"edit", "debug"} else None

        ranked: list[tuple[Symbol, float, str]] = []
        ranking_scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for symbol in snapshot.symbols:
            score = self.symbol_ranker.score(
                symbol=symbol,
                plan=plan,
                recent_touch_count=recent_touch_counts.get(symbol.file_path, 0),
                exact_seed_ids=exact_seed_ids,
                memory_hints=memory_hints,
                patch_hints=patch_hints,
            )
            score += self.graph_ranker.proximity_bonus(symbol.id, snapshot.call_graph, exact_seed_ids)
            if score <= 0:
                continue
            score = round(score, 3)
            reason = self._reason(symbol, plan, exact_seed_ids, recent_touch_counts.get(symbol.file_path, 0), score)
            ranked.append((symbol, score, reason))
            ranking_scores[symbol.id] = score
            reasons[symbol.id] = reason
        ranked.sort(key=lambda item: item[1], reverse=True)

        compressed_context, selected_symbols, excluded_symbols, estimated_tokens = self.compressor.compress(
            ranked_symbols=ranked[: self.config.max_symbols],
            token_budget=budget,
            exact_seed_ids=exact_seed_ids,
        )
        selected_symbol_ids = {symbol.id for symbol in selected_symbols}
        call_chain = None
        if self._should_include_call_chain(plan=plan, query=query, selected_symbols=selected_symbols):
            call_chain = self.call_chain_builder.build(
                query=query,
                snapshot=snapshot,
                selected_symbols=selected_symbols,
                max_depth=4,
            )
            if call_chain and call_chain.summary:
                chain_section = "Functional call chain:\n" + call_chain.summary
                combined = chain_section + ("\n\n" + compressed_context if compressed_context else "")
                if self.budget_manager.fits(combined, budget):
                    compressed_context = combined
                    estimated_tokens = self.budget_manager.estimate_tokens(combined)

        if patch_hints and patch_hints.summary and selected_symbols:
            patch_section = "Patch-aware context:\n" + patch_hints.summary
            combined = patch_section + ("\n\n" + compressed_context if compressed_context else "")
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)

        selected_commits = self._select_commits(
            plan=plan,
            commit_changes=snapshot.commit_changes,
            selected_symbols=selected_symbols,
            max_commits=self.config.max_commits,
        )
        if selected_commits:
            commit_section = self._commit_section(selected_commits)
            combined = compressed_context + ("\n\n" if compressed_context else "") + commit_section
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)

        memory_summary = self._memory_summary(memory_hints)
        if memory_summary and selected_symbols:
            memory_section = self._memory_section(memory_summary)
            combined = memory_section + ("\n\n" + compressed_context if compressed_context else "")
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)

        llm_brief = self._llm_brief(
            plan=plan,
            query=query,
            selected_symbols=selected_symbols,
            patch_hints=patch_hints,
            call_chain=call_chain,
        )
        coverage_section = self._coverage_checklist_section(
            plan=plan,
            selected_symbols=selected_symbols,
            patch_hints=patch_hints,
            memory_summary=memory_summary,
            call_chain=call_chain,
        )
        priorities_section = self._context_priorities_section(
            selected_symbols=selected_symbols,
            excluded_symbols=excluded_symbols,
            exact_seed_ids=exact_seed_ids,
        )
        if priorities_section:
            combined = priorities_section + ("\n\n" + compressed_context if compressed_context else "")
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)
        if llm_brief:
            brief_section = self._llm_brief_section(llm_brief)
            combined = brief_section + ("\n\n" + compressed_context if compressed_context else "")
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)
        if coverage_section:
            combined = coverage_section + ("\n\n" + compressed_context if compressed_context else "")
            if self.budget_manager.fits(combined, budget):
                compressed_context = combined
                estimated_tokens = self.budget_manager.estimate_tokens(combined)

        provenance = [
            symbol_provenance(symbol, reasons.get(symbol.id, "selected for relevance"), ranking_scores.get(symbol.id, 0.0))
            for symbol in selected_symbols
        ]
        provenance.extend(commit_provenance(commit, 0.5) for commit in selected_commits)
        return self.builder.build(
            query=query,
            token_budget=budget,
            symbols=selected_symbols,
            commits=selected_commits,
            compressed_context=compressed_context,
            estimated_tokens=estimated_tokens,
            provenance=provenance,
            ranking_scores={key: value for key, value in ranking_scores.items() if key in selected_symbol_ids},
            excluded_symbols=excluded_symbols,
            session_id=memory_hints.session_id if memory_hints else None,
            memory_summary=memory_summary,
            call_chain=call_chain,
            patch_context=patch_hints,
            llm_brief=llm_brief,
        )

    def _seed_ids(self, plan: QueryPlan, symbols: list[Symbol]) -> set[str]:
        seeds: set[str] = set()
        for symbol in symbols:
            leaf_name = symbol.name.split(".")[-1].lower()
            if any(candidate.lower().endswith(leaf_name) for candidate in plan.candidate_symbols):
                seeds.add(symbol.id)
            if symbol.file_path in plan.candidate_files:
                seeds.add(symbol.id)
        return seeds

    def _recent_touch_counts(self, changes: list[CommitChange]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for change in changes[:20]:
            for path in change.file_paths:
                counts[path] += 1
        return counts

    def _select_commits(
        self,
        plan: QueryPlan,
        commit_changes: list[CommitChange],
        selected_symbols: list[Symbol],
        max_commits: int,
    ) -> list[CommitChange]:
        symbol_files = {symbol.file_path for symbol in selected_symbols}
        symbol_names = {symbol.name for symbol in selected_symbols}
        scored: list[tuple[float, CommitChange]] = []
        for change in commit_changes:
            score = 0.0
            if symbol_files & set(change.file_paths):
                score += 2.0
            if symbol_names & set(change.symbols_changed):
                score += 2.0
            if plan.needs_git_history:
                score += 1.0
            if any(keyword in change.message.lower() for keyword in plan.keywords):
                score += 0.6
            if score > 0:
                scored.append((score, change))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [change for _, change in scored[:max_commits]]

    def _reason(
        self,
        symbol: Symbol,
        plan: QueryPlan,
        exact_seed_ids: set[str],
        recent_touch_count: int,
        score: float,
    ) -> str:
        reasons: list[str] = []
        if symbol.id in exact_seed_ids:
            reasons.append("exact symbol or file match")
        if any(keyword in symbol.body.lower() for keyword in plan.keywords):
            reasons.append("keyword overlap")
        if recent_touch_count:
            reasons.append(f"recently changed x{recent_touch_count}")
        if not reasons:
            reasons.append(f"graph-relevant candidate ({score:.2f})")
        return ", ".join(reasons)

    def _commit_section(self, commits: list[CommitChange]) -> str:
        lines = ["Relevant commits:"]
        for commit in commits:
            lines.append(
                f"- {commit.commit_hash[:8]} {commit.change_type}: {commit.message} [{', '.join(commit.file_paths[:3])}]"
            )
        return "\n".join(lines)

    def _memory_summary(self, memory_hints: SessionMemoryHints | None) -> SessionMemorySummary | None:
        if memory_hints is None or memory_hints.prior_turn_count == 0:
            return None
        return SessionMemorySummary(
            session_id=memory_hints.session_id,
            prior_turn_count=memory_hints.prior_turn_count,
            recalled_symbols=list(memory_hints.preferred_symbols[:5]),
            recalled_files=list(memory_hints.preferred_files[:4]),
            recent_queries=list(memory_hints.recent_queries[:3]),
            validated_facts=list(memory_hints.validated_facts[:3]),
        )

    def _memory_section(self, summary: SessionMemorySummary) -> str:
        lines = [
            "Session memory:",
            f"- Prior turns: {summary.prior_turn_count}",
            f"- Recalled symbols: {', '.join(summary.recalled_symbols) or 'none'}",
            f"- Recalled files: {', '.join(summary.recalled_files) or 'none'}",
        ]
        if summary.recent_queries:
            lines.append(f"- Recent queries: {' | '.join(summary.recent_queries)}")
        if summary.validated_facts:
            lines.append(f"- Validated facts: {' | '.join(summary.validated_facts)}")
        return "\n".join(lines)

    def _llm_brief(
        self,
        *,
        plan: QueryPlan,
        query: str,
        selected_symbols: list[Symbol],
        patch_hints: PatchContextHints | None,
        call_chain,
    ) -> LLMContextBrief | None:
        if plan.intent == "search" and not patch_hints and call_chain is None:
            return None
        focus: list[str] = []
        focus.extend(symbol.name for symbol in selected_symbols[:4])
        if patch_hints:
            focus.extend(patch_hints.changed_symbol_names[:3])
            focus.extend(patch_hints.related_test_files[:2])
        if call_chain and call_chain.entry_symbol:
            focus.append(f"call-chain:{call_chain.entry_symbol}")
        focus = list(dict.fromkeys(item for item in focus if item))
        objective = {
            "edit": "Plan or explain a grounded code enhancement using edited surfaces and their dependencies.",
            "debug": "Analyze the likely failure path using the retrieved symbols, changed code, and surrounding flow.",
            "architecture": "Explain the functional flow and cross-symbol orchestration without inventing missing behavior.",
            "explain": "Summarize the grounded code path and responsibilities clearly and concisely.",
        }.get(plan.intent, "Use the grounded context only and avoid unsupported claims.")
        recommended_output = {
            "edit": "Return an enhancement plan or patch-oriented explanation with impacted files, callers/callees, and tests.",
            "debug": "Return a grounded diagnosis with probable causes, touched code, and validation checks.",
            "architecture": "Return a flow-oriented explanation with boundaries, call chain, and dependency touchpoints.",
            "explain": "Return a concise explanation tied to exact files and symbols.",
        }.get(plan.intent, "Return a grounded answer with exact file paths and symbols.")
        return LLMContextBrief(
            objective=objective,
            focus_areas=focus,
            recommended_output=recommended_output,
        )

    def _llm_brief_section(self, brief: LLMContextBrief) -> str:
        lines = [
            "LLM task brief:",
            f"- Objective: {brief.objective}",
            f"- Focus areas: {', '.join(brief.focus_areas) or 'none'}",
            f"- Recommended output: {brief.recommended_output}",
        ]
        return "\n".join(lines)

    def _context_priorities_section(
        self,
        *,
        selected_symbols: list[Symbol],
        excluded_symbols: list[str],
        exact_seed_ids: set[str],
    ) -> str:
        if not selected_symbols:
            return ""
        anchors = [symbol.name for symbol in selected_symbols if symbol.id in exact_seed_ids][:4]
        supporting = [symbol.name for symbol in selected_symbols if symbol.id not in exact_seed_ids][:4]
        lines = ["Context priorities:"]
        lines.append(f"- Anchor symbols: {', '.join(anchors) or selected_symbols[0].name}")
        lines.append(f"- Supporting symbols: {', '.join(supporting) or 'none'}")
        if excluded_symbols:
            lines.append(f"- Omitted symbols due to budget: {len(excluded_symbols)}")
        return "\n".join(lines)

    def _coverage_checklist_section(
        self,
        *,
        plan: QueryPlan,
        selected_symbols: list[Symbol],
        patch_hints: PatchContextHints | None,
        memory_summary: SessionMemorySummary | None,
        call_chain,
    ) -> str:
        if not selected_symbols:
            return ""
        lines = ["Coverage checklist:"]
        lines.append("- Treat anchor symbols as the source of truth before generalizing.")
        if call_chain and call_chain.summary:
            lines.append("- Preserve the functional flow described in the call chain.")
        if patch_hints:
            if patch_hints.changed_symbol_names:
                lines.append(f"- Include changed symbols: {', '.join(patch_hints.changed_symbol_names[:4])}")
            if patch_hints.related_test_files:
                lines.append(f"- Include impacted tests: {', '.join(patch_hints.related_test_files[:3])}")
            if patch_hints.interface_files:
                lines.append(f"- Include interfaces or boundaries: {', '.join(patch_hints.interface_files[:3])}")
        if memory_summary and memory_summary.validated_facts:
            lines.append("- Reuse prior validated facts from session memory when they match this task.")
        if plan.needs_runtime_context:
            lines.append("- Check runtime behavior and failure path details before suggesting a fix.")
        if plan.needs_git_history:
            lines.append("- Include recent change history only if it materially explains the behavior.")
        if plan.intent == "edit":
            lines.append("- Call out impacted callers, callees, tests, and persistence boundaries.")
        elif plan.intent == "debug":
            lines.append("- Prefer the shortest grounded failure path and avoid speculative root causes.")
        elif plan.intent == "architecture":
            lines.append("- Explain boundaries and responsibilities without restating the whole repo.")
        return "\n".join(lines)

    def _should_include_call_chain(self, *, plan: QueryPlan, query: str, selected_symbols: list[Symbol]) -> bool:
        if not selected_symbols:
            return False
        lowered = query.lower()
        if plan.intent in {"edit", "architecture", "debug"}:
            return True
        trigger_phrases = ("call chain", "trace", "flow", "path", "reach", "through", "orchestrate")
        return any(phrase in lowered for phrase in trigger_phrases)
