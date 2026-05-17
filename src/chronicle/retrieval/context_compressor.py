from __future__ import annotations

from ..core.models import Symbol
from ..llm.guardrails import Guardrails
from .token_budget import TokenBudgetManager


class ContextCompressor:
    def __init__(self, budget_manager: TokenBudgetManager) -> None:
        self.budget_manager = budget_manager
        self.guardrails = Guardrails()

    def compress(
        self,
        ranked_symbols: list[tuple[Symbol, float, str]],
        token_budget: int,
        exact_seed_ids: set[str],
        focus_terms: set[str] | None = None,
    ) -> tuple[str, list[Symbol], list[str], int]:
        sections: list[str] = []
        selected: list[Symbol] = []
        excluded: list[str] = []
        anchor_ids = self._anchor_ids(ranked_symbols, exact_seed_ids)

        for symbol, score, reason in ranked_symbols:
            coverage_mode = "anchor" if symbol.id in anchor_ids else "support"
            chunk = self._chunk(
                symbol=symbol,
                score=score,
                reason=reason,
                exact_match=symbol.id in exact_seed_ids,
                coverage_mode=coverage_mode,
                focus_terms=focus_terms or set(),
            )
            tentative = "\n\n".join(sections + [chunk]) if sections else chunk
            if self.budget_manager.fits(tentative, token_budget):
                sections.append(chunk)
                selected.append(symbol)
                continue

            summary_chunk = self._summary_chunk(
                symbol=symbol,
                score=score,
                reason=reason,
                coverage_mode=coverage_mode,
                focus_terms=focus_terms or set(),
            )
            tentative_summary = "\n\n".join(sections + [summary_chunk]) if sections else summary_chunk
            if self.budget_manager.fits(tentative_summary, token_budget):
                sections.append(summary_chunk)
                selected.append(symbol)
                continue

            excluded.append(symbol.id)

        context = "\n\n".join(sections).strip()
        return context, selected, excluded, self.budget_manager.estimate_tokens(context)

    def _anchor_ids(
        self,
        ranked_symbols: list[tuple[Symbol, float, str]],
        exact_seed_ids: set[str],
    ) -> set[str]:
        anchors = set(exact_seed_ids)
        for symbol, _, _ in ranked_symbols[:2]:
            anchors.add(symbol.id)
        return anchors

    def _chunk(
        self,
        symbol: Symbol,
        score: float,
        reason: str,
        exact_match: bool,
        coverage_mode: str,
        focus_terms: set[str],
    ) -> str:
        title = f"Symbol: {symbol.name} ({symbol.type})"
        header = [
            title,
            f"File: {symbol.file_path}:{symbol.start_line}",
            f"Score: {score:.2f}",
            f"Reason: {reason}",
            f"Signature: {symbol.signature or 'n/a'}",
        ]
        if symbol.docstring:
            header.append(f"Docstring: {self.guardrails.redact(symbol.docstring)}")
        if exact_match or coverage_mode == "anchor":
            body = self.guardrails.redact(self._anchor_body(symbol))
        else:
            body = self.guardrails.redact(self._focused_excerpt(symbol, focus_terms, limit=14))
        return "\n".join(header + ["Body:", body])

    def _summary_chunk(
        self,
        symbol: Symbol,
        score: float,
        reason: str,
        coverage_mode: str,
        focus_terms: set[str],
    ) -> str:
        excerpt_limit = 8 if coverage_mode == "anchor" else 6
        excerpt = self.guardrails.redact(self._focused_excerpt(symbol, focus_terms, limit=excerpt_limit))
        return "\n".join(
            [
                f"Symbol: {symbol.name} ({symbol.type})",
                f"File: {symbol.file_path}:{symbol.start_line}",
                f"Score: {score:.2f}",
                f"Reason: {reason}",
                f"Signature: {symbol.signature or 'n/a'}",
                f"Calls: {', '.join(symbol.calls[:6]) or 'none'}",
                "Excerpt:",
                excerpt,
            ]
        )

    def _anchor_body(self, symbol: Symbol) -> str:
        body = symbol.body or ""
        lines = body.splitlines()
        line_limit = 18 if symbol.type == "class" else 42
        char_limit = 1600 if symbol.type == "class" else 2800
        if len(lines) <= line_limit and len(body) <= char_limit:
            return body
        excerpt = "\n".join(lines[:line_limit])
        if len(lines) > line_limit:
            excerpt += "\n..."
        return excerpt

    def _focused_excerpt(self, symbol: Symbol, focus_terms: set[str], limit: int) -> str:
        body_lines = [line for line in symbol.body.splitlines() if line.strip()]
        if not body_lines:
            return "n/a"
        if not focus_terms:
            return "\n".join(body_lines[:limit])
        lowered_lines = [line.lower() for line in body_lines]
        focus_hits = [
            index
            for index, line in enumerate(lowered_lines)
            if any(term in line for term in focus_terms if term)
        ]
        if not focus_hits:
            return "\n".join(body_lines[:limit])
        selected_indexes: list[int] = []
        for hit in focus_hits[:3]:
            start = max(0, hit - 1)
            end = min(len(body_lines), hit + 3)
            for index in range(start, end):
                if index not in selected_indexes:
                    selected_indexes.append(index)
                if len(selected_indexes) >= limit:
                    break
            if len(selected_indexes) >= limit:
                break
        selected_indexes.sort()
        excerpt = "\n".join(body_lines[index] for index in selected_indexes[:limit])
        if selected_indexes and selected_indexes[-1] < len(body_lines) - 1:
            excerpt += "\n..."
        return excerpt
