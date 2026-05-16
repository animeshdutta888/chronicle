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
    ) -> tuple[str, list[Symbol], list[str], int]:
        sections: list[str] = []
        selected: list[Symbol] = []
        excluded: list[str] = []

        for symbol, score, reason in ranked_symbols:
            chunk = self._chunk(symbol=symbol, score=score, reason=reason, exact_match=symbol.id in exact_seed_ids)
            tentative = "\n\n".join(sections + [chunk]) if sections else chunk
            if self.budget_manager.fits(tentative, token_budget):
                sections.append(chunk)
                selected.append(symbol)
                continue

            summary_chunk = self._summary_chunk(symbol=symbol, score=score, reason=reason)
            tentative_summary = "\n\n".join(sections + [summary_chunk]) if sections else summary_chunk
            if self.budget_manager.fits(tentative_summary, token_budget):
                sections.append(summary_chunk)
                selected.append(symbol)
                continue

            excluded.append(symbol.id)

        context = "\n\n".join(sections).strip()
        return context, selected, excluded, self.budget_manager.estimate_tokens(context)

    def _chunk(self, symbol: Symbol, score: float, reason: str, exact_match: bool) -> str:
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
        if exact_match:
            body = self.guardrails.redact(symbol.body)
        else:
            body_lines = symbol.body.splitlines()
            body = self.guardrails.redact("\n".join(body_lines[: min(len(body_lines), 12)]))
        return "\n".join(header + ["Body:", body])

    def _summary_chunk(self, symbol: Symbol, score: float, reason: str) -> str:
        return "\n".join(
            [
                f"Symbol: {symbol.name} ({symbol.type})",
                f"File: {symbol.file_path}:{symbol.start_line}",
                f"Score: {score:.2f}",
                f"Reason: {reason}",
                f"Signature: {symbol.signature or 'n/a'}",
                f"Calls: {', '.join(symbol.calls[:6]) or 'none'}",
            ]
        )
