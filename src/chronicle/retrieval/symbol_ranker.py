from __future__ import annotations

import re

from ..core.models import PatchContextHints, QueryPlan, SessionMemoryHints, Symbol


class SymbolRanker:
    def score(
        self,
        symbol: Symbol,
        plan: QueryPlan,
        recent_touch_count: int,
        exact_seed_ids: set[str],
        memory_hints: SessionMemoryHints | None = None,
        patch_hints: PatchContextHints | None = None,
    ) -> float:
        haystack = " ".join(
            filter(
                None,
                [
                    symbol.name.lower(),
                    symbol.file_path.lower(),
                    symbol.signature.lower() if symbol.signature else "",
                    symbol.docstring.lower() if symbol.docstring else "",
                    symbol.body.lower(),
                ],
            )
        )
        score = 0.0
        exact_name = symbol.name.split(".")[-1].lower()
        normalized_leaf = self._normalize_identifier(symbol.name.split(".")[-1])
        normalized_candidates = {self._normalize_identifier(candidate) for candidate in plan.candidate_symbols}
        normalized_keywords = {self._normalize_identifier(keyword) for keyword in plan.keywords}
        if exact_name and exact_name in plan.keywords:
            score += 5.0
        if normalized_leaf and normalized_leaf in normalized_candidates:
            score += 8.0
        if normalized_leaf and normalized_leaf in normalized_keywords:
            score += 3.0
        if any(candidate.lower().endswith(exact_name) for candidate in plan.candidate_symbols):
            score += 4.0
        if symbol.file_path in plan.candidate_files:
            score += 4.0
        score += sum(0.6 for keyword in plan.keywords if keyword in haystack)
        if normalized_leaf and normalized_leaf in self._normalize_identifier(symbol.file_path):
            score += 1.0
        if symbol.type == "class" and any(candidate[:1].isupper() for candidate in plan.candidate_symbols):
            score += 0.5
        if symbol.file_path.startswith("src/"):
            score += 0.3
        mentions_tests = any(keyword in {"test", "tests"} or keyword.startswith("test_") for keyword in plan.keywords)
        mentions_tests = mentions_tests or any("test" in candidate.lower() for candidate in plan.candidate_files)
        if not mentions_tests and (symbol.file_path.startswith("tests/") or "/tests/" in symbol.file_path):
            score -= 0.8
        if exact_name.startswith("__") and exact_name.endswith("__") and not any("__" in candidate for candidate in plan.candidate_symbols):
            score -= 2.5
        if symbol.id in exact_seed_ids:
            score += 1.0
        if memory_hints is not None:
            if symbol.name in memory_hints.preferred_symbols or symbol.id in memory_hints.preferred_symbols:
                score += 1.8
            if symbol.file_path in memory_hints.preferred_files:
                score += 1.0
            if any(keyword in haystack for keyword in memory_hints.recent_queries):
                score += 0.4
        if patch_hints is not None:
            if symbol.id in patch_hints.changed_symbol_ids or symbol.name in patch_hints.changed_symbol_names:
                score += 5.0
            if symbol.id in patch_hints.related_symbol_ids or symbol.name in patch_hints.related_symbol_names:
                score += 2.2
            if symbol.file_path in patch_hints.changed_files:
                score += 3.0
            if symbol.file_path in patch_hints.related_test_files:
                score += 1.3
            if symbol.file_path in patch_hints.interface_files:
                score += 1.1
        score += min(recent_touch_count, 5) * 0.15
        return round(score, 3)

    def _normalize_identifier(self, text: str) -> str:
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", text.replace(".", "_"))
        if parts:
            return "".join(part.lower() for part in parts)
        return re.sub(r"[^a-z0-9]+", "", text.lower())
