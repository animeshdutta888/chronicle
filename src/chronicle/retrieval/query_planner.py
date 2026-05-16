from __future__ import annotations

import re

from ..core.models import QueryPlan


class DeterministicQueryPlanner:
    def plan(self, query: str) -> QueryPlan:
        lowered = query.lower()
        intent = self._intent(lowered)
        keywords = self._keywords(lowered)
        candidate_symbols = self._candidate_symbols(query)
        candidate_files = re.findall(r"[\w./-]+\.py", query)
        needs_git_history = any(token in lowered for token in ("when", "changed", "history", "regression", "recent"))
        needs_runtime_context = any(token in lowered for token in ("runtime", "stack trace", "traceback", "exception"))
        needs_patch_context = any(
            token in lowered
            for token in (
                "edit",
                "change",
                "update",
                "modify",
                "patch",
                "enhance",
                "improve",
                "refactor",
                "implement",
                "feature",
                "after change",
            )
        )
        return QueryPlan(
            intent=intent,
            keywords=keywords,
            candidate_symbols=list(dict.fromkeys(candidate_symbols)),
            candidate_files=list(dict.fromkeys(candidate_files)),
            needs_git_history=needs_git_history,
            needs_runtime_context=needs_runtime_context,
            needs_patch_context=needs_patch_context,
        )

    def _intent(self, lowered: str) -> str:
        if any(token in lowered for token in ("why", "explain", "how does", "what does")):
            return "explain"
        if any(token in lowered for token in ("bug", "error", "failing", "failure", "debug", "fix")):
            return "debug"
        if any(token in lowered for token in ("edit", "change", "update", "modify", "patch", "enhance", "improve", "refactor", "implement")):
            return "edit"
        if any(token in lowered for token in ("architecture", "design", "flow", "system")):
            return "architecture"
        return "search"

    def _keywords(self, lowered: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "where",
            "what",
            "does",
            "when",
            "into",
            "from",
            "about",
            "handled",
            "defined",
            "definition",
        }
        return [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]+", lowered)
            if len(token) > 2 and token not in stopwords
        ]

    def _candidate_symbols(self, query: str) -> list[str]:
        candidates: list[str] = []
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]+", query):
            if "_" in token or "." in token:
                candidates.append(token)
                continue
            if any(character.isupper() for character in token[1:]):
                candidates.append(token)
        return list(dict.fromkeys(candidates))
