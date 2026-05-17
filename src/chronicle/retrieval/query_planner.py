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
        if self._is_locator_query(lowered):
            return "locator"
        if self._is_performance_query(lowered):
            return "performance"
        if self._is_dataflow_query(lowered):
            return "dataflow"
        if any(token in lowered for token in ("why", "explain", "how does", "what does")):
            return "explain"
        if any(token in lowered for token in ("bug", "error", "failing", "failure", "debug", "fix")):
            return "debug"
        if "refactor" in lowered:
            return "refactor"
        if any(token in lowered for token in ("edit", "change", "update", "modify", "patch", "enhance", "improve", "implement")):
            return "edit"
        if any(token in lowered for token in ("architecture", "design", "flow", "system")):
            return "architecture"
        return "search"

    def _is_locator_query(self, lowered: str) -> bool:
        locator_phrases = (
            "where is",
            "which file",
            "where does",
            "where can i find",
            "defined",
            "definition",
        )
        return any(phrase in lowered for phrase in locator_phrases)

    def _is_performance_query(self, lowered: str) -> bool:
        performance_terms = (
            "latency",
            "slow",
            "faster",
            "speed",
            "performance",
            "throughput",
            "optimiz",
            "bottleneck",
            "async",
            "asynchronous",
            "concurrency",
            "parallel",
            "queue",
            "batch",
        )
        return any(term in lowered for term in performance_terms)

    def _is_dataflow_query(self, lowered: str) -> bool:
        dataflow_terms = (
            "flow",
            "path",
            "call chain",
            "trace",
            "orchestr",
            "how does",
            "through",
            "from",
            "into",
            "pipeline",
        )
        return any(term in lowered for term in dataflow_terms)

    def _keywords(self, lowered: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "how",
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
