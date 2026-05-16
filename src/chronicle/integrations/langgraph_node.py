from __future__ import annotations

from pathlib import Path

from ..api import Chronicle


class ChronicleContextNode:
    def __init__(self, repo_path: str | Path, token_budget: int | None = None, index_dir: str | Path | None = None) -> None:
        self.chronicle = Chronicle(repo_path=repo_path, index_dir=index_dir)
        self.token_budget = token_budget

    def __call__(self, state: dict) -> dict:
        query = state.get("query") or state.get("question") or ""
        session_id = state.get("session_id")
        bus_id = state.get("bus_id")
        role = state.get("role")
        context = self.chronicle.context(query=query, token_budget=self.token_budget, session_id=session_id)
        bus = None
        if bus_id and role:
            bus = self.chronicle.bus_context(
                bus_id=str(bus_id),
                role=str(role),
                query=query,
                token_budget=self.token_budget,
                session_id=session_id,
            )
        return {
            "query": query,
            "context_pack": context.model_dump(),
            "compressed_context": context.compressed_context,
            "bus": bus.model_dump() if bus else None,
        }
