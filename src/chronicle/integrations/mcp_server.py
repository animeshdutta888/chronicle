from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api import Chronicle


class ChronicleMCPServer:
    """Minimal MCP-style callable surface for Chronicle integrations."""

    def __init__(self, repo_path: str | Path, index_dir: str | Path | None = None) -> None:
        self.chronicle = Chronicle(repo_path=repo_path, index_dir=index_dir)

    def handle(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool == "index":
            snapshot = self.chronicle.index()
            return {
                "repo": str(self.chronicle.config.repo_path),
                "symbol_count": len(snapshot.symbols),
                "commit_change_count": len(snapshot.commit_changes),
            }
        if tool == "context":
            context = self.chronicle.context(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
            )
            return context.model_dump()
        if tool == "evaluate":
            report = self.chronicle.evaluate(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
            )
            return report.model_dump()
        if tool == "doctor":
            return self.chronicle.diagnose(
                query=arguments.get("query"),
                token_budget=arguments.get("token_budget"),
            )
        if tool == "bus_start":
            bus = self.chronicle.start_agent_bus(
                root_query=str(arguments["query"]),
                bus_id=arguments.get("bus_id"),
                session_id=arguments.get("session_id"),
            )
            return bus.model_dump()
        if tool == "bus_context":
            bus = self.chronicle.bus_context(
                bus_id=str(arguments["bus_id"]),
                role=str(arguments["role"]),
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                notes=arguments.get("notes"),
            )
            return bus.model_dump()
        if tool == "bus_show":
            bus = self.chronicle.agent_bus(str(arguments["bus_id"]))
            if bus is None:
                raise ValueError(f"Chronicle could not find bus `{arguments['bus_id']}`.")
            return bus.model_dump()
        raise ValueError(f"Unsupported Chronicle MCP tool: {tool}")
