from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api import Chronicle


class ChronicleMCPServer:
    """Chronicle tool surface that can back MCP or other agent adapters."""

    def __init__(self, repo_path: str | Path, index_dir: str | Path | None = None) -> None:
        self.repo_path = Path(repo_path)
        self.index_dir = Path(index_dir) if index_dir is not None else None

    def _chronicle_for(self, arguments: dict[str, Any]) -> Chronicle:
        repo_path = Path(str(arguments.get("repo_path", self.repo_path)))
        index_dir_value = arguments.get("index_dir")
        if index_dir_value is None:
            index_dir = self.index_dir
        else:
            index_dir = Path(str(index_dir_value))
        return Chronicle(repo_path=repo_path, index_dir=index_dir)

    @staticmethod
    def tool_definitions() -> list[dict[str, Any]]:
        common_repo_props = {
            "repo_path": {
                "type": "string",
                "description": "Optional local repository path override for this call.",
            },
            "index_dir": {
                "type": "string",
                "description": "Optional Chronicle index directory override for this call.",
            },
        }
        common_query_props = {
            **common_repo_props,
            "query": {"type": "string", "description": "Question or task for Chronicle."},
            "token_budget": {"type": "integer", "description": "Optional max token budget for Chronicle context."},
            "session_id": {"type": "string", "description": "Optional Chronicle session id for multi-turn memory."},
        }
        return [
            {
                "name": "index",
                "description": "Build and persist Chronicle repository intelligence.",
                "inputSchema": {
                    "type": "object",
                    "properties": common_repo_props,
                    "additionalProperties": False,
                },
            },
            {
                "name": "context",
                "description": "Retrieve grounded Chronicle context for a repository question.",
                "inputSchema": {
                    "type": "object",
                    "properties": common_query_props,
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "evaluate",
                "description": "Compare Chronicle context against a larger baseline context.",
                "inputSchema": {
                    "type": "object",
                    "properties": common_query_props,
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "doctor",
                "description": "Diagnose repository indexing and retrieval readiness.",
                "inputSchema": {
                    "type": "object",
                    "properties": common_query_props,
                    "additionalProperties": False,
                },
            },
            {
                "name": "call_chain",
                "description": "Build a functional call chain for a repository question.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_query_props,
                        "max_depth": {"type": "integer", "description": "Maximum call-chain depth to trace."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "prepare",
                "description": "Prepare a grounded context packet for a coding agent and save a replayable run.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_query_props,
                        "target": {
                            "type": "string",
                            "enum": ["codex", "claude", "cursor", "generic"],
                            "description": "Agent packet target.",
                        },
                        "view": {
                            "type": "string",
                            "enum": ["compact", "full"],
                            "description": "Compact response or full run artifact.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "prepare_prompt_packet",
                "description": "Prepare a grounded prompt packet for an external LLM call.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_query_props,
                        "include_prompt": {
                            "type": "boolean",
                            "description": "Whether to include a ready-to-send prompt when Chronicle recommends an LLM call.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "status",
                "description": "Show Chronicle index, change, and artifact status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "view": {
                            "type": "string",
                            "enum": ["compact", "full"],
                            "description": "Compact response or full status payload.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "review",
                "description": "Prepare a grounded review packet for recent code changes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_query_props,
                        "view": {
                            "type": "string",
                            "enum": ["compact", "full"],
                            "description": "Compact response or full review artifact.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "handoff",
                "description": "Create a concise handoff packet from latest prepare/review state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "task": {"type": "string", "description": "Optional handoff task title."},
                        "tests": {"type": "string", "description": "Optional tests run or test result summary."},
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional handoff notes.",
                        },
                        "view": {
                            "type": "string",
                            "enum": ["compact", "full"],
                            "description": "Compact response or full handoff artifact.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "session_start",
                "description": "Create or reuse a Chronicle session for multi-turn memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "session_id": {"type": "string", "description": "Optional custom Chronicle session id."},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "session_show",
                "description": "Inspect Chronicle session memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "session_id": {"type": "string", "description": "Chronicle session id to inspect."},
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_start",
                "description": "Create a shared Chronicle multi-agent context bus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "query": {"type": "string", "description": "Root query for the multi-agent workflow."},
                        "bus_id": {"type": "string", "description": "Optional custom Chronicle bus id."},
                        "session_id": {"type": "string", "description": "Optional Chronicle session id to attach."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_context",
                "description": "Add a grounded phase to a Chronicle multi-agent bus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_query_props,
                        "bus_id": {"type": "string", "description": "Chronicle bus id."},
                        "role": {
                            "type": "string",
                            "enum": ["planner", "coder", "reviewer", "critic", "governance", "retriever"],
                            "description": "Role adding this bus phase.",
                        },
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional notes to store with this phase.",
                        },
                    },
                    "required": ["bus_id", "role", "query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_handoff",
                "description": "Record a deterministic handoff between Chronicle bus roles.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "bus_id": {"type": "string", "description": "Chronicle bus id."},
                        "from_role": {
                            "type": "string",
                            "enum": ["planner", "coder", "reviewer", "critic", "governance", "retriever"],
                        },
                        "to_role": {
                            "type": "string",
                            "enum": ["planner", "coder", "reviewer", "critic", "governance", "retriever"],
                        },
                        "reason": {"type": "string", "description": "Why the handoff happened."},
                    },
                    "required": ["bus_id", "from_role", "to_role", "reason"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_show",
                "description": "Inspect a Chronicle multi-agent context bus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "bus_id": {"type": "string", "description": "Chronicle bus id."},
                    },
                    "required": ["bus_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_validate_latest",
                "description": "Validate the latest bus phase output against its grounded Chronicle context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "bus_id": {"type": "string", "description": "Chronicle bus id."},
                        "output_text": {"type": "string", "description": "Agent output to validate."},
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional validation notes to attach to the latest phase.",
                        },
                    },
                    "required": ["bus_id", "output_text"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "bus_summary",
                "description": "Return a compact summary of a Chronicle multi-agent bus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        **common_repo_props,
                        "bus_id": {"type": "string", "description": "Chronicle bus id."},
                    },
                    "required": ["bus_id"],
                    "additionalProperties": False,
                },
            },
        ]

    def handle(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        chronicle = self._chronicle_for(arguments)
        if tool == "index":
            snapshot = chronicle.index()
            return {
                "repo": str(chronicle.config.repo_path),
                "symbol_count": len(snapshot.symbols),
                "commit_change_count": len(snapshot.commit_changes),
            }
        if tool == "context":
            context = chronicle.context(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
            )
            return context.model_dump()
        if tool == "evaluate":
            report = chronicle.evaluate(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
            )
            return report.model_dump()
        if tool == "doctor":
            return chronicle.diagnose(
                query=arguments.get("query"),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
            )
        if tool == "call_chain":
            return chronicle.call_chain(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                max_depth=int(arguments.get("max_depth", 4)),
            )
        if tool == "prepare":
            return chronicle.prepare(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                target=str(arguments.get("target", "generic")),
                view=str(arguments.get("view", "compact")),
            )
        if tool == "prepare_prompt_packet":
            packet = chronicle.prepare_prompt_packet(
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                include_prompt=bool(arguments.get("include_prompt", True)),
            )
            return packet.model_dump()
        if tool == "status":
            return chronicle.status(view=str(arguments.get("view", "compact")))
        if tool == "review":
            return chronicle.review(
                query=str(arguments.get("query", "Review recent code changes and impacted tests")),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                view=str(arguments.get("view", "compact")),
            )
        if tool == "handoff":
            return chronicle.handoff(
                task=arguments.get("task"),
                tests=arguments.get("tests"),
                notes=arguments.get("notes"),
                view=str(arguments.get("view", "compact")),
            )
        if tool == "session_start":
            session = chronicle.start_session(session_id=arguments.get("session_id"))
            return session.model_dump()
        if tool == "session_show":
            session = chronicle.session(str(arguments["session_id"]))
            if session is None:
                raise ValueError(f"Chronicle could not find session `{arguments['session_id']}`.")
            return session.model_dump()
        if tool == "bus_start":
            bus = chronicle.start_agent_bus(
                root_query=str(arguments["query"]),
                bus_id=arguments.get("bus_id"),
                session_id=arguments.get("session_id"),
            )
            return bus.model_dump()
        if tool == "bus_context":
            bus = chronicle.bus_context(
                bus_id=str(arguments["bus_id"]),
                role=str(arguments["role"]),
                query=str(arguments["query"]),
                token_budget=arguments.get("token_budget"),
                session_id=arguments.get("session_id"),
                notes=arguments.get("notes"),
            )
            return bus.model_dump()
        if tool == "bus_handoff":
            bus = chronicle.bus_handoff(
                bus_id=str(arguments["bus_id"]),
                from_role=str(arguments["from_role"]),
                to_role=str(arguments["to_role"]),
                reason=str(arguments["reason"]),
            )
            return bus.model_dump()
        if tool == "bus_show":
            bus = chronicle.agent_bus(str(arguments["bus_id"]))
            if bus is None:
                raise ValueError(f"Chronicle could not find bus `{arguments['bus_id']}`.")
            return bus.model_dump()
        if tool == "bus_validate_latest":
            return chronicle.bus_validate_latest(
                bus_id=str(arguments["bus_id"]),
                output_text=str(arguments["output_text"]),
                notes=arguments.get("notes"),
            )
        if tool == "bus_summary":
            return chronicle.bus_summary(str(arguments["bus_id"]))
        raise ValueError(f"Unsupported Chronicle MCP tool: {tool}")
