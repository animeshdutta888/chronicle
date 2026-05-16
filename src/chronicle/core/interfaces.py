from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    AgentRole,
    ContextPack,
    IndexSnapshot,
    LLMDecision,
    MultiAgentContextBus,
    QueryPlan,
    SessionMemory,
    SessionTurn,
    ValidationResult,
)


class Indexer(Protocol):
    def build(self) -> IndexSnapshot: ...


class QueryPlanner(Protocol):
    def plan(self, query: str) -> QueryPlan: ...


class Router(Protocol):
    def route(self, plan: QueryPlan, context: ContextPack) -> LLMDecision: ...


class Validator(Protocol):
    def validate(self, output_text: str, context: ContextPack) -> ValidationResult: ...


class PersistedSnapshotStore(Protocol):
    def save(self, snapshot: IndexSnapshot, index_dir: Path) -> None: ...
    def load(self, index_dir: Path) -> IndexSnapshot | None: ...


class SessionStore(Protocol):
    def get_or_create(self, index_dir: Path, repo_path: Path, session_id: str) -> SessionMemory: ...
    def load(self, index_dir: Path, session_id: str) -> SessionMemory | None: ...
    def append_turn(self, index_dir: Path, session_id: str, turn: SessionTurn) -> SessionMemory: ...
    def update_latest_turn(
        self,
        index_dir: Path,
        session_id: str,
        *,
        validation_confidence: float | None = None,
        grounded: bool | None = None,
        notes: list[str] | None = None,
    ) -> SessionMemory | None: ...


class AgentBusStore(Protocol):
    def create_bus(
        self,
        index_dir: Path,
        *,
        repo_path: Path,
        bus_id: str,
        root_query: str,
        session_id: str | None,
    ) -> MultiAgentContextBus: ...
    def load_bus(self, index_dir: Path, bus_id: str) -> MultiAgentContextBus | None: ...
    def append_phase(
        self,
        index_dir: Path,
        *,
        bus_id: str,
        role: AgentRole,
        query: str,
        token_budget: int,
        context_pack: ContextPack,
        llm_decision: LLMDecision | None,
        notes: list[str] | None,
    ) -> MultiAgentContextBus: ...
    def add_handoff(
        self,
        index_dir: Path,
        *,
        bus_id: str,
        from_role: AgentRole,
        to_role: AgentRole,
        reason: str,
    ) -> MultiAgentContextBus: ...
    def update_latest_phase_validation(
        self,
        index_dir: Path,
        *,
        bus_id: str,
        validation: ValidationResult,
        notes: list[str] | None = None,
    ) -> MultiAgentContextBus | None: ...
