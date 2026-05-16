from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..core.interfaces import AgentBusStore
from ..core.models import (
    AgentHandoffRecord,
    AgentPhaseState,
    AgentRole,
    ContextPack,
    LLMDecision,
    MultiAgentContextBus,
    ValidationResult,
)


class SQLiteAgentBusStore(AgentBusStore):
    def __init__(self, database_name: str = "agent_bus.sqlite3") -> None:
        self.database_name = database_name

    def create_bus(
        self,
        index_dir: Path,
        *,
        repo_path: Path,
        bus_id: str,
        root_query: str,
        session_id: str | None,
    ) -> MultiAgentContextBus:
        self._ensure_schema(index_dir)
        existing = self.load_bus(index_dir, bus_id)
        if existing is not None:
            return existing
        timestamp = self._now()
        with sqlite3.connect(index_dir / self.database_name) as connection:
            connection.execute(
                """
                INSERT INTO buses (bus_id, repo_path, root_query, created_at, updated_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bus_id, str(repo_path), root_query, timestamp, timestamp, session_id),
            )
            connection.commit()
        return MultiAgentContextBus(
            bus_id=bus_id,
            repo_path=str(repo_path),
            root_query=root_query,
            created_at=timestamp,
            updated_at=timestamp,
            session_id=session_id,
            phases=[],
            handoffs=[],
        )

    def load_bus(self, index_dir: Path, bus_id: str) -> MultiAgentContextBus | None:
        database_path = index_dir / self.database_name
        if not database_path.exists():
            return None
        self._ensure_schema(index_dir)
        with sqlite3.connect(database_path) as connection:
            bus_row = connection.execute(
                """
                SELECT bus_id, repo_path, root_query, created_at, updated_at, session_id
                FROM buses
                WHERE bus_id = ?
                """,
                (bus_id,),
            ).fetchone()
            if bus_row is None:
                return None
            phase_rows = connection.execute(
                """
                SELECT role, query, token_budget, context_pack_json, validation_json,
                       llm_decision_json, notes_json, created_at
                FROM bus_phases
                WHERE bus_id = ?
                ORDER BY phase_index ASC
                """,
                (bus_id,),
            ).fetchall()
            handoff_rows = connection.execute(
                """
                SELECT from_role, to_role, reason, created_at
                FROM bus_handoffs
                WHERE bus_id = ?
                ORDER BY handoff_index ASC
                """,
                (bus_id,),
            ).fetchall()
        phases = [
            AgentPhaseState(
                role=str(row[0]),
                query=str(row[1]),
                token_budget=int(row[2]),
                context_pack=ContextPack.model_validate(json.loads(row[3])),
                validation=ValidationResult.model_validate(json.loads(row[4])) if row[4] else None,
                llm_decision=LLMDecision.model_validate(json.loads(row[5])) if row[5] else None,
                notes=self._loads(row[6]),
                created_at=str(row[7]),
            )
            for row in phase_rows
        ]
        handoffs = [
            AgentHandoffRecord(
                from_role=str(row[0]),
                to_role=str(row[1]),
                reason=str(row[2]),
                created_at=str(row[3]),
            )
            for row in handoff_rows
        ]
        return MultiAgentContextBus(
            bus_id=str(bus_row[0]),
            repo_path=str(bus_row[1]),
            root_query=str(bus_row[2]),
            created_at=str(bus_row[3]),
            updated_at=str(bus_row[4]),
            session_id=str(bus_row[5]) if bus_row[5] else None,
            phases=phases,
            handoffs=handoffs,
        )

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
    ) -> MultiAgentContextBus:
        self._ensure_schema(index_dir)
        timestamp = self._now()
        with sqlite3.connect(index_dir / self.database_name) as connection:
            next_index = connection.execute(
                "SELECT COALESCE(MAX(phase_index), 0) + 1 FROM bus_phases WHERE bus_id = ?",
                (bus_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO bus_phases (
                    bus_id, phase_index, role, query, token_budget, context_pack_json,
                    validation_json, llm_decision_json, notes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bus_id,
                    int(next_index),
                    role,
                    query,
                    token_budget,
                    json.dumps(context_pack.model_dump()),
                    None,
                    json.dumps(llm_decision.model_dump()) if llm_decision else None,
                    json.dumps(notes or []),
                    timestamp,
                ),
            )
            connection.execute("UPDATE buses SET updated_at = ? WHERE bus_id = ?", (timestamp, bus_id))
            connection.commit()
        loaded = self.load_bus(index_dir, bus_id)
        if loaded is None:
            raise RuntimeError(f"Chronicle failed to reload agent bus `{bus_id}`.")
        return loaded

    def add_handoff(
        self,
        index_dir: Path,
        *,
        bus_id: str,
        from_role: AgentRole,
        to_role: AgentRole,
        reason: str,
    ) -> MultiAgentContextBus:
        self._ensure_schema(index_dir)
        timestamp = self._now()
        with sqlite3.connect(index_dir / self.database_name) as connection:
            next_index = connection.execute(
                "SELECT COALESCE(MAX(handoff_index), 0) + 1 FROM bus_handoffs WHERE bus_id = ?",
                (bus_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO bus_handoffs (bus_id, handoff_index, from_role, to_role, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bus_id, int(next_index), from_role, to_role, reason, timestamp),
            )
            connection.execute("UPDATE buses SET updated_at = ? WHERE bus_id = ?", (timestamp, bus_id))
            connection.commit()
        loaded = self.load_bus(index_dir, bus_id)
        if loaded is None:
            raise RuntimeError(f"Chronicle failed to reload agent bus `{bus_id}`.")
        return loaded

    def update_latest_phase_validation(
        self,
        index_dir: Path,
        *,
        bus_id: str,
        validation: ValidationResult,
        notes: list[str] | None = None,
    ) -> MultiAgentContextBus | None:
        database_path = index_dir / self.database_name
        if not database_path.exists():
            return None
        self._ensure_schema(index_dir)
        timestamp = self._now()
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                """
                SELECT id, notes_json
                FROM bus_phases
                WHERE bus_id = ?
                ORDER BY phase_index DESC
                LIMIT 1
                """,
                (bus_id,),
            ).fetchone()
            if row is None:
                return None
            merged_notes = self._loads(row[1])
            if notes:
                merged_notes.extend(notes)
            connection.execute(
                """
                UPDATE bus_phases
                SET validation_json = ?, notes_json = ?
                WHERE id = ?
                """,
                (json.dumps(validation.model_dump()), json.dumps(list(dict.fromkeys(merged_notes))), int(row[0])),
            )
            connection.execute("UPDATE buses SET updated_at = ? WHERE bus_id = ?", (timestamp, bus_id))
            connection.commit()
        return self.load_bus(index_dir, bus_id)

    def _ensure_schema(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index_dir / self.database_name) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS buses (
                    bus_id TEXT PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    root_query TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    session_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_phases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bus_id TEXT NOT NULL,
                    phase_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    query TEXT NOT NULL,
                    token_budget INTEGER NOT NULL,
                    context_pack_json TEXT NOT NULL,
                    validation_json TEXT,
                    llm_decision_json TEXT,
                    notes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(bus_id) REFERENCES buses(bus_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_handoffs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bus_id TEXT NOT NULL,
                    handoff_index INTEGER NOT NULL,
                    from_role TEXT NOT NULL,
                    to_role TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(bus_id) REFERENCES buses(bus_id)
                )
                """
            )
            connection.commit()

    def _loads(self, raw: str | None) -> list[str]:
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
