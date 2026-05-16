from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..core.interfaces import SessionStore
from ..core.models import SessionMemory, SessionTurn


class SQLiteSessionStore(SessionStore):
    def __init__(self, database_name: str = "sessions.sqlite3") -> None:
        self.database_name = database_name

    def get_or_create(self, index_dir: Path, repo_path: Path, session_id: str) -> SessionMemory:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema(index_dir)
        existing = self.load(index_dir, session_id)
        if existing is not None:
            return existing
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(index_dir / self.database_name) as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, repo_path, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, str(repo_path), timestamp, timestamp),
            )
            connection.commit()
        return SessionMemory(
            session_id=session_id,
            repo_path=str(repo_path),
            created_at=timestamp,
            updated_at=timestamp,
            turns=[],
        )

    def load(self, index_dir: Path, session_id: str) -> SessionMemory | None:
        database_path = index_dir / self.database_name
        if not database_path.exists():
            return None
        self._ensure_schema(index_dir)
        with sqlite3.connect(database_path) as connection:
            session_row = connection.execute(
                """
                SELECT session_id, repo_path, created_at, updated_at
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if session_row is None:
                return None
            turn_rows = connection.execute(
                """
                SELECT turn_id, query, intent, token_budget, estimated_tokens,
                       selected_symbols_json, selected_files_json, excluded_symbols_json,
                       validation_confidence, grounded, notes_json, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_index ASC
                """,
                (session_id,),
            ).fetchall()
        turns = [
            SessionTurn(
                turn_id=str(row[0]),
                query=str(row[1]),
                intent=str(row[2]),
                token_budget=int(row[3]),
                estimated_tokens=int(row[4]),
                selected_symbols=self._loads(row[5]),
                selected_files=self._loads(row[6]),
                excluded_symbols=self._loads(row[7]),
                validation_confidence=float(row[8]) if row[8] is not None else None,
                grounded=bool(row[9]) if row[9] is not None else None,
                notes=self._loads(row[10]),
                created_at=str(row[11]),
            )
            for row in turn_rows
        ]
        return SessionMemory(
            session_id=str(session_row[0]),
            repo_path=str(session_row[1]),
            created_at=str(session_row[2]),
            updated_at=str(session_row[3]),
            turns=turns,
        )

    def append_turn(self, index_dir: Path, session_id: str, turn: SessionTurn) -> SessionMemory:
        database_path = index_dir / self.database_name
        self._ensure_schema(index_dir)
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(database_path) as connection:
            next_index = connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO turns (
                    session_id, turn_index, turn_id, query, intent, token_budget, estimated_tokens,
                    selected_symbols_json, selected_files_json, excluded_symbols_json,
                    validation_confidence, grounded, notes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(next_index),
                    turn.turn_id,
                    turn.query,
                    turn.intent,
                    turn.token_budget,
                    turn.estimated_tokens,
                    json.dumps(turn.selected_symbols),
                    json.dumps(turn.selected_files),
                    json.dumps(turn.excluded_symbols),
                    turn.validation_confidence,
                    int(turn.grounded) if turn.grounded is not None else None,
                    json.dumps(turn.notes),
                    turn.created_at,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            connection.commit()
        loaded = self.load(index_dir, session_id)
        if loaded is None:
            raise RuntimeError(f"Chronicle failed to reload session memory for {session_id}.")
        return loaded

    def update_latest_turn(
        self,
        index_dir: Path,
        session_id: str,
        *,
        validation_confidence: float | None = None,
        grounded: bool | None = None,
        notes: list[str] | None = None,
    ) -> SessionMemory | None:
        database_path = index_dir / self.database_name
        if not database_path.exists():
            return None
        self._ensure_schema(index_dir)
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                """
                SELECT id, notes_json
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_index DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            merged_notes = self._loads(row[1])
            if notes:
                merged_notes.extend(notes)
            connection.execute(
                """
                UPDATE turns
                SET validation_confidence = COALESCE(?, validation_confidence),
                    grounded = COALESCE(?, grounded),
                    notes_json = ?
                WHERE id = ?
                """,
                (
                    validation_confidence,
                    int(grounded) if grounded is not None else None,
                    json.dumps(list(dict.fromkeys(merged_notes))),
                    int(row[0]),
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            connection.commit()
        return self.load(index_dir, session_id)

    def summarize_usage(self, session: SessionMemory, *, symbol_limit: int = 5, file_limit: int = 4) -> tuple[list[str], list[str]]:
        symbol_counter = Counter(symbol for turn in session.turns for symbol in turn.selected_symbols)
        file_counter = Counter(file for turn in session.turns for file in turn.selected_files)
        return (
            [name for name, _ in symbol_counter.most_common(symbol_limit)],
            [name for name, _ in file_counter.most_common(file_limit)],
        )

    def _ensure_schema(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(index_dir / self.database_name) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    turn_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    token_budget INTEGER NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    selected_symbols_json TEXT NOT NULL,
                    selected_files_json TEXT NOT NULL,
                    excluded_symbols_json TEXT NOT NULL,
                    validation_confidence REAL,
                    grounded INTEGER,
                    notes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
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
