from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..core.interfaces import PersistedSnapshotStore
from ..core.models import IndexSnapshot
from .schema import SnapshotEnvelope
from .store import JsonSnapshotStore


class SQLiteSnapshotStore(PersistedSnapshotStore):
    """SQLite-backed snapshot persistence with JSON fallback for compatibility."""

    def __init__(self, database_name: str = "index.sqlite3") -> None:
        self.database_name = database_name
        self._json_fallback = JsonSnapshotStore()

    def save(self, snapshot: IndexSnapshot, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        envelope = SnapshotEnvelope.create(snapshot)
        payload = json.dumps(envelope.model_dump(mode="json"))
        database_path = index_dir / self.database_name

        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO snapshots (id, schema_version, indexed_at, snapshot_json)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    indexed_at=excluded.indexed_at,
                    snapshot_json=excluded.snapshot_json
                """,
                (envelope.schema_version, snapshot.indexed_at, payload),
            )
            connection.commit()

        self._json_fallback.save(snapshot=snapshot, index_dir=index_dir)

    def load(self, index_dir: Path) -> IndexSnapshot | None:
        database_path = index_dir / self.database_name
        if database_path.exists():
            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT snapshot_json FROM snapshots WHERE id = 1"
                ).fetchone()
            if row and row[0]:
                raw = json.loads(row[0])
                return SnapshotEnvelope.from_dict(raw).snapshot
        return self._json_fallback.load(index_dir=index_dir)
