from __future__ import annotations

import json
from pathlib import Path

from ..core.interfaces import PersistedSnapshotStore
from ..core.models import IndexSnapshot
from .schema import SnapshotEnvelope


class JsonSnapshotStore(PersistedSnapshotStore):
    def save(self, snapshot: IndexSnapshot, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        envelope = SnapshotEnvelope.create(snapshot)
        (index_dir / "index.json").write_text(json.dumps(envelope.model_dump(), indent=2), encoding="utf-8")

    def load(self, index_dir: Path) -> IndexSnapshot | None:
        target = index_dir / "index.json"
        if not target.exists():
            return None
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if "snapshot" in raw:
            return SnapshotEnvelope.from_dict(raw).snapshot
        return IndexSnapshot.from_dict(raw)
