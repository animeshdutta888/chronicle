from __future__ import annotations

from ..core.models import ChronicleModel, IndexSnapshot
from ..core.pydantic_compat import Field
from .migrations import CURRENT_SCHEMA_VERSION


class SnapshotEnvelope(ChronicleModel):
    snapshot: IndexSnapshot
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION)

    @classmethod
    def create(cls, snapshot: IndexSnapshot) -> "SnapshotEnvelope":
        return cls(snapshot=snapshot)

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotEnvelope":
        return cls.model_validate(data)
