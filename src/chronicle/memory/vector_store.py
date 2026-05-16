from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InMemoryVectorStore:
    vectors: dict[str, list[float]] = field(default_factory=dict)

    def save(self, identifier: str, vector: list[float]) -> None:
        self.vectors[identifier] = list(vector)

    def get(self, identifier: str) -> list[float] | None:
        vector = self.vectors.get(identifier)
        return list(vector) if vector is not None else None
