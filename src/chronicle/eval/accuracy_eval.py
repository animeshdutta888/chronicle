from __future__ import annotations

from ..core.models import ContextPack


def accuracy_estimate(context: ContextPack) -> float:
    return round(context.confidence, 2)
