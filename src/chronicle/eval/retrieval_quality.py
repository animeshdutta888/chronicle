from __future__ import annotations

from ..core.models import ContextPack


def retrieval_quality(context: ContextPack) -> float:
    if not context.selected_symbols:
        return 0.0
    return round(context.confidence, 2)
