from __future__ import annotations

from ..core.models import ContextPack
from .grounding_checker import GroundingChecker


class HallucinationChecker:
    def __init__(self) -> None:
        self.grounding = GroundingChecker()

    def validate(self, output_text: str, context: ContextPack) -> list[str]:
        return self.grounding.validate(output_text, context)
