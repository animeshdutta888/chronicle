from __future__ import annotations

from ..core.models import ContextPack, ValidationResult
from .grounding_checker import GroundingChecker
from .hallucination_checker import HallucinationChecker
from .patch_validator import PatchValidator


class OutputValidator:
    def __init__(self) -> None:
        self.grounding_checker = GroundingChecker()
        self.hallucination_checker = HallucinationChecker()
        self.patch_validator = PatchValidator()

    def validate(self, output_text: str, context: ContextPack) -> ValidationResult:
        grounding_issues, stats = self.grounding_checker.analyze(output_text, context)
        issues = []
        issues.extend(grounding_issues)
        issues.extend(self.hallucination_checker.validate(output_text, context))
        issues.extend(self.patch_validator.validate(output_text, context))
        unique_issues = list(dict.fromkeys(issues))
        grounded = stats["ungrounded_references"] == 0 and (
            stats["grounded_references"] > 0 or not context.selected_symbols
        )
        penalty = 0.12 * len(unique_issues)
        confidence = round(
            max(0.0, min(0.99, (context.confidence * 0.6) + (stats["coverage_score"] * 0.4) - penalty)),
            2,
        )
        return ValidationResult(
            valid=not unique_issues,
            issues=unique_issues,
            grounded=grounded,
            confidence=confidence,
            grounded_references=stats["grounded_references"],
            ungrounded_references=stats["ungrounded_references"],
            coverage_score=stats["coverage_score"],
        )
