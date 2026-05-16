from __future__ import annotations


class TokenBudgetManager:
    def __init__(self, defaults: dict[str, int]) -> None:
        self.defaults = defaults

    def budget_for_intent(self, intent: str, requested_budget: int | None = None) -> int:
        if requested_budget is not None:
            return max(200, requested_budget)
        return self.defaults.get(intent, 3000)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def fits(self, text: str, budget: int) -> bool:
        return self.estimate_tokens(text) <= budget
