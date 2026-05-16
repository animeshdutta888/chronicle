from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TokenUsage:
    baseline_tokens: int
    chronicle_tokens: int

    @property
    def saved_tokens(self) -> int:
        return max(0, self.baseline_tokens - self.chronicle_tokens)

    @property
    def reduction_percent(self) -> float:
        if self.baseline_tokens <= 0:
            return 0.0
        return round((self.saved_tokens / self.baseline_tokens) * 100, 2)

    def model_dump(self) -> dict:
        payload = asdict(self)
        payload["saved_tokens"] = self.saved_tokens
        payload["reduction_percent"] = self.reduction_percent
        return payload
