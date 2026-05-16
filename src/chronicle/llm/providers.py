from __future__ import annotations

from ..ollama import OllamaClient, OllamaError


class OllamaProvider:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def generate_text(self, model: str, prompt: str) -> str | None:
        return self.client.generate_text(model, prompt)

    def embed(self, model: str, text: str) -> list[float] | None:
        return self.client.embed(model, text)


__all__ = ["OllamaProvider", "OllamaError"]
