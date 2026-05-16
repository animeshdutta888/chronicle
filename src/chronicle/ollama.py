from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    """Raised when Chronicle cannot complete a local Ollama request."""


class OllamaClient:
    """Small client for local Ollama generation and embedding calls."""

    def __init__(self, base_url: str | None = None, timeout_seconds: int = 300) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        env_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS")
        if env_timeout:
            try:
                timeout_seconds = int(env_timeout)
            except ValueError:
                pass
        self.timeout_seconds = timeout_seconds

    def generate_json(self, model: str, prompt: str) -> dict[str, Any] | None:
        body = self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
        )
        if not body:
            return None
        response_text = body.get("response")
        if not isinstance(response_text, str):
            raise OllamaError("Ollama response did not include a text `response` field.")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            raise OllamaError("Ollama returned non-JSON text while Chronicle expected JSON.")
        if isinstance(payload, dict):
            return payload
        raise OllamaError("Ollama returned JSON, but not an object.")

    def generate_text(self, model: str, prompt: str) -> str | None:
        body = self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )
        if not body:
            return None
        response_text = body.get("response")
        if not isinstance(response_text, str):
            raise OllamaError("Ollama response did not include a text `response` field.")
        return response_text

    def embed(self, model: str, text: str) -> list[float] | None:
        body = self._post("/api/embed", {"model": model, "input": text})
        if body and isinstance(body.get("embeddings"), list) and body["embeddings"]:
            vector = body["embeddings"][0]
            if isinstance(vector, list):
                return [float(value) for value in vector]

        # Backward-compatible Ollama embedding endpoint.
        legacy = self._post("/api/embeddings", {"model": model, "prompt": text})
        if legacy and isinstance(legacy.get("embedding"), list):
            return [float(value) for value in legacy["embedding"]]
        return None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            raise OllamaError(
                f"Ollama HTTP {exc.code} for {url}. "
                f"Response: {details[:400] or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(
                f"Could not reach Ollama at {url}. Reason: {reason}. "
                "Check `OLLAMA_BASE_URL` and verify `ollama serve` is running."
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama request to {url} timed out after {self.timeout_seconds}s. "
                "Try a smaller context budget or set `OLLAMA_TIMEOUT_SECONDS=180` and retry."
            ) from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"Ollama returned invalid JSON from {url}: {body[:400]}"
            ) from exc
        if isinstance(parsed, dict):
            if parsed.get("error"):
                raise OllamaError(f"Ollama returned an error: {parsed['error']}")
            return parsed
        raise OllamaError("Ollama returned JSON, but not an object payload.")
