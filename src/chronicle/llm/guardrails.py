from __future__ import annotations

import re

from ..core.models import GuardrailCheck


class Guardrails:
    SECRET_PATTERNS = {
        "openai_key": re.compile(r"sk-[A-Za-z0-9]{10,}"),
        "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "assignment_secret": re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]"),
        "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    }

    def redact(self, text: str) -> str:
        return self.inspect(text).redacted_text

    def contains_secret(self, text: str) -> bool:
        return self.inspect(text).contains_secrets

    def inspect(self, text: str) -> GuardrailCheck:
        redacted = text
        blocked_patterns: list[str] = []
        redaction_count = 0
        for name, pattern in self.SECRET_PATTERNS.items():
            matches = list(pattern.finditer(redacted))
            if not matches:
                continue
            blocked_patterns.append(name)
            redaction_count += len(matches)
            redacted = pattern.sub("[REDACTED]", redacted)
        return GuardrailCheck(
            safe_to_send=redaction_count == 0 or "[REDACTED]" in redacted,
            contains_secrets=redaction_count > 0,
            redacted_text=redacted,
            blocked_patterns=blocked_patterns,
            redaction_count=redaction_count,
        )
