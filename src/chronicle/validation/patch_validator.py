from __future__ import annotations

import re

from ..core.models import ContextPack


class PatchValidator:
    def validate(self, output_text: str, context: ContextPack) -> list[str]:
        issues: list[str] = []
        allowed_files = {symbol.file_path for symbol in context.selected_symbols}
        patch_files = set(re.findall(r"(?:\+\+\+ b/|--- a/)([\w./-]+)", output_text))
        patch_files.update(re.findall(r"Modify `([\w./-]+)`", output_text))
        for file_path in patch_files:
            if file_path not in allowed_files:
                issues.append(f"Patch references unrelated file: {file_path}")
        return issues
