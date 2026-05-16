from __future__ import annotations

import re
from typing import Any

from ..core.models import ContextPack


class GroundingChecker:
    IGNORED_IDENTIFIERS = {
        "body",
        "dict",
        "file",
        "files",
        "json",
        "list",
        "message",
        "not",
        "path",
        "paths",
        "reason",
        "response",
        "score",
        "service",
        "set",
        "str",
        "strip",
        "symbol",
        "symbols",
        "text",
        "true",
        "false",
    }

    def validate(self, output_text: str, context: ContextPack) -> list[str]:
        issues, _ = self.analyze(output_text, context)
        return issues

    def analyze(self, output_text: str, context: ContextPack) -> tuple[list[str], dict[str, Any]]:
        issues: list[str] = []
        files = {symbol.file_path for symbol in context.selected_symbols}
        symbol_names = {symbol.name for symbol in context.selected_symbols}
        leaf_names = {symbol.name.split(".")[-1] for symbol in context.selected_symbols}
        literal_context = context.compressed_context

        grounded_references = 0
        ungrounded_references = 0

        referenced_files = set(re.findall(r"[\w./-]+\.py", output_text))
        for file_path in referenced_files:
            if file_path in files:
                grounded_references += 1
                continue
            issues.append(f"Referenced file is not grounded: {file_path}")
            ungrounded_references += 1

        referenced_symbols = self._referenced_symbols(output_text)
        for symbol_name in referenced_symbols:
            if self._is_grounded_symbol(symbol_name, symbol_names, leaf_names, literal_context):
                grounded_references += 1
                continue
            issues.append(f"Referenced symbol is not grounded: {symbol_name}")
            ungrounded_references += 1

        grounded_file_seen = any(file_path in output_text for file_path in files)
        if grounded_references == 0 and not grounded_file_seen and context.selected_symbols:
            issues.append("Output is not grounded in retrieved context.")

        total_references = grounded_references + ungrounded_references
        coverage_score = round(grounded_references / total_references, 2) if total_references else 0.0
        stats = {
            "grounded_references": grounded_references,
            "ungrounded_references": ungrounded_references,
            "coverage_score": coverage_score,
        }
        return issues, stats

    def _referenced_symbols(self, output_text: str) -> set[str]:
        symbols = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", output_text))
        symbols.update(re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(", output_text))
        return {
            symbol
            for symbol in symbols
            if symbol.lower() not in self.IGNORED_IDENTIFIERS and len(symbol) > 2
        }

    def _is_grounded_symbol(
        self,
        symbol_name: str,
        symbol_names: set[str],
        leaf_names: set[str],
        literal_context: str,
    ) -> bool:
        if symbol_name in symbol_names or symbol_name in leaf_names:
            return True
        if any(name.endswith(f".{symbol_name}") for name in symbol_names):
            return True
        if symbol_name.split(".")[-1] in leaf_names:
            return True
        return symbol_name in literal_context
