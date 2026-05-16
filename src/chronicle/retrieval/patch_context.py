from __future__ import annotations

from pathlib import Path
import re
import subprocess

from ..core.config import ChronicleConfig
from ..core.models import IndexSnapshot, PatchContextHints, Symbol


class PatchContextAnalyzer:
    def __init__(self, config: ChronicleConfig) -> None:
        self.config = config

    def analyze(self, snapshot: IndexSnapshot) -> PatchContextHints | None:
        repo_path = self.config.repo_path
        if not (repo_path / ".git").exists():
            return None
        changed_files = self._changed_python_files(repo_path)
        if not changed_files:
            return None

        changed_lines = self._changed_line_map(repo_path)
        symbols_by_file: dict[str, list[Symbol]] = {}
        for symbol in snapshot.symbols:
            symbols_by_file.setdefault(symbol.file_path, []).append(symbol)

        changed_symbols: list[Symbol] = []
        for file_path in changed_files:
            file_symbols = symbols_by_file.get(file_path, [])
            line_ranges = changed_lines.get(file_path, [])
            for symbol in file_symbols:
                if not line_ranges or any(self._overlaps(symbol.start_line, symbol.end_line, start, end) for start, end in line_ranges):
                    changed_symbols.append(symbol)
        changed_symbols = self._unique_symbols(changed_symbols)
        if not changed_symbols:
            for file_path in changed_files:
                changed_symbols.extend(symbols_by_file.get(file_path, [])[:2])
        else:
            for file_path in changed_files:
                changed_symbols.extend(symbols_by_file.get(file_path, [])[:4])
            changed_symbols = self._unique_symbols(changed_symbols)

        related_symbols = self._related_symbols(snapshot, changed_symbols)
        related_tests = self._related_tests(snapshot, changed_files, changed_symbols)
        interface_files = self._interface_files(snapshot, changed_files)

        summary = self._summary(changed_files, changed_symbols, related_symbols, related_tests, interface_files)
        return PatchContextHints(
            changed_files=changed_files,
            changed_symbol_ids=[symbol.id for symbol in changed_symbols],
            changed_symbol_names=[symbol.name for symbol in changed_symbols],
            related_symbol_ids=[symbol.id for symbol in related_symbols],
            related_symbol_names=[symbol.name for symbol in related_symbols],
            related_test_files=related_tests,
            interface_files=interface_files,
            summary=summary,
        )

    def _changed_python_files(self, repo_path: Path) -> list[str]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", "*.py"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        files = [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".py")]
        return list(dict.fromkeys(files))

    def _changed_line_map(self, repo_path: Path) -> dict[str, list[tuple[int, int]]]:
        result = subprocess.run(
            ["git", "diff", "--unified=0", "--", "*.py"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {}
        mapping: dict[str, list[tuple[int, int]]] = {}
        current_file: str | None = None
        for line in result.stdout.splitlines():
            if line.startswith("+++ b/"):
                current_file = line.replace("+++ b/", "", 1).strip()
                mapping.setdefault(current_file, [])
                continue
            if current_file and line.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if not match:
                    continue
                start = int(match.group(1))
                length = int(match.group(2) or "1")
                mapping[current_file].append((start, start + max(length - 1, 0)))
        return mapping

    def _related_symbols(self, snapshot: IndexSnapshot, changed_symbols: list[Symbol]) -> list[Symbol]:
        if not changed_symbols:
            return []
        symbol_index = {symbol.id: symbol for symbol in snapshot.symbols}
        changed_ids = {symbol.id for symbol in changed_symbols}
        related_ids: list[str] = []
        for symbol in changed_symbols:
            related_ids.extend(snapshot.call_graph.get(symbol.id, []))
            for caller, callees in snapshot.call_graph.items():
                if symbol.id in callees:
                    related_ids.append(caller)
        ordered: list[Symbol] = []
        for symbol_id in related_ids:
            if symbol_id in changed_ids:
                continue
            symbol = symbol_index.get(symbol_id)
            if symbol is None:
                continue
            if symbol not in ordered:
                ordered.append(symbol)
            if len(ordered) >= self.config.patch_related_symbol_limit:
                break
        return ordered

    def _related_tests(
        self,
        snapshot: IndexSnapshot,
        changed_files: list[str],
        changed_symbols: list[Symbol],
    ) -> list[str]:
        candidates: list[str] = []
        stems = {Path(file_path).stem for file_path in changed_files}
        stems.update(symbol.name.split(".")[-1].replace("test_", "") for symbol in changed_symbols)
        for symbol in snapshot.symbols:
            file_path = symbol.file_path
            if "test" not in file_path.lower():
                continue
            if any(stem and stem in file_path for stem in stems):
                if file_path not in candidates:
                    candidates.append(file_path)
            if len(candidates) >= self.config.patch_related_test_limit:
                break
        return candidates

    def _interface_files(self, snapshot: IndexSnapshot, changed_files: list[str]) -> list[str]:
        module_by_file = {file_path: file_path[:-3].replace("/", ".") for file_path in snapshot.dependency_graph if file_path.endswith(".py")}
        reverse_imports: dict[str, list[str]] = {}
        for importer, imports in snapshot.dependency_graph.items():
            for imported in imports:
                reverse_imports.setdefault(imported, []).append(importer)
        interface_files: list[str] = []
        for file_path in changed_files:
            imports = snapshot.dependency_graph.get(file_path, [])
            for imported in imports:
                resolved = self._resolve_import_to_file(module_by_file, imported)
                if resolved and resolved not in interface_files and resolved not in changed_files:
                    interface_files.append(resolved)
            module_name = module_by_file.get(file_path)
            if module_name:
                for importer in reverse_imports.get(module_name, []):
                    if importer not in interface_files and importer not in changed_files:
                        interface_files.append(importer)
        return interface_files[: self.config.patch_related_test_limit]

    def _resolve_import_to_file(self, module_by_file: dict[str, str], imported: str) -> str | None:
        for file_path, module_name in module_by_file.items():
            if imported == module_name or imported.startswith(f"{module_name}."):
                return file_path
        return None

    def _summary(
        self,
        changed_files: list[str],
        changed_symbols: list[Symbol],
        related_symbols: list[Symbol],
        related_tests: list[str],
        interface_files: list[str],
    ) -> str:
        lines = [
            f"- Changed files: {', '.join(changed_files) or 'none'}",
            f"- Edited symbols: {', '.join(symbol.name for symbol in changed_symbols[:6]) or 'none'}",
            f"- Related callers/callees: {', '.join(symbol.name for symbol in related_symbols[:6]) or 'none'}",
            f"- Related tests: {', '.join(related_tests) or 'none'}",
            f"- Interfaces/dependencies: {', '.join(interface_files) or 'none'}",
        ]
        return "\n".join(lines)

    def _overlaps(self, symbol_start: int, symbol_end: int, change_start: int, change_end: int) -> bool:
        return not (symbol_end < change_start or symbol_start > change_end)

    def _unique_symbols(self, symbols: list[Symbol]) -> list[Symbol]:
        seen: set[str] = set()
        ordered: list[Symbol] = []
        for symbol in symbols:
            if symbol.id in seen:
                continue
            seen.add(symbol.id)
            ordered.append(symbol)
        return ordered
