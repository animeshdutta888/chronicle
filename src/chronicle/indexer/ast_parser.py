from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ParsedModule:
    path: Path
    relative_path: str
    source: str
    tree: ast.AST | None
    syntax_error: str | None = None


class PythonAstParser:
    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def parse_files(self, files: list[Path]) -> list[ParsedModule]:
        return [self.parse_file(path) for path in files]

    def parse_file(self, path: str | Path) -> ParsedModule:
        target = Path(path).resolve()
        source = target.read_text(encoding="utf-8")
        relative_path = target.relative_to(self.repo_path).as_posix()
        try:
            tree = ast.parse(source, filename=relative_path)
            return ParsedModule(path=target, relative_path=relative_path, source=source, tree=tree)
        except SyntaxError as exc:
            return ParsedModule(
                path=target,
                relative_path=relative_path,
                source=source,
                tree=None,
                syntax_error=str(exc),
            )
