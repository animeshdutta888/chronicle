from __future__ import annotations

from pathlib import Path


class RepoScanner:
    def __init__(self, repo_path: str | Path, ignored_dirs: set[str], file_extensions: tuple[str, ...]) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.ignored_dirs = ignored_dirs
        self.file_extensions = file_extensions

    def scan(self) -> list[Path]:
        files: list[Path] = []
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in self.file_extensions:
                continue
            try:
                relative_parts = path.relative_to(self.repo_path).parts
            except ValueError:
                relative_parts = path.parts
            if any(part in self.ignored_dirs for part in relative_parts):
                continue
            files.append(path)
        return sorted(files)
