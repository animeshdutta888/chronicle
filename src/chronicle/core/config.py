from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ChronicleConfig:
    repo_path: Path
    index_dir: Path
    ignored_dirs: set[str] = field(
        default_factory=lambda: {
            ".git",
            ".chronicle",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
        }
    )
    file_extensions: tuple[str, ...] = (".py",)
    default_token_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "search": 1500,
            "explain": 3000,
            "debug": 5000,
            "edit": 7000,
            "architecture": 9000,
        }
    )
    max_symbols: int = 12
    max_commits: int = 6
    session_query_recall_limit: int = 3
    session_symbol_recall_limit: int = 5
    session_file_recall_limit: int = 4
    patch_related_symbol_limit: int = 8
    patch_related_test_limit: int = 4

    @classmethod
    def from_paths(cls, repo_path: str | Path, index_dir: str | Path | None = None) -> "ChronicleConfig":
        repo = Path(repo_path).resolve()
        index = Path(index_dir).resolve() if index_dir else repo / ".chronicle"
        return cls(repo_path=repo, index_dir=index)
