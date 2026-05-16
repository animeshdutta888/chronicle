from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess

from ..core.models import CommitChange, Symbol


class GitEvolutionAnalyzer:
    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()

    def analyze(self, symbols: list[Symbol], limit: int = 100) -> tuple[list[CommitChange], dict[str, int]]:
        if not (self.repo_path / ".git").exists():
            return [], {}

        symbol_by_file: dict[str, list[Symbol]] = {}
        for symbol in symbols:
            symbol_by_file.setdefault(symbol.file_path, []).append(symbol)

        raw = self._git(
            [
                "log",
                f"--max-count={limit}",
                "--name-only",
                "--format=%H%n%s",
            ],
            allow_fail=True,
        )
        if not raw:
            return [], {}

        entries: list[CommitChange] = []
        churn = Counter()
        current_hash: str | None = None
        current_message: str | None = None
        current_files: list[str] = []

        def flush() -> None:
            nonlocal current_hash, current_message, current_files
            if not current_hash or current_message is None:
                return
            file_paths = [path for path in current_files if path]
            if not file_paths:
                return
            churn.update(file_paths)
            symbols_changed = [
                symbol.name
                for path in file_paths
                for symbol in symbol_by_file.get(path, [])
            ]
            entries.append(
                CommitChange(
                    commit_hash=current_hash,
                    message=current_message,
                    file_paths=file_paths,
                    symbols_changed=list(dict.fromkeys(symbols_changed)),
                    change_type=self._classify_change_type(current_message),
                    risk_flags=self._risk_flags(current_message, file_paths),
                )
            )

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                current_hash = None
                current_message = None
                current_files = []
                continue
            if current_hash is None:
                current_hash = stripped
                continue
            if current_message is None:
                current_message = stripped
                continue
            current_files.append(stripped)
        flush()
        return entries, dict(churn)

    def _classify_change_type(self, message: str) -> str:
        text = message.lower()
        if any(token in text for token in ("validate", "guard", "schema")):
            return "validation"
        if any(token in text for token in ("error", "retry", "except", "timeout")):
            return "error_handling"
        if any(token in text for token in ("api", "endpoint", "signature", "contract")):
            return "api_change"
        if any(token in text for token in ("refactor", "rename", "cleanup")):
            return "refactor"
        if any(token in text for token in ("fix", "add", "support", "change")):
            return "logic"
        return "unknown"

    def _risk_flags(self, message: str, file_paths: list[str]) -> list[str]:
        flags: list[str] = []
        lowered = message.lower()
        if any(token in lowered for token in ("auth", "token", "secret", "credential")):
            flags.append("sensitive-surface")
        if len(file_paths) >= 8:
            flags.append("wide-change")
        if any(path.endswith(("settings.py", "config.py")) for path in file_paths):
            flags.append("configuration-change")
        return flags

    def _git(self, args: list[str], allow_fail: bool = False) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if allow_fail:
                return ""
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout
