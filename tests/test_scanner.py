import tempfile
import unittest
from pathlib import Path

from chronicle.indexer.repo_scanner import RepoScanner


class RepoScannerTests(unittest.TestCase):
    def test_scanner_does_not_ignore_repo_just_because_parent_path_contains_chronicle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="outer-chronicle-") as tmp:
            root = (Path(tmp) / ".chronicle" / "repos" / "demo").resolve()
            (root / "src").mkdir(parents=True)
            (root / "src" / "main.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "ignored.py").write_text("x = 1\n", encoding="utf-8")

            scanner = RepoScanner(
                repo_path=root,
                ignored_dirs={".git", ".chronicle", "__pycache__"},
                file_extensions=(".py",),
            )

            files = scanner.scan()
            relative_paths = [path.relative_to(root).as_posix() for path in files]

            self.assertEqual(relative_paths, ["src/main.py"])


if __name__ == "__main__":
    unittest.main()
