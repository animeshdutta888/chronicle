import subprocess
import tempfile
import unittest
from pathlib import Path

from chronicle import Chronicle
from chronicle.remote_repo import resolve_repo_path


class RemoteRepoTests(unittest.TestCase):
    def test_resolve_repo_path_clones_from_local_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()

            self._git(["init"], cwd=source)
            self._git(["config", "user.email", "chronicle@example.com"], cwd=source)
            self._git(["config", "user.name", "Chronicle"], cwd=source)
            (source / "README.md").write_text("hello\n", encoding="utf-8")
            self._git(["add", "README.md"], cwd=source)
            self._git(["commit", "-m", "init"], cwd=source)

            repos_dir = root / "clones"
            cloned_path = resolve_repo_path(
                repo=None,
                repo_url=str(source),
                repos_dir=repos_dir,
            )

            self.assertTrue(cloned_path.exists())
            self.assertTrue((cloned_path / ".git").exists())
            self.assertTrue((cloned_path / "README.md").exists())

            chronicle = Chronicle(repo_path=cloned_path)
            snapshot = chronicle.index()
            self.assertEqual(snapshot.repo_path, str(cloned_path.resolve()))
            diagnosis = chronicle.diagnose(query="Where is hello defined?", token_budget=300)
            self.assertEqual(diagnosis["repo"], str(cloned_path.resolve()))
            self.assertGreaterEqual(diagnosis["symbol_count"], 0)
            self.assertIn("warnings", diagnosis)

    def _git(self, args: list[str], cwd: Path) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
