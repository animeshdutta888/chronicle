import tempfile
import unittest
from pathlib import Path

from chronicle import Chronicle


class GraphTests(unittest.TestCase):
    def test_index_builds_symbols_calls_and_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "app"
            pkg.mkdir()
            (pkg / "helpers.py").write_text(
                "def refresh_token(client):\n"
                "    return client.refresh()\n",
                encoding="utf-8",
            )
            (pkg / "auth.py").write_text(
                "from app.helpers import refresh_token\n\n"
                "class AuthService:\n"
                "    def handle_refresh(self, client):\n"
                "        return refresh_token(client)\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            snapshot = chronicle.index()

            symbol_names = {symbol.name for symbol in snapshot.symbols}
            self.assertIn("refresh_token", symbol_names)
            self.assertIn("AuthService", symbol_names)
            self.assertIn("AuthService.handle_refresh", symbol_names)

            service_symbol = next(symbol for symbol in snapshot.symbols if symbol.name == "AuthService.handle_refresh")
            self.assertTrue(any("refresh_token" in call for call in service_symbol.calls))
            self.assertIn("app.helpers.refresh_token", snapshot.dependency_graph["app/auth.py"])

    def test_call_chain_report_returns_mermaid_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "app"
            pkg.mkdir()
            (pkg / "helpers.py").write_text(
                "def refresh_token(client):\n"
                "    return client.refresh()\n",
                encoding="utf-8",
            )
            (pkg / "auth.py").write_text(
                "from app.helpers import refresh_token\n\n"
                "class AuthService:\n"
                "    def handle_refresh(self, client):\n"
                "        return refresh_token(client)\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            report = chronicle.call_chain("Trace how AuthService.handle_refresh calls refresh_token", token_budget=500)

            self.assertEqual(report["entry_symbol"], "AuthService.handle_refresh")
            self.assertIn("refresh_token", report["summary"])
            self.assertIn("flowchart TD", report["mermaid"])


if __name__ == "__main__":
    unittest.main()
