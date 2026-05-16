import tempfile
import unittest
from pathlib import Path
import json
from dataclasses import dataclass
import importlib.util
import os
import subprocess

from chronicle import Chronicle
from chronicle import cli as chronicle_cli
from chronicle.integrations import ChronicleMCPServer
from chronicle.integrations.langgraph_node import ChronicleContextNode
from chronicle.llm.guardrails import Guardrails
from chronicle.service.app import create_app


@dataclass
class FakeProvider:
    def generate_text(self, model: str, prompt: str) -> str | None:
        if "RequestContext" in prompt:
            return "RequestContext is defined in app.py as the RequestContext class."
        return "I cannot find the answer in the provided context."


@dataclass
class RepairingProvider:
    def generate_text(self, model: str, prompt: str) -> str | None:
        if "Repaired answer:" in prompt:
            return "Use `app.py` and `RequestContext.push()` only. `RequestContext` is defined in `app.py`."
        return "Modify `missing.py` and call `missing_symbol()` from there."


class ReasoningTests(unittest.TestCase):
    def test_context_prefers_exact_symbol_and_respects_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "auth.py").write_text(
                "def refresh_access_token(user_id, client):\n"
                "    token = client.fetch(user_id)\n"
                "    return token\n\n"
                "def refresh_background_job(queue):\n"
                "    return queue.enqueue('refresh')\n",
                encoding="utf-8",
            )
            (src / "other.py").write_text(
                "def unrelated_helper(value):\n"
                "    return value * 2\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            chronicle.index()
            context = chronicle.context("Where is refresh_access_token handled?", token_budget=260)

            self.assertTrue(context.selected_symbols)
            self.assertEqual(context.selected_symbols[0].name, "refresh_access_token")
            self.assertLessEqual(context.estimated_tokens, context.token_budget)
            assert context.llm_decision is not None
            self.assertFalse(context.llm_decision.call_llm)

    def test_output_validator_catches_ungrounded_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "payments.py").write_text(
                "def process_payment(card):\n"
                "    return card.charge()\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context("Explain process_payment", token_budget=400)
            result = chronicle.validate_output(
                "Modify `src/other.py` and call `missing_symbol()` from there.",
                context,
            )

            self.assertFalse(result.valid)
            self.assertTrue(any("unrelated file" in issue.lower() or "not grounded" in issue.lower() for issue in result.issues))

    def test_langgraph_node_returns_context_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "def build_context(query):\n"
                "    return query.strip()\n",
                encoding="utf-8",
            )

            node = ChronicleContextNode(repo_path=root, token_budget=500)
            result = node({"query": "Where is build_context defined?"})

            self.assertEqual(result["query"], "Where is build_context defined?")
            self.assertIn("context_pack", result)
            self.assertIn("compressed_context", result)

    def test_context_rebuilds_when_cached_snapshot_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "context_agent.py").write_text(
                "def build_pack(query):\n"
                "    return query\n",
                encoding="utf-8",
            )
            index_dir = root / ".chronicle"
            index_dir.mkdir()
            (index_dir / "index.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "snapshot": {
                            "repo_path": str(root),
                            "indexed_at": "2026-01-01T00:00:00+00:00",
                            "symbols": [],
                            "call_graph": {},
                            "dependency_graph": {},
                            "commit_changes": [],
                            "churn_by_file": {},
                        },
                    }
                ),
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root, index_dir=index_dir)
            context = chronicle.context("Where is build_pack defined?", token_budget=300)

            self.assertTrue(context.selected_symbols)
            self.assertEqual(context.selected_symbols[0].name, "build_pack")

    def test_demo_returns_index_context_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app.py"
            app.write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        result = True\n"
                "        details = {\n"
                "            'status': 'active',\n"
                "            'kind': 'request',\n"
                "            'description': 'pushes a request context onto the stack',\n"
                "        }\n"
                "        if details['status'] == 'active':\n"
                "            return result\n"
                "        return False\n\n"
                "def build_request_context(app, environ):\n"
                "    context = RequestContext()\n"
                "    return context\n",
                encoding="utf-8",
            )
            (root / "helpers.py").write_text(
                "def helper_one(value):\n"
                "    total = 0\n"
                "    for item in range(300):\n"
                "        total += item + value\n"
                "    return total\n\n"
                "def helper_two(value):\n"
                "    words = []\n"
                "    for item in range(300):\n"
                "        words.append(f'word-{item}-{value}')\n"
                "    return ','.join(words)\n\n"
                "class BackgroundProcessor:\n"
                "    def run(self, value):\n"
                "        return helper_one(value), helper_two(value)\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            demo = chronicle.demo("Where is RequestContext defined?", token_budget=400)

            self.assertIn("index", demo)
            self.assertIn("context", demo)
            self.assertIn("evaluation", demo)
            self.assertIn("human_summary", demo)
            self.assertIn("repo_insight", demo)
            self.assertIn("llm_readiness", demo)
            self.assertIn("payload_preview", demo["llm_readiness"])
            self.assertIn("focus_summary", demo["llm_readiness"]["payload_preview"])
            self.assertIn("response_policy", demo["llm_readiness"]["payload_preview"])
            self.assertIn("max_output_tokens", demo["llm_readiness"]["payload_preview"]["response_policy"])
            self.assertIn("recommended_next_step", demo["llm_readiness"])
            self.assertGreater(demo["evaluation"]["baseline_tokens"], 0)
            self.assertGreater(demo["evaluation"]["chronicle_tokens"], 0)

    def test_demo_adds_domain_aware_summary_for_image_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "finalpro.py").write_text(
                "import cv2\n\n"
                "def FrameCapture(path):\n"
                "    return cv2.imread(path)\n\n"
                "def cropb(img1):\n"
                "    gray = cv2.imread(img1, 0)\n"
                "    _, thresh = cv2.threshold(gray, 127, 255, 1)\n"
                "    return thresh\n\n"
                "def centroids(imageA):\n"
                "    contours = cv2.findContours(imageA, 1, 2)\n"
                "    return contours\n\n"
                "def labelsquare(board_img):\n"
                "    return 'white-square'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            demo = chronicle.demo("How to detect squares better?", token_budget=500)

            self.assertIn("computer-vision pipeline", demo["repo_insight"])
            self.assertIn("cropb", demo["repo_insight"])
            self.assertIn("centroids", demo["repo_insight"])
            self.assertIn("labelsquare", demo["repo_insight"])
            self.assertIn("detection stage", demo["llm_readiness"]["query_strategy"].lower())
            self.assertIn("call-chain", demo["llm_readiness"]["recommended_next_step"])

    def test_request_context_query_prefers_exact_class_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ctx.py").write_text(
                "class AppContext:\n"
                "    pass\n\n"
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context("Where is RequestContext defined?", token_budget=500)

            self.assertTrue(context.selected_symbols)
            self.assertEqual(context.selected_symbols[0].name, "RequestContext")

    def test_ab_test_compares_baseline_and_chronicle_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )
            (root / "extra.py").write_text(
                "def helper():\n"
                "    data = []\n"
                "    for item in range(400):\n"
                "        data.append(f'helper-{item}')\n"
                "    return ','.join(data)\n\n"
                "class AnotherContext:\n"
                "    def run(self):\n"
                "        return helper()\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            report = chronicle.ab_test(
                query="Where is RequestContext defined?",
                model="fake-model",
                token_budget=400,
                baseline_token_budget=2000,
                llm_provider=FakeProvider(),
            )

            self.assertIn("baseline", report)
            self.assertIn("chronicle", report)
            self.assertIn("comparison", report)
            self.assertGreater(report["baseline"]["estimated_input_tokens"], 0)
            self.assertGreater(report["chronicle"]["estimated_input_tokens"], 0)
            self.assertTrue(report["comparison"]["same_or_better_grounding"])
            self.assertIn("guardrails", report["chronicle"])

    def test_ab_test_applies_grounded_repair_when_first_answer_is_weak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            report = chronicle.ab_test(
                query="Where is RequestContext defined?",
                model="fake-model",
                token_budget=500,
                baseline_token_budget=1200,
                llm_provider=RepairingProvider(),
            )

            self.assertTrue(report["chronicle"]["repaired"])
            self.assertTrue(report["baseline"]["repaired"])
            self.assertGreaterEqual(report["chronicle"]["validation"]["confidence"], 0.5)
            self.assertTrue(report["chronicle"]["repair_notes"])

    def test_guardrails_redact_secrets_before_external_send(self) -> None:
        guardrails = Guardrails()

        result = guardrails.inspect('token="sk-1234567890ABCDE"')

        self.assertTrue(result.contains_secrets)
        self.assertIn("[REDACTED]", result.redacted_text)
        self.assertGreater(result.redaction_count, 0)

    def test_mcp_server_handles_context_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            server = ChronicleMCPServer(repo_path=root)
            payload = server.handle("context", {"query": "Where is ManagerAgent.run defined?", "token_budget": 400})

            self.assertEqual(payload["query"], "Where is ManagerAgent.run defined?")
            self.assertTrue(payload["selected_symbols"])

    def test_session_memory_records_and_recalls_prior_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manager.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self._run_with_retry()\n\n"
                "    def _run_with_retry(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            session = chronicle.start_session("phase2-test")
            first = chronicle.context(
                "Where is ManagerAgent.run defined?",
                token_budget=500,
                session_id=session.session_id,
            )
            second = chronicle.context(
                "How does ManagerAgent.run call retry logic?",
                token_budget=500,
                session_id=session.session_id,
            )
            stored = chronicle.session(session.session_id)

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(len(stored.turns), 2)
            self.assertEqual(second.session_id, session.session_id)
            self.assertIsNotNone(second.memory_summary)
            assert second.memory_summary is not None
            self.assertGreaterEqual(second.memory_summary.prior_turn_count, 1)
            self.assertIn("ManagerAgent", ",".join(second.memory_summary.recalled_symbols))
            self.assertTrue(first.selected_symbols)

    def test_validate_output_updates_session_turn_after_ab_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            session = chronicle.start_session("ab-memory")
            chronicle.ab_test(
                query="Where is RequestContext defined?",
                model="fake-model",
                token_budget=400,
                baseline_token_budget=2000,
                llm_provider=FakeProvider(),
                session_id=session.session_id,
            )
            stored = chronicle.session(session.session_id)

            self.assertIsNotNone(stored)
            assert stored is not None
            self.assertEqual(len(stored.turns), 1)
            self.assertIsNotNone(stored.turns[0].validation_confidence)

    def test_edit_style_query_includes_call_chain_in_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self.execute()\n\n"
                "    def execute(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context(
                "Update ManagerAgent.run to add retry logic and trace its flow",
                token_budget=700,
            )

            self.assertIsNotNone(context.call_chain)
            assert context.call_chain is not None
            self.assertIn("Functional call chain:", context.compressed_context)
            self.assertIn("Context priorities:", context.compressed_context)
            self.assertIn("Coverage checklist:", context.compressed_context)
            self.assertIn("ManagerAgent.run", context.call_chain.summary)

    def test_patch_aware_context_detects_changed_symbols_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "app"
            tests = root / "tests"
            src.mkdir()
            tests.mkdir()
            (src / "manager.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self.execute()\n\n"
                "    def execute(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )
            (tests / "test_manager.py").write_text(
                "from app.manager import ManagerAgent\n\n"
                "def test_run():\n"
                "    assert ManagerAgent().run() == 'ok'\n",
                encoding="utf-8",
            )

            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "chronicle@example.com"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Chronicle"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)

            (src / "manager.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self.execute()\n\n"
                "    def execute(self):\n"
                "        return self.retry()\n\n"
                "    def retry(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context(
                "Enhance ManagerAgent.run to support retry and update impacted tests",
                token_budget=1200,
            )

            self.assertIsNotNone(context.patch_context)
            assert context.patch_context is not None
            self.assertIn("app/manager.py", context.patch_context.changed_files)
            self.assertTrue(any("ManagerAgent.run" in name for name in context.patch_context.changed_symbol_names))
            self.assertTrue(any("tests/test_manager.py" == path for path in context.patch_context.related_test_files))
            self.assertIsNotNone(context.llm_brief)
            assert context.llm_brief is not None
            self.assertIn("enhancement", context.llm_brief.objective.lower())
            self.assertIn("Patch-aware context:", context.compressed_context)

    def test_multi_agent_context_bus_preserves_handoffs_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self.execute()\n\n"
                "    def execute(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            bus = chronicle.start_agent_bus(root_query="Improve ManagerAgent.run flow", bus_id="bus-test")
            self.assertEqual(bus.bus_id, "bus-test")

            bus = chronicle.bus_context(
                bus_id=bus.bus_id,
                role="planner",
                query="Plan the enhancement for ManagerAgent.run",
                token_budget=700,
                notes=["planner phase"],
            )
            self.assertEqual(len(bus.phases), 1)
            self.assertEqual(bus.phases[0].role, "planner")

            bus = chronicle.bus_handoff(
                bus_id=bus.bus_id,
                from_role="planner",
                to_role="coder",
                reason="Plan is grounded and ready for implementation.",
            )
            self.assertEqual(len(bus.handoffs), 1)
            self.assertEqual(bus.handoffs[0].to_role, "coder")

            validation_payload = chronicle.bus_validate_latest(
                bus_id=bus.bus_id,
                output_text="Update `agent.py` around `ManagerAgent.run()` and `ManagerAgent.execute()` only.",
                notes=["validated planner output"],
            )
            self.assertTrue(validation_payload["validation"]["grounded"])

            summary = chronicle.bus_summary(bus.bus_id)
            self.assertEqual(summary["phase_count"], 1)
            self.assertEqual(summary["handoff_count"], 1)
            self.assertEqual(summary["latest_role"], "planner")

            compact = chronicle_cli._compact_bus({"bus": bus.model_dump(), "summary": summary})
            self.assertIn("latest_phase", compact)
            self.assertEqual(compact["summary"]["bus_id"], "bus-test")
            self.assertEqual(compact["latest_phase"]["role"], "planner")
            self.assertIn("human_summary", compact)
            self.assertIn("bus-test", compact["human_summary"])

    def test_compact_context_view_exposes_human_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return self.execute()\n\n"
                "    def execute(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context("Explain ManagerAgent.run", token_budget=600)

            compact = chronicle_cli._compact_context(context.model_dump())
            self.assertIn("human_summary", compact)
            self.assertIn("Chronicle selected", compact["human_summary"])

    def test_low_confidence_context_blocks_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "def alpha_handler():\n"
                "    return 'a'\n\n"
                "def beta_worker():\n"
                "    return 'b'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context("Explain the quantum persistence bridge", token_budget=400)

            self.assertIsNotNone(context.llm_decision)
            assert context.llm_decision is not None
            self.assertFalse(context.llm_decision.call_llm)
            self.assertTrue(
                "too low" in context.llm_decision.reason.lower()
                or "speculation" in context.llm_decision.reason.lower()
            )

    def test_mcp_server_handles_bus_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            server = ChronicleMCPServer(repo_path=root)
            bus = server.handle("bus_start", {"query": "Improve ManagerAgent.run", "bus_id": "mcp-bus"})
            self.assertEqual(bus["bus_id"], "mcp-bus")

            updated = server.handle(
                "bus_context",
                {
                    "bus_id": "mcp-bus",
                    "role": "planner",
                    "query": "Plan ManagerAgent.run enhancement",
                    "token_budget": 500,
                },
            )
            self.assertEqual(len(updated["phases"]), 1)

    def test_hosted_service_dependency_hint_or_health_route(self) -> None:
        if importlib.util.find_spec("fastapi") is None:
            with self.assertRaises(RuntimeError):
                create_app()
            return

        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        root_response = client.get("/")
        response = client.get("/health")

        self.assertEqual(root_response.status_code, 200)
        self.assertIn("Sharper context. Lower spend. Grounded output.", root_response.text)
        self.assertIn(">Chronicle<", root_response.text)
        self.assertIn("Run demo", root_response.text)
        self.assertIn("value=\"demo\"", root_response.text)
        self.assertIn("Result model", root_response.text)
        self.assertIn("repo_url", root_response.text)
        self.assertNotIn(">Docs<", root_response.text)
        self.assertNotIn(">Health<", root_response.text)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_hosted_service_api_key_protects_mutating_endpoints(self) -> None:
        if importlib.util.find_spec("fastapi") is None:
            return

        from fastapi.testclient import TestClient

        original_key = os.environ.get("CHRONICLE_API_KEY")
        os.environ["CHRONICLE_API_KEY"] = "secret-demo-key"
        try:
            app = create_app()
            client = TestClient(app)

            root_response = client.get("/")
            unauthorized = client.post(
                "/doctor",
                json={"repo": ".", "query": "Where is Chronicle defined?", "token_budget": 400},
            )
            authorized = client.post(
                "/doctor",
                headers={"X-API-Key": "secret-demo-key"},
                json={"repo": ".", "query": "Where is Chronicle defined?", "token_budget": 400},
            )

            self.assertEqual(root_response.status_code, 200)
            self.assertIn("Protected mode is enabled", root_response.text)
            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)
        finally:
            if original_key is None:
                os.environ.pop("CHRONICLE_API_KEY", None)
            else:
                os.environ["CHRONICLE_API_KEY"] = original_key


if __name__ == "__main__":
    unittest.main()
