import tempfile
import unittest
from pathlib import Path
import json
from dataclasses import dataclass
import importlib.util
import os
import subprocess
from unittest.mock import patch

from chronicle import Chronicle
from chronicle import cli as chronicle_cli
from chronicle.integrations import ChronicleMCPServer
from chronicle.mcp_stdio import ChronicleMCPStdioServer
from chronicle.integrations.langgraph_node import ChronicleContextNode
from chronicle.llm.guardrails import Guardrails
from chronicle.llm.router import LLMRouter
from chronicle.retrieval.query_planner import DeterministicQueryPlanner
from chronicle.service.app import create_app
from chronicle.core.models import ValidationResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@dataclass
class GroundingSkewProvider:
    def generate_text(self, model: str, prompt: str) -> str | None:
        if "File: app.py" in prompt:
            return "RequestContext is defined in app.py as the RequestContext class."
        return "RequestContext is defined in missing.py and handled by missing_symbol()."


class ReasoningTests(unittest.TestCase):
    def test_setup_agents_create_or_append_workflow_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Existing notes\n", encoding="utf-8")

            chronicle = Chronicle(repo_path=root)
            result = chronicle.setup_agent("all", configure_mcp=False)

            self.assertEqual(len(result["updated"]), 3)
            agents = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("# Existing notes", agents)
            self.assertIn("## Chronicle Context Workflow", agents)
            self.assertIn("chronicle prepare", (root / "CLAUDE.md").read_text(encoding="utf-8"))
            self.assertTrue((root / ".cursor" / "rules" / "chronicle.mdc").exists())

            second = chronicle.setup_agent("codex", configure_mcp=False)
            self.assertEqual(second["updated"][0]["action"], "unchanged")
            self.assertEqual((root / "AGENTS.md").read_text(encoding="utf-8").count("## Chronicle Context Workflow"), 1)

    def test_setup_codex_mcp_skips_when_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chronicle = Chronicle(repo_path=root)

            with patch("chronicle.api.shutil.which", return_value=None):
                result = chronicle.setup_agent("codex", instructions=False)

            self.assertEqual(result["updated"], [])
            self.assertEqual(result["configured"][0]["status"], "skipped")
            self.assertIn("codex mcp add chronicle", result["configured"][0]["manual_command"])

    def test_setup_cursor_mcp_writes_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chronicle = Chronicle(repo_path=root)

            result = chronicle.setup_agent("cursor", instructions=False)
            mcp_json = root / ".cursor" / "mcp.json"
            config = json.loads(mcp_json.read_text(encoding="utf-8"))

            self.assertEqual(result["configured"][0]["status"], "configured")
            self.assertEqual(config["mcpServers"]["chronicle"]["command"], "chronicle-mcp")
            self.assertEqual(config["mcpServers"]["chronicle"]["args"], ["--repo", str(root.resolve())])

    def test_pr_review_handles_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Chronicle Test"], cwd=root, check=True)
            (root / "auth.py").write_text("def refresh_token(client):\n    return client.refresh()\n", encoding="utf-8")
            subprocess.run(["git", "add", "auth.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "checkout", "-b", "feature"], cwd=root, check=True, capture_output=True)
            (root / "auth.py").write_text("def refresh_token(client):\n    token = client.refresh()\n    return token\n", encoding="utf-8")

            chronicle = Chronicle(repo_path=root)
            result = chronicle.pr_review(base="main")

            self.assertIn("auth.py", result["changed_files"])
            self.assertIn("refresh_token", result["impacted_symbols"])
            self.assertTrue(Path(result["saved"]["pr_review_md"]).exists())

    def test_pr_review_requires_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chronicle = Chronicle(repo_path=Path(tmp))

            with self.assertRaises(ValueError):
                chronicle.pr_review(base="main")

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

    def test_prepare_prompt_packet_returns_sdk_ready_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n\n"
                "def build_request_context(app, environ):\n"
                "    context = RequestContext()\n"
                "    return context\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            packet = chronicle.prepare_prompt_packet(
                "Explain how RequestContext is used.",
                token_budget=500,
            )

            self.assertEqual(packet.query, "Explain how RequestContext is used.")
            self.assertTrue(packet.selected_symbols)
            self.assertTrue(packet.compressed_context)
            self.assertIn("output_format", packet.response_policy)

    def test_context_compression_preserves_richer_anchor_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "async_pipeline.py").write_text(
                "async def load_batch(source):\n"
                "    payload = await source.fetch()\n"
                "    return payload\n\n"
                "def chunk_batch(items):\n"
                "    chunks = [items[i:i+10] for i in range(0, len(items), 10)]\n"
                "    return chunks\n\n"
                "async def async_pipeline(source):\n"
                "    data = await load_batch(source)\n"
                "    chunks = chunk_batch(data)\n"
                "    return await fetch_preprocess_train_chunks(chunks)\n\n"
                "async def fetch_preprocess_train_chunks(chunks):\n"
                "    results = []\n"
                "    for chunk in chunks:\n"
                "        results.append(chunk)\n"
                "    return results\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context(
                "How to reduce latency in async_pipeline and fetch_preprocess_train_chunks?",
                token_budget=700,
                remember=False,
            )

            self.assertIn("data = await load_batch(source)", context.compressed_context)
            self.assertIn("results.append(chunk)", context.compressed_context)

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

    def test_demo_uses_performance_guidance_for_latency_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "async_pipeline.py").write_text(
                "async def fetch_preprocess_train_chunks(source):\n"
                "    data = await source.fetch()\n"
                "    chunks = [data[i:i+10] for i in range(0, len(data), 10)]\n"
                "    return chunks\n\n"
                "async def async_pipeline(source):\n"
                "    chunks = await fetch_preprocess_train_chunks(source)\n"
                "    return chunks\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            demo = chronicle.demo("How to improve latency of asynchronous pipelines?", token_budget=500)

            self.assertIn("performance-oriented pipeline", demo["repo_insight"])
            self.assertIn("latency stage", demo["llm_readiness"]["query_strategy"].lower())
            self.assertNotIn("classification", demo["llm_readiness"]["query_strategy"].lower())
            self.assertNotIn("classification", demo["llm_readiness"]["recommended_next_step"].lower())

    def test_query_planner_classifies_locator_and_performance_intents(self) -> None:
        planner = DeterministicQueryPlanner()

        self.assertEqual(planner.plan("Where is RequestContext defined?").intent, "locator")
        self.assertEqual(planner.plan("How to improve latency of asynchronous pipelines?").intent, "performance")

    def test_router_uses_evidence_aware_policy_for_locator_and_performance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ctx.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )
            (root / "async_pipeline.py").write_text(
                "async def load_batch(source):\n"
                "    return await source.fetch()\n\n"
                "def chunk_batch(items):\n"
                "    return [items[i:i+10] for i in range(0, len(items), 10)]\n\n"
                "async def async_pipeline(source):\n"
                "    return await fetch_preprocess_train_chunks(source)\n\n"
                "async def fetch_preprocess_train_chunks(source):\n"
                "    data = await load_batch(source)\n"
                "    return chunk_batch(data)\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            planner = DeterministicQueryPlanner()
            router = LLMRouter()

            locator_query = "Where is RequestContext defined?"
            locator_context = chronicle.context(locator_query, token_budget=700, remember=False)
            locator_decision = router.route(plan=planner.plan(locator_query), context=locator_context)

            performance_query = "How to reduce latency in async_pipeline and fetch_preprocess_train_chunks?"
            performance_context = chronicle.context(performance_query, token_budget=1200, remember=False)
            performance_decision = router.route(plan=planner.plan(performance_query), context=performance_context)

            self.assertFalse(locator_decision.call_llm)
            self.assertTrue(performance_decision.call_llm)

    def test_context_selection_covers_multiple_query_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agents").mkdir()
            (root / "runtime").mkdir()
            (root / "app").mkdir()
            (root / "app" / "tools").mkdir()
            (root / "agents" / "manager.py").write_text(
                "class ManagerAgent:\n"
                "    async def run(self, state):\n"
                "        memory = await self._memory.recall(state)\n"
                "        return memory\n",
                encoding="utf-8",
            )
            (root / "agents" / "memory.py").write_text(
                "class MemoryAgent:\n"
                "    async def recall(self, state):\n"
                "        return []\n",
                encoding="utf-8",
            )
            (root / "app" / "tools" / "reminders.py").write_text(
                "def add_reminder(path, body, due=None):\n"
                "    return {'body': body, 'due': due}\n\n"
                "def list_reminders(path):\n"
                "    return []\n",
                encoding="utf-8",
            )
            (root / "runtime" / "service.py").write_text(
                "from agents.manager import ManagerAgent\n"
                "from agents.memory import MemoryAgent\n"
                "from app.tools.reminders import add_reminder, list_reminders\n\n"
                "class NudgeRuntime:\n"
                "    def __init__(self):\n"
                "        self._memory = MemoryAgent()\n"
                "        self._manager = ManagerAgent()\n\n"
                "    async def remind(self, body):\n"
                "        return add_reminder('reminders.json', body)\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            context = chronicle.context(
                "How does ManagerAgent orchestrate reminders and memory?",
                token_budget=900,
                remember=False,
            )

            names = {symbol.name for symbol in context.selected_symbols}
            self.assertIn("ManagerAgent", names)
            self.assertIn("NudgeRuntime", names)
            self.assertIn("ManagerAgent.run", names)
            self.assertTrue({"MemoryAgent", "add_reminder", "list_reminders"} & names)
            self.assertIn("add_reminder", context.compressed_context)
            self.assertIn("no direct evidence here for reminder", context.compressed_context)

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

    def test_ab_winner_summary_flags_quality_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(
                "class RequestContext:\n"
                "    def push(self):\n"
                "        return True\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            baseline = chronicle.validate_output(
                "RequestContext is defined in app.py as the RequestContext class.",
                chronicle.context("Where is RequestContext defined?", token_budget=400, remember=False),
            )
            chronicle_result = ValidationResult(
                valid=False,
                issues=baseline.issues,
                grounded=False,
                confidence=0.2,
                grounded_references=baseline.grounded_references,
                ungrounded_references=baseline.ungrounded_references,
            )

            summary = chronicle._ab_winner_summary(
                baseline_validation=baseline,
                chronicle_validation=chronicle_result,
                reduction_percent=75.0,
            )

            self.assertIn("Baseline answer appears more grounded", summary)

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

    def test_mcp_server_exposes_prompt_packet_and_session_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            server = ChronicleMCPServer(repo_path=root)
            session = server.handle("session_start", {"session_id": "demo-session"})
            packet = server.handle(
                "prepare_prompt_packet",
                {
                    "query": "Where is ManagerAgent.run defined?",
                    "token_budget": 450,
                    "session_id": session["session_id"],
                },
            )
            stored = server.handle("session_show", {"session_id": session["session_id"]})

            self.assertEqual(session["session_id"], "demo-session")
            self.assertEqual(packet["query"], "Where is ManagerAgent.run defined?")
            self.assertIn("selected_symbols", packet)
            self.assertEqual(stored["session_id"], "demo-session")

    def test_prepare_replay_explain_inspect_and_mcp_prepare_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src" / "auth"
            tests = root / "tests"
            src.mkdir(parents=True)
            tests.mkdir()
            (src / "service.py").write_text(
                "class AuthService:\n"
                "    def refresh_token(self, token):\n"
                "        return token\n",
                encoding="utf-8",
            )
            (tests / "test_service.py").write_text(
                "from src.auth.service import AuthService\n\n"
                "def test_refresh_token():\n"
                "    assert AuthService().refresh_token('t') == 't'\n",
                encoding="utf-8",
            )

            chronicle = Chronicle(repo_path=root)
            prepared = chronicle.prepare("Fix auth token refresh bug", target="codex")
            replayed = chronicle.replay(latest=True)
            explained = chronicle.explain(latest=True)
            inspected_file = chronicle.inspect_file("src/auth/service.py")
            inspected_symbol = chronicle.inspect_symbol("AuthService.refresh_token")
            mcp_prepared = ChronicleMCPServer(repo_path=root).handle(
                "prepare",
                {"query": "Fix auth token refresh bug", "target": "codex"},
            )

            self.assertTrue((root / "chronicle_logs" / "runs").exists())
            self.assertEqual(prepared["run_id"], replayed["run_id"])
            self.assertIn("src/auth/service.py", prepared["selected_files"])
            self.assertIn("selection_reasons", explained)
            self.assertEqual(inspected_file["file_path"], "src/auth/service.py")
            self.assertEqual(inspected_symbol["symbol"]["name"], "AuthService.refresh_token")
            self.assertIn("saved", mcp_prepared)

    def test_status_review_handoff_and_mcp_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src" / "auth"
            tests = root / "tests"
            src.mkdir(parents=True)
            tests.mkdir()
            (src / "service.py").write_text(
                "class AuthService:\n"
                "    def refresh_token(self, token):\n"
                "        return token\n",
                encoding="utf-8",
            )
            (tests / "test_service.py").write_text(
                "from src.auth.service import AuthService\n\n"
                "def test_refresh_token():\n"
                "    assert AuthService().refresh_token('t') == 't'\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "chronicle@example.com"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Chronicle"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)

            chronicle = Chronicle(repo_path=root)
            prepared = chronicle.prepare("Fix auth token refresh bug")
            (src / "service.py").write_text(
                "class AuthService:\n"
                "    def refresh_token(self, token):\n"
                "        if not token:\n"
                "            return 'missing'\n"
                "        return token\n",
                encoding="utf-8",
            )

            status = chronicle.status()
            review = chronicle.review()
            handoff = chronicle.handoff(tests="pytest tests/test_service.py passed", notes=["Ready for reviewer."])
            server = ChronicleMCPServer(repo_path=root)
            mcp_status = server.handle("status", {})
            mcp_review = server.handle("review", {})
            mcp_handoff = server.handle("handoff", {"tests": "pytest passed"})

            self.assertEqual(prepared["task"], "Fix auth token refresh bug")
            self.assertIn("src/auth/service.py", status["changed_files"])
            self.assertIn("src/auth/service.py", review["changed_files"])
            self.assertIn("tests/test_service.py", review["related_tests"])
            self.assertIn("handoff_md", handoff["saved"])
            self.assertIn("src/auth/service.py", mcp_status["changed_files"])
            self.assertIn("review_md", mcp_review["saved"])
            self.assertIn("handoff_md", mcp_handoff["saved"])

    def test_review_finds_integration_tests_by_content_not_only_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "chronicle"
            tests = root / "tests"
            package.mkdir(parents=True)
            tests.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "api.py").write_text(
                "class Chronicle:\n"
                "    def prepare(self):\n"
                "        return 'prepare'\n"
                "    def review(self):\n"
                "        return 'review'\n",
                encoding="utf-8",
            )
            (package / "cli.py").write_text(
                "def main():\n"
                "    return 'cli prepare review handoff'\n",
                encoding="utf-8",
            )
            (package / "integrations.py").write_text(
                "class ChronicleMCPServer:\n"
                "    def handle(self):\n"
                "        return 'status review handoff'\n",
                encoding="utf-8",
            )
            (tests / "test_reasoning.py").write_text(
                "from chronicle.api import Chronicle\n"
                "from chronicle import cli as chronicle_cli\n"
                "from chronicle.integrations import ChronicleMCPServer\n\n"
                "def test_prepare_replay_explain_inspect_and_mcp_prepare_flow():\n"
                "    assert Chronicle().prepare()\n\n"
                "def test_status_review_handoff_and_mcp_flow():\n"
                "    assert ChronicleMCPServer().handle()\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "chronicle@example.com"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Chronicle"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)

            (package / "api.py").write_text(
                "class Chronicle:\n"
                "    def prepare(self):\n"
                "        return 'prepare packet'\n"
                "    def review(self):\n"
                "        return 'review packet'\n"
                "    def handoff(self):\n"
                "        return 'handoff packet'\n",
                encoding="utf-8",
            )

            review = Chronicle(repo_path=root).review()

            self.assertIn("src/chronicle/api.py", review["changed_files"])
            self.assertIn("tests/test_reasoning.py", review["related_tests"])

    def test_handoff_task_anchors_call_chain_on_handoff_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src" / "chronicle"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "api.py").write_text(
                "class Chronicle:\n"
                "    def review(self):\n"
                "        self.index()\n"
                "        self._render_review_packet()\n"
                "    def handoff(self):\n"
                "        status = self.status()\n"
                "        payload = self._load_artifact_payload()\n"
                "        self._render_handoff_packet()\n"
                "        self._write_latest_artifact()\n"
                "        return status, payload\n"
                "    def status(self):\n"
                "        return 'status'\n"
                "    def index(self):\n"
                "        return 'index'\n"
                "    def _load_artifact_payload(self):\n"
                "        return {}\n"
                "    def _render_review_packet(self):\n"
                "        return 'review'\n"
                "    def _render_handoff_packet(self):\n"
                "        return 'handoff'\n"
                "    def _write_latest_artifact(self):\n"
                "        return 'latest'\n",
                encoding="utf-8",
            )

            context = Chronicle(repo_path=root).context(
                "is the handoff logic well drafted?",
                token_budget=1800,
            )

            self.assertIsNotNone(context.call_chain)
            assert context.call_chain is not None
            self.assertEqual(context.call_chain.entry_symbol, "Chronicle.handoff")
            self.assertIn("Chronicle.handoff", context.call_chain.summary)
            self.assertIn("Direct calls from Chronicle.handoff", context.call_chain.summary)
            self.assertIn("Chronicle._render_handoff_packet", context.call_chain.summary)
            self.assertIn("Primary context:", context.compressed_context)
            self.assertIn("Supporting context:", context.compressed_context)
            self.assertIn("Verification context:", context.compressed_context)

    def test_handoff_warns_when_review_is_stale_or_tests_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "feature.py").write_text(
                "def prepare_flow():\n"
                "    return 'prepare'\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "chronicle@example.com"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Chronicle"], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True, text=True)

            chronicle = Chronicle(repo_path=root)
            chronicle.prepare("Prepare flow")
            (src / "feature.py").write_text(
                "def prepare_flow():\n"
                "    return 'prepare'\n\n"
                "def review_flow():\n"
                "    return 'review'\n",
                encoding="utf-8",
            )
            chronicle.review("Review feature changes")
            (src / "extra.py").write_text(
                "def handoff_flow():\n"
                "    return 'handoff'\n",
                encoding="utf-8",
            )

            handoff = chronicle.handoff("Different handoff task")

            self.assertIn("Tests were not recorded", " ".join(handoff["warnings"]))
            self.assertIn("Latest review does not cover changed files", " ".join(handoff["warnings"]))
            self.assertIn("Handoff task differs", " ".join(handoff["warnings"]))
            self.assertIn(
                "## Handoff Warnings",
                (root / "chronicle_logs" / "runs" / handoff["handoff_id"] / "handoff.md").read_text(encoding="utf-8"),
            )

    def test_mcp_stdio_server_supports_initialize_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.py").write_text(
                "class ManagerAgent:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            proc = subprocess.Popen(
                [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-m", "chronicle.mcp_stdio", "--repo", str(root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=env,
            )
            assert proc.stdin is not None
            assert proc.stdout is not None

            def rpc(payload: dict[str, object]) -> dict[str, object]:
                body = json.dumps(payload).encode("utf-8") + b"\n"
                proc.stdin.write(body)
                proc.stdin.flush()
                response_body = proc.stdout.readline()
                return json.loads(response_body.decode("utf-8"))

            try:
                initialize = rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {},
                            "clientInfo": {"name": "chronicle-test", "version": "1.0"},
                        },
                    }
                )
                tools = rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    }
                )
                resources = rpc(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "resources/list",
                        "params": {},
                    }
                )

                self.assertEqual(initialize["result"]["serverInfo"]["name"], "chronicle-mcp")
                self.assertTrue(any(tool["name"] == "context" for tool in tools["result"]["tools"]))
                self.assertTrue(any(resource["uri"] == "chronicle://server-info" for resource in resources["result"]["resources"]))
            finally:
                proc.kill()
                proc.wait(timeout=5)
                if proc.stdin is not None:
                    proc.stdin.close()
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()

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
        self.assertIn("<title>Chronicle</title>", root_response.text)
        self.assertIn(">Chronicle<", root_response.text)
        self.assertIn("Run demo", root_response.text)
        self.assertIn("Result model", root_response.text)
        self.assertIn("repo_url", root_response.text)
        self.assertIn("Use as SDK", root_response.text)
        self.assertIn("Use Chronicle as a Python SDK", root_response.text)
        self.assertIn("Install from PyPI", root_response.text)
        self.assertIn("pip install chronicle-sdk", root_response.text)
        self.assertNotIn(">Docs<", root_response.text)
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
            self.assertIn("id=\"api_key\"", root_response.text)
            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)
        finally:
            if original_key is None:
                os.environ.pop("CHRONICLE_API_KEY", None)
            else:
                os.environ["CHRONICLE_API_KEY"] = original_key

    def test_hosted_service_shows_python_repo_hint_for_indexing_errors(self) -> None:
        if importlib.util.find_spec("fastapi") is None:
            return

        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        root_response = client.get("/")

        self.assertEqual(root_response.status_code, 200)
        self.assertIn("Please use a repo with Python source files.", root_response.text)


if __name__ == "__main__":
    unittest.main()
