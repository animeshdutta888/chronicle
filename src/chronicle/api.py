from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from .core.config import ChronicleConfig
from .core.interfaces import AgentBusStore, PersistedSnapshotStore, SessionStore
from .core.models import (
    AgentRole,
    ContextPack,
    EvaluationReport,
    IndexSnapshot,
    SDKPromptPacket,
    MultiAgentContextBus,
    QueryPlan,
    SessionMemory,
    SessionMemoryHints,
    SessionTurn,
    ValidationResult,
)
from .eval.token_savings import TokenSavingsEvaluator
from .indexer.ast_parser import PythonAstParser
from .indexer.call_graph_builder import CallGraphBuilder
from .indexer.dependency_graph_builder import DependencyGraphBuilder
from .indexer.git_evolution_analyzer import GitEvolutionAnalyzer
from .indexer.repo_scanner import RepoScanner
from .indexer.symbol_extractor import SymbolExtractor
from .llm.guardrails import Guardrails
from .llm.prompts import build_answer_prompt, build_repair_prompt
from .llm.providers import OllamaError, OllamaProvider
from .llm.router import LLMRouter
from .memory.agent_bus_store import SQLiteAgentBusStore
from .memory.session_store import SQLiteSessionStore
from .memory.sqlite_store import SQLiteSnapshotStore
from .retrieval.query_planner import DeterministicQueryPlanner
from .retrieval.retrieval_orchestrator import RetrievalOrchestrator
from .retrieval.provenance import symbol_provenance
from .retrieval.token_budget import TokenBudgetManager
from .validation.output_validator import OutputValidator


class Chronicle:
    def __init__(
        self,
        repo_path: str | Path,
        index_dir: str | Path | None = None,
        snapshot_store: PersistedSnapshotStore | None = None,
        session_store: SessionStore | None = None,
        agent_bus_store: AgentBusStore | None = None,
    ) -> None:
        self.config = ChronicleConfig.from_paths(repo_path=repo_path, index_dir=index_dir)
        self.scanner = RepoScanner(
            repo_path=self.config.repo_path,
            ignored_dirs=self.config.ignored_dirs,
            file_extensions=self.config.file_extensions,
        )
        self.parser = PythonAstParser(self.config.repo_path)
        self.symbol_extractor = SymbolExtractor()
        self.call_graph_builder = CallGraphBuilder()
        self.dependency_builder = DependencyGraphBuilder()
        self.git_evolution = GitEvolutionAnalyzer(self.config.repo_path)
        self.query_planner = DeterministicQueryPlanner()
        self.retrieval = RetrievalOrchestrator(self.config)
        self.router = LLMRouter()
        self.guardrails = Guardrails()
        self.validator = OutputValidator()
        self.evaluator = TokenSavingsEvaluator()
        self.snapshot_store = snapshot_store or SQLiteSnapshotStore()
        self.session_store = session_store or SQLiteSessionStore()
        self.agent_bus_store = agent_bus_store or SQLiteAgentBusStore()
        self.budget_manager = TokenBudgetManager(self.config.default_token_budgets)

    def index(self) -> IndexSnapshot:
        files = self.scanner.scan()
        modules = self.parser.parse_files(files)
        symbols = self.symbol_extractor.extract(modules)
        dependency_graph = self.dependency_builder.build(modules)
        for symbol in symbols:
            symbol.imports = dependency_graph.get(symbol.file_path, [])
        call_graph = self.call_graph_builder.build(symbols)
        commit_changes, churn_by_file = self.git_evolution.analyze(symbols)
        snapshot = IndexSnapshot(
            repo_path=str(self.config.repo_path),
            indexed_at=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            call_graph=call_graph,
            dependency_graph=dependency_graph,
            commit_changes=commit_changes,
            churn_by_file=churn_by_file,
        )
        self._persist_snapshot(snapshot)
        return snapshot

    def plan(self, query: str):
        return self.query_planner.plan(query)

    def call_chain(
        self,
        query: str,
        *,
        token_budget: int | None = None,
        session_id: str | None = None,
        max_depth: int = 4,
    ) -> dict[str, Any]:
        snapshot = self._ensure_snapshot()
        plan = self.query_planner.plan(query)
        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        ordered_symbols = self._prefer_query_entry_symbols(context.selected_symbols, plan)
        call_chain = self.retrieval.call_chain_builder.build(
            query=query,
            snapshot=snapshot,
            selected_symbols=ordered_symbols,
            max_depth=max_depth,
        )
        return {
            "query": query,
            "session_id": session_id,
            "entry_symbol": call_chain.entry_symbol if call_chain else None,
            "max_depth": max_depth,
            "selected_symbols": [
                {
                    "name": symbol.name,
                    "file_path": symbol.file_path,
                    "start_line": symbol.start_line,
                }
                for symbol in context.selected_symbols[:8]
            ],
            "summary": call_chain.summary if call_chain else "",
            "chains": call_chain.model_dump().get("chains", []) if call_chain else [],
            "mermaid": call_chain.mermaid if call_chain else "",
            "context_estimated_tokens": context.estimated_tokens,
        }

    def start_session(self, session_id: str | None = None) -> SessionMemory:
        return self.session_store.get_or_create(
            index_dir=self.config.index_dir,
            repo_path=self.config.repo_path,
            session_id=session_id or self._new_session_id(),
        )

    def session(self, session_id: str) -> SessionMemory | None:
        return self.session_store.load(index_dir=self.config.index_dir, session_id=session_id)

    def start_agent_bus(
        self,
        *,
        root_query: str,
        bus_id: str | None = None,
        session_id: str | None = None,
    ) -> MultiAgentContextBus:
        return self.agent_bus_store.create_bus(
            self.config.index_dir,
            repo_path=self.config.repo_path,
            bus_id=bus_id or self._new_bus_id(),
            root_query=root_query,
            session_id=session_id,
        )

    def agent_bus(self, bus_id: str) -> MultiAgentContextBus | None:
        return self.agent_bus_store.load_bus(self.config.index_dir, bus_id)

    def bus_context(
        self,
        *,
        bus_id: str,
        role: AgentRole,
        query: str,
        token_budget: int | None = None,
        session_id: str | None = None,
        notes: list[str] | None = None,
    ) -> MultiAgentContextBus:
        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        self.agent_bus_store.append_phase(
            self.config.index_dir,
            bus_id=bus_id,
            role=role,
            query=query,
            token_budget=context.token_budget,
            context_pack=context,
            llm_decision=context.llm_decision,
            notes=notes,
        )
        loaded = self.agent_bus(bus_id)
        if loaded is None:
            raise RuntimeError(f"Chronicle could not reload agent bus `{bus_id}`.")
        return loaded

    def bus_handoff(
        self,
        *,
        bus_id: str,
        from_role: AgentRole,
        to_role: AgentRole,
        reason: str,
    ) -> MultiAgentContextBus:
        return self.agent_bus_store.add_handoff(
            self.config.index_dir,
            bus_id=bus_id,
            from_role=from_role,
            to_role=to_role,
            reason=reason,
        )

    def bus_validate_latest(
        self,
        *,
        bus_id: str,
        output_text: str,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        bus = self.agent_bus(bus_id)
        if bus is None or not bus.phases:
            raise ValueError(f"Chronicle could not find a latest phase for bus `{bus_id}`.")
        latest = bus.phases[-1]
        validation = self.validate_output(output_text, latest.context_pack)
        updated = self.agent_bus_store.update_latest_phase_validation(
            self.config.index_dir,
            bus_id=bus_id,
            validation=validation,
            notes=notes,
        )
        return {
            "bus": updated.model_dump() if updated else None,
            "validation": validation.model_dump(),
        }

    def bus_summary(self, bus_id: str) -> dict[str, Any]:
        bus = self.agent_bus(bus_id)
        if bus is None:
            raise ValueError(f"Chronicle could not find bus `{bus_id}`.")
        return {
            "bus_id": bus.bus_id,
            "root_query": bus.root_query,
            "session_id": bus.session_id,
            "phase_count": len(bus.phases),
            "handoff_count": len(bus.handoffs),
            "roles": [phase.role for phase in bus.phases],
            "latest_role": bus.phases[-1].role if bus.phases else None,
            "latest_query": bus.phases[-1].query if bus.phases else None,
        }

    def context(
        self,
        query: str,
        token_budget: int | None = None,
        session_id: str | None = None,
        *,
        remember: bool = True,
    ) -> ContextPack:
        snapshot = self._ensure_snapshot()
        plan = self.query_planner.plan(query)
        memory_hints = self._session_hints(session_id=session_id, create_if_missing=remember) if session_id else None
        context = self.retrieval.build_context(
            query=query,
            plan=plan,
            snapshot=snapshot,
            token_budget=token_budget,
            memory_hints=memory_hints,
        )
        context.confidence = self._refine_context_confidence(plan=plan, context=context)
        context.llm_decision = self.router.route(plan=plan, context=context)
        if remember and session_id:
            self._remember_context(session_id=session_id, plan=plan, context=context)
        return context

    def validate_output(self, output_text: str, context: ContextPack) -> ValidationResult:
        return self.validator.validate(output_text=output_text, context=context)

    def evaluate(self, query: str, token_budget: int | None = None, session_id: str | None = None) -> EvaluationReport:
        snapshot = self._ensure_snapshot()
        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        baseline = self.baseline_context(query=query, token_budget=max((token_budget or 3000) * 4, 12000))
        baseline_text = baseline.compressed_context
        plan = self.query_planner.plan(query)
        if not baseline_text.strip():
            raise ValueError(
                "Chronicle indexed no Python symbols, so token-savings evaluation is not meaningful. "
                "Run `chronicle index ...` and verify `symbol_count > 0`, or try a repository with Python source files."
            )
        return self.evaluator.evaluate(plan=plan, baseline_text=baseline_text, context=context)

    def diagnose(
        self,
        query: str | None = None,
        token_budget: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        file_count = len(self.scanner.scan())
        snapshot = self._load_snapshot()
        if snapshot is None or (file_count > 0 and not snapshot.symbols):
            snapshot = self.index()
        payload: dict[str, Any] = {
            "repo": str(self.config.repo_path),
            "index_dir": str(self.config.index_dir),
            "python_file_count": file_count,
            "symbol_count": len(snapshot.symbols),
            "commit_change_count": len(snapshot.commit_changes),
            "call_graph_nodes": len(snapshot.call_graph),
            "dependency_graph_nodes": len(snapshot.dependency_graph),
            "sample_symbols": self._sample_symbols(snapshot),
            "warnings": self._warnings(snapshot, file_count),
        }
        payload["health"] = self._index_health(snapshot=snapshot, file_count=file_count)
        if session_id:
            payload["session"] = self._session_payload(session_id=session_id)
        if query and snapshot.symbols:
            plan = self.query_planner.plan(query)
            context = self.context(query=query, token_budget=token_budget, session_id=session_id, remember=False)
            payload["query"] = query
            payload["context_estimated_tokens"] = context.estimated_tokens
            payload["context_confidence"] = context.confidence
            payload["selected_symbols"] = [symbol.name for symbol in context.selected_symbols[:5]]
            payload["top_matches"] = self._top_matches(context)
            payload["query_diagnosis"] = self._query_diagnosis(plan=plan, context=context)
            if context.call_chain:
                payload["call_chain_summary"] = context.call_chain.summary
            if context.patch_context:
                payload["patch_context_summary"] = context.patch_context.summary
            if context.llm_brief:
                payload["llm_brief"] = context.llm_brief.model_dump()
            payload["human_summary"] = self._human_summary(payload["health"], payload["query_diagnosis"])
        elif query:
            payload["query"] = query
            payload["retrieval_error"] = (
                "Chronicle cannot test retrieval because no Python symbols were indexed for this repository."
            )
            payload["human_summary"] = (
                "Chronicle found the repository but could not extract Python symbols, so this repo is not ready for evaluation yet."
            )
        else:
            payload["human_summary"] = self._human_summary(payload["health"], None)
        return payload

    def demo(self, query: str, token_budget: int | None = None, session_id: str | None = None) -> dict[str, Any]:
        self.index()
        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        report = self.evaluate(query=query, token_budget=token_budget, session_id=session_id)
        llm_payload = self._llm_payload_preview(query=query, context=context)
        repo_insight = self._demo_repo_insight(query=query, context=context)
        return {
            "repo": str(self.config.repo_path),
            "index": self.diagnose(session_id=session_id),
            "query": query,
            "human_summary": self._demo_human_summary(query=query, context=context, report=report),
            "repo_insight": repo_insight,
            "llm_readiness": {
                "send_to_llm": bool(context.llm_decision.call_llm) if context.llm_decision else False,
                "reason": self._demo_llm_reason(context=context),
                "query_strategy": self._demo_query_strategy(query=query, context=context),
                "context_strategy": self._demo_context_strategy(context=context),
                "recommended_next_step": self._demo_next_step(query=query, context=context),
                "model_class": context.llm_decision.model_class if context.llm_decision else "local",
                "max_input_tokens": context.llm_decision.max_input_tokens if context.llm_decision else context.estimated_tokens,
                "max_output_tokens": context.llm_decision.max_output_tokens if context.llm_decision else 0,
                "payload_preview": llm_payload,
            },
            "context": {
                "estimated_tokens": context.estimated_tokens,
                "confidence": context.confidence,
                "session_id": context.session_id,
                "memory_summary": context.memory_summary.model_dump() if context.memory_summary else None,
                "selected_symbols": [
                    {
                        "name": symbol.name,
                        "file_path": symbol.file_path,
                        "start_line": symbol.start_line,
                    }
                    for symbol in context.selected_symbols[:5]
                ],
                "llm_decision": context.llm_decision.model_dump() if context.llm_decision else None,
            },
            "evaluation": {
                **report.model_dump(),
                "human_summary": (
                    f"Chronicle reduced estimated input from {report.baseline_tokens} to {report.chronicle_tokens} tokens "
                    f"({report.token_reduction_percent:.2f}% reduction) with {report.benchmark_confidence} confidence."
                ),
            },
        }

    def prepare_prompt_packet(
        self,
        query: str,
        token_budget: int | None = None,
        session_id: str | None = None,
        *,
        include_prompt: bool = True,
    ) -> SDKPromptPacket:
        self.index()
        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        response_policy = self._response_policy(query=query, context=context)
        prompt = None
        if include_prompt and context.llm_decision and context.llm_decision.call_llm:
            prompt = build_answer_prompt(
                query=self._llm_query_envelope(query=query, context=context, response_policy=response_policy),
                context=context.compressed_context,
            )
        report = self.evaluate(query=query, token_budget=token_budget, session_id=session_id)
        return SDKPromptPacket(
            query=query,
            repo_path=str(self.config.repo_path),
            should_call_llm=bool(context.llm_decision.call_llm) if context.llm_decision else False,
            human_summary=self._demo_human_summary(query=query, context=context, report=report),
            compressed_context=context.compressed_context,
            estimated_input_tokens=context.estimated_tokens,
            response_policy=response_policy,
            prompt=prompt,
            selected_symbols=[symbol.name for symbol in context.selected_symbols[:8]],
            selected_files=list(dict.fromkeys(symbol.file_path for symbol in context.selected_symbols[:8])),
            llm_decision_reason=context.llm_decision.reason if context.llm_decision else None,
        )

    def prepare(
        self,
        query: str,
        token_budget: int | None = None,
        session_id: str | None = None,
        *,
        target: str = "generic",
        view: str = "compact",
        auto_index: bool = True,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        if force_reindex:
            self.index()
        elif auto_index:
            self._ensure_snapshot()
        elif self._load_snapshot() is None:
            raise ValueError("Chronicle has no index yet. Run `chronicle index` or omit `--no-auto-index`.")

        context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        baseline = self.baseline_context(query=query, token_budget=max((token_budget or 3000) * 4, 12000))
        selected_files = self._selected_files(context)
        related_tests = self._related_tests(context)
        selection_reasons = self._selection_reasons(context=context, query=query)
        warnings = self._missing_context_warnings(query=query, context=context, selected_files=selected_files, related_tests=related_tests)
        readiness = self._prepare_readiness(context=context, warnings=warnings)
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        run_dir = self.config.index_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        context_md_path = run_dir / "context.md"
        run_json_path = run_dir / "run.json"
        context_packet = self._render_agent_packet(
            query=query,
            target=target,
            context=context,
            selected_files=selected_files,
            related_tests=related_tests,
            warnings=warnings,
        )
        token_stats = {
            "estimated_raw_tokens": baseline.estimated_tokens,
            "packet_tokens": context.estimated_tokens,
            "reduction_percent": round(
                max(0.0, ((baseline.estimated_tokens - context.estimated_tokens) / max(baseline.estimated_tokens, 1)) * 100),
                2,
            ),
        }
        run = {
            "run_id": run_id,
            "command": "prepare",
            "task": query,
            "repo_path": str(self.config.repo_path),
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "selected_files": selected_files,
            "selected_symbols": [symbol.name for symbol in context.selected_symbols],
            "related_tests": related_tests,
            "excluded_candidates": context.excluded_symbols,
            "selection_reasons": selection_reasons,
            "missing_context_warnings": warnings,
            "risk_warnings": [],
            "readiness": readiness,
            "token_stats": token_stats,
            "context_packet": context_packet,
            "context": context.model_dump() if view == "full" else None,
            "output_files": {
                "context_md": str(context_md_path),
                "run_json": str(run_json_path),
            },
        }
        context_md_path.write_text(context_packet, encoding="utf-8")
        run_json_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        latest_path = self.config.index_dir / "runs" / "latest.json"
        latest_path.write_text(json.dumps({"run_id": run_id, "run_json": str(run_json_path)}, indent=2), encoding="utf-8")
        return run if view == "full" else self._compact_prepare_run(run)

    def replay(self, *, run_id: str | None = None, latest: bool = False, list_runs: bool = False, view: str = "compact") -> dict[str, Any]:
        runs_dir = self.config.index_dir / "runs"
        if list_runs:
            runs = []
            for run_json in sorted(runs_dir.glob("run_*/run.json"), reverse=True):
                data = json.loads(run_json.read_text(encoding="utf-8"))
                runs.append(
                    {
                        "run_id": data.get("run_id"),
                        "task": data.get("task"),
                        "target": data.get("target"),
                        "generated_at": data.get("generated_at"),
                        "readiness": data.get("readiness"),
                    }
                )
            return {"runs": runs}
        if latest:
            latest_path = runs_dir / "latest.json"
            if not latest_path.exists():
                raise ValueError("Chronicle has no prepared runs yet.")
            run_path = Path(json.loads(latest_path.read_text(encoding="utf-8"))["run_json"])
        elif run_id:
            run_path = runs_dir / run_id / "run.json"
        else:
            raise ValueError("Use `--latest`, `--list`, or `--run <run_id>`.")
        if not run_path.exists():
            raise ValueError(f"Chronicle could not find prepared run `{run_id or 'latest'}`.")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        return run if view == "full" else self._compact_prepare_run(run)

    def explain(self, *, run_id: str | None = None, latest: bool = False, view: str = "compact") -> dict[str, Any]:
        run = self.replay(run_id=run_id, latest=latest, view="full")
        explanation = {
            "run_id": run.get("run_id"),
            "task": run.get("task"),
            "readiness": run.get("readiness"),
            "selection_reasons": run.get("selection_reasons", {}),
            "warnings": run.get("missing_context_warnings", []),
            "excluded_candidates": run.get("excluded_candidates", []),
            "token_stats": run.get("token_stats", {}),
        }
        if view == "full":
            return explanation
        return {
            "run_id": explanation["run_id"],
            "task": explanation["task"],
            "readiness": explanation["readiness"],
            "selection_reasons": dict(list(explanation["selection_reasons"].items())[:8]),
            "warnings": explanation["warnings"],
        }

    def inspect_file(self, file_path: str, *, view: str = "compact") -> dict[str, Any]:
        snapshot = self._ensure_snapshot()
        symbols = [symbol for symbol in snapshot.symbols if symbol.file_path == file_path]
        imports = sorted({item for symbol in symbols for item in symbol.imports})
        incoming = sorted(
            {
                symbol.file_path
                for symbol in snapshot.symbols
                if file_path in snapshot.dependency_graph.get(symbol.file_path, []) and symbol.file_path != file_path
            }
        )
        payload = {
            "file_path": file_path,
            "indexed_at": snapshot.indexed_at,
            "symbols": [
                {"name": symbol.name, "type": symbol.type, "start_line": symbol.start_line, "end_line": symbol.end_line}
                for symbol in symbols
            ],
            "imports": imports,
            "outgoing_dependencies": snapshot.dependency_graph.get(file_path, []),
            "incoming_references": incoming,
            "related_tests": self._related_tests_for_files(snapshot, [file_path]),
        }
        if view == "full":
            payload["symbol_ids"] = [symbol.id for symbol in symbols]
        return payload

    def inspect_symbol(self, symbol_name: str, *, view: str = "compact") -> dict[str, Any]:
        snapshot = self._ensure_snapshot()
        matches = [symbol for symbol in snapshot.symbols if symbol.name == symbol_name or symbol.name.endswith(f".{symbol_name}")]
        if not matches:
            raise ValueError(f"Chronicle could not find symbol `{symbol_name}`.")
        symbol = matches[0]
        callers = sorted(source_id for source_id, callees in snapshot.call_graph.items() if symbol.id in callees)
        payload = {
            "symbol": {
                "name": symbol.name,
                "type": symbol.type,
                "file_path": symbol.file_path,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "signature": symbol.signature,
            },
            "callers": callers,
            "callees": snapshot.call_graph.get(symbol.id, []),
            "related_tests": self._related_tests_for_files(snapshot, [symbol.file_path]),
            "selection_hints": [
                f"contains symbol {symbol.name}",
                f"located in {symbol.file_path}",
            ],
        }
        if view == "full":
            payload["body"] = symbol.body
            payload["imports"] = symbol.imports
        return payload

    def status(self, *, view: str = "compact") -> dict[str, Any]:
        snapshot = self._load_snapshot()
        changed_files = self._changed_python_files(include_untracked=True)
        latest_prepare = self._latest_artifact("runs")
        latest_review = self._latest_artifact("reviews")
        latest_handoff = self._latest_artifact("handoffs")
        indexed = snapshot is not None and bool(snapshot.symbols)
        stale = bool(changed_files)
        payload = {
            "repo": str(self.config.repo_path),
            "index_dir": str(self.config.index_dir),
            "indexed": indexed,
            "index_status": "stale" if indexed and stale else "ready" if indexed else "missing",
            "indexed_at": snapshot.indexed_at if snapshot else None,
            "symbol_count": len(snapshot.symbols) if snapshot else 0,
            "changed_files": changed_files,
            "latest_prepare": latest_prepare,
            "latest_review": latest_review,
            "latest_handoff": latest_handoff,
            "next_steps": self._status_next_steps(indexed=indexed, changed_files=changed_files, latest_prepare=latest_prepare),
        }
        return payload if view == "full" else payload

    def review(
        self,
        query: str = "Review recent code changes and impacted tests",
        *,
        token_budget: int | None = None,
        session_id: str | None = None,
        view: str = "compact",
    ) -> dict[str, Any]:
        snapshot = self.index()
        changed_files = self._changed_python_files(include_untracked=True)
        changed_symbols = self._symbols_for_files(snapshot, changed_files)
        related_tests = self._related_tests_for_files(snapshot, changed_files)
        related_files = self._related_files_for_symbols(snapshot, changed_symbols, changed_files)
        latest_prepare = self._latest_artifact("runs")
        warnings = self._review_warnings(
            changed_files=changed_files,
            related_tests=related_tests,
            latest_prepare=latest_prepare,
        )
        context = self.context(query=query, token_budget=token_budget, session_id=session_id, remember=False)
        review_id = f"review_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        review_dir = self.config.index_dir / "reviews" / review_id
        review_dir.mkdir(parents=True, exist_ok=True)
        review_md_path = review_dir / "review.md"
        review_json_path = review_dir / "review.json"
        review_md = self._render_review_packet(
            query=query,
            changed_files=changed_files,
            changed_symbols=[symbol.name for symbol in changed_symbols],
            related_files=related_files,
            related_tests=related_tests,
            warnings=warnings,
            context=context,
        )
        payload = {
            "review_id": review_id,
            "command": "review",
            "query": query,
            "repo_path": str(self.config.repo_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "changed_files": changed_files,
            "changed_symbols": [symbol.name for symbol in changed_symbols],
            "related_files": related_files,
            "related_tests": related_tests,
            "warnings": warnings,
            "latest_prepare": latest_prepare,
            "context": context.model_dump() if view == "full" else None,
            "review_packet": review_md,
            "output_files": {
                "review_md": str(review_md_path),
                "review_json": str(review_json_path),
            },
        }
        review_md_path.write_text(review_md, encoding="utf-8")
        review_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_latest_artifact("reviews", review_id, review_json_path)
        return payload if view == "full" else self._compact_review(payload)

    def handoff(
        self,
        task: str | None = None,
        *,
        tests: str | None = None,
        notes: list[str] | None = None,
        view: str = "compact",
    ) -> dict[str, Any]:
        status = self.status(view="full")
        latest_prepare = self._latest_artifact("runs")
        latest_review = self._latest_artifact("reviews")
        prepare_payload = self._load_artifact_payload(latest_prepare)
        review_payload = self._load_artifact_payload(latest_review)
        resolved_task = task or (prepare_payload or {}).get("task") or "Chronicle handoff"
        handoff_warnings = self._handoff_warnings(
            task=resolved_task,
            status=status,
            prepare_payload=prepare_payload,
            review_payload=review_payload,
            tests=tests,
        )
        handoff_id = f"handoff_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        handoff_dir = self.config.index_dir / "handoffs" / handoff_id
        handoff_dir.mkdir(parents=True, exist_ok=True)
        handoff_md_path = handoff_dir / "handoff.md"
        handoff_json_path = handoff_dir / "handoff.json"
        handoff_md = self._render_handoff_packet(
            task=resolved_task,
            status=status,
            prepare_payload=prepare_payload,
            review_payload=review_payload,
            tests=tests,
            notes=notes or [],
            warnings=handoff_warnings,
        )
        payload = {
            "handoff_id": handoff_id,
            "command": "handoff",
            "task": resolved_task,
            "repo_path": str(self.config.repo_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "changed_files": status["changed_files"],
            "latest_prepare": latest_prepare,
            "latest_review": latest_review,
            "tests": tests,
            "notes": notes or [],
            "warnings": handoff_warnings,
            "handoff_packet": handoff_md,
            "output_files": {
                "handoff_md": str(handoff_md_path),
                "handoff_json": str(handoff_json_path),
            },
        }
        handoff_md_path.write_text(handoff_md, encoding="utf-8")
        handoff_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_latest_artifact("handoffs", handoff_id, handoff_json_path)
        return payload if view == "full" else self._compact_handoff(payload)

    def baseline_context(self, query: str, token_budget: int | None = None) -> ContextPack:
        snapshot = self._ensure_snapshot()
        plan = self.query_planner.plan(query)
        budget = token_budget or max(self.config.default_token_budgets.get(plan.intent, 3000) * 4, 12000)
        file_scores = self._score_files_for_baseline(plan=plan, snapshot=snapshot)
        selected_files = [file_path for file_path, score in sorted(file_scores.items(), key=lambda item: item[1], reverse=True) if score > 0]
        fallback_source_files = sorted({symbol.file_path for symbol in snapshot.symbols if symbol.file_path.startswith("src/") or symbol.file_path.endswith(".py")})
        if not selected_files:
            selected_files = fallback_source_files[:3]
        else:
            for file_path in fallback_source_files:
                if file_path not in selected_files:
                    selected_files.append(file_path)
                if len(selected_files) >= 3:
                    break

        sections: list[str] = []
        selected_symbols: list[Any] = []
        selected_file_set: set[str] = set()
        for file_path in selected_files:
            content = self._read_repo_file(file_path)
            if not content.strip():
                continue
            chunk = f"File: {file_path}\n{content}"
            tentative = "\n\n".join(sections + [chunk]) if sections else chunk
            if sections and not self.budget_manager.fits(tentative, budget):
                continue
            sections.append(chunk)
            selected_file_set.add(file_path)
            selected_symbols.extend(symbol for symbol in snapshot.symbols if symbol.file_path == file_path)
            if self.budget_manager.estimate_tokens("\n\n".join(sections)) >= budget:
                break

        context_text = "\n\n".join(sections).strip()
        estimated_tokens = self.budget_manager.estimate_tokens(context_text)
        provenance = [
            symbol_provenance(symbol, "naive full-file baseline context", file_scores.get(symbol.file_path, 0.5))
            for symbol in selected_symbols[:50]
        ]
        return ContextPack(
            query=query,
            token_budget=budget,
            selected_symbols=selected_symbols,
            selected_commits=[],
            compressed_context=context_text,
            estimated_tokens=estimated_tokens,
            provenance=provenance,
            confidence=0.65 if selected_symbols else 0.0,
            ranking_scores={symbol.id: file_scores.get(symbol.file_path, 0.0) for symbol in selected_symbols},
            excluded_symbols=[],
            llm_decision=None,
        )

    def ab_test(
        self,
        query: str,
        model: str,
        token_budget: int | None = None,
        baseline_token_budget: int | None = None,
        llm_provider: OllamaProvider | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        provider = llm_provider or OllamaProvider()
        chronicle_context = self.context(query=query, token_budget=token_budget, session_id=session_id)
        baseline_context = self.baseline_context(query=query, token_budget=baseline_token_budget)

        try:
            baseline_report = self._answer_with_optional_repair(
                provider=provider,
                model=model,
                query=query,
                context=baseline_context,
                label="baseline",
            )
            chronicle_report = self._answer_with_optional_repair(
                provider=provider,
                model=model,
                query=query,
                context=chronicle_context,
                label="chronicle",
            )
        except OllamaError as exc:
            raise RuntimeError(str(exc)) from exc
        baseline_answer = baseline_report["answer"]
        chronicle_answer = chronicle_report["answer"]
        baseline_validation = ValidationResult.model_validate(baseline_report["validation"])
        chronicle_validation = ValidationResult.model_validate(chronicle_report["validation"])
        if session_id:
            self._remember_validation(
                session_id=session_id,
                validation=chronicle_validation,
                notes=[
                    f"ab-test model={model}",
                    f"answer_similarity={self._token_overlap(baseline_answer, chronicle_answer)}",
                ],
            )
        comparison = {
            "same_or_better_grounding": chronicle_validation.grounded
            and (
                not baseline_validation.grounded
                or chronicle_validation.confidence >= baseline_validation.confidence - 0.05
            ),
            "both_grounded": baseline_validation.grounded and chronicle_validation.grounded,
            "answer_similarity": self._token_overlap(baseline_answer, chronicle_answer),
            "input_token_reduction_percent": round(
                max(
                    0.0,
                    ((baseline_context.estimated_tokens - chronicle_context.estimated_tokens) / max(baseline_context.estimated_tokens, 1)) * 100,
                ),
                2,
            ),
        }
        return {
            "query": query,
            "model": model,
            "baseline": baseline_report,
            "chronicle": chronicle_report,
            "comparison": {
                **comparison,
                "winner_summary": self._ab_winner_summary(
                    baseline_validation=baseline_validation,
                    chronicle_validation=chronicle_validation,
                    reduction_percent=comparison["input_token_reduction_percent"],
                ),
            },
        }

    def _persist_snapshot(self, snapshot: IndexSnapshot) -> None:
        self.snapshot_store.save(snapshot=snapshot, index_dir=self.config.index_dir)

    def _selected_files(self, context: ContextPack) -> list[str]:
        return list(dict.fromkeys(symbol.file_path for symbol in context.selected_symbols))

    def _related_tests(self, context: ContextPack) -> list[str]:
        snapshot = self._ensure_snapshot()
        return self._related_tests_for_files(snapshot, self._selected_files(context))

    def _related_tests_for_files(self, snapshot: IndexSnapshot, files: list[str]) -> list[str]:
        file_set = set(files)
        test_files = sorted(
            {
                symbol.file_path
                for symbol in snapshot.symbols
                if Path(symbol.file_path).name.startswith("test_") or "/tests/" in f"/{symbol.file_path}"
            }
        )
        if not file_set:
            return test_files[:4]
        scored: list[tuple[int, str]] = []
        stems = {Path(file_path).stem.replace("test_", "") for file_path in file_set}
        changed_symbols = self._symbols_for_files(snapshot, list(file_set))
        search_terms = self._test_search_terms(files=list(file_set), symbols=changed_symbols)
        for test_file in test_files:
            test_stem = Path(test_file).stem.replace("test_", "")
            score = 0
            if test_file in file_set:
                score += 6
            if test_stem in stems:
                score += 4
            if any(stem and stem in test_file for stem in stems):
                score += 2
            content = self._read_repo_file(test_file).lower()
            if content:
                score += self._score_test_content(content=content, terms=search_terms)
            if score:
                scored.append((score, test_file))
        return [file_path for _, file_path in sorted(scored, key=lambda item: (-item[0], item[1]))[:4]]

    def _test_search_terms(self, *, files: list[str], symbols: list[Any]) -> set[str]:
        terms: set[str] = set()
        generic = {"src", "chronicle", "py", "test", "tests", "__init__"}
        for file_path in files:
            path = Path(file_path)
            module = file_path[:-3].replace("/", ".") if file_path.endswith(".py") else file_path.replace("/", ".")
            terms.add(module.lower())
            terms.add(path.stem.lower())
            terms.update(part.lower() for part in path.parts if len(part) > 2)
        for symbol in symbols:
            name = symbol.name.lower()
            terms.add(name)
            terms.add(name.split(".")[-1])
            terms.update(part for part in name.replace("_", ".").split(".") if len(part) > 2)
        return {term for term in terms if len(term) > 2 and term not in generic}

    def _score_test_content(self, *, content: str, terms: set[str]) -> int:
        score = 0
        for term in terms:
            if term not in content:
                continue
            if "." in term or "_" in term:
                score += 3
            else:
                score += 1
        return min(score, 12)

    def _selection_reasons(self, *, context: ContextPack, query: str) -> dict[str, list[str]]:
        query_terms = {term.lower() for term in query.replace("_", " ").replace(".", " ").split() if len(term) > 2}
        reasons: dict[str, list[str]] = {}
        provenance_by_file: dict[str, list[str]] = {}
        for record in context.provenance:
            provenance_by_file.setdefault(record.file_path, []).append(record.reason)
        for symbol in context.selected_symbols:
            file_reasons = reasons.setdefault(symbol.file_path, [])
            symbol_text = f"{symbol.name} {symbol.file_path}".lower().replace("_", " ").replace(".", " ")
            matched_terms = sorted(term for term in query_terms if term in symbol_text)
            if matched_terms:
                file_reasons.append(f"matches task concepts: {', '.join(matched_terms[:4])}")
            file_reasons.append(f"contains symbol {symbol.name}")
            for reason in provenance_by_file.get(symbol.file_path, [])[:2]:
                file_reasons.append(reason)
        return {file_path: list(dict.fromkeys(file_reasons))[:5] for file_path, file_reasons in reasons.items()}

    def _missing_context_warnings(
        self,
        *,
        query: str,
        context: ContextPack,
        selected_files: list[str],
        related_tests: list[str],
    ) -> list[str]:
        warnings: list[str] = []
        lowered = query.lower()
        if not related_tests:
            warnings.append("No related test file found.")
        if "config" in lowered and not any("config" in file_path or "settings" in file_path for file_path in selected_files):
            warnings.append("Query mentions config/settings, but no config file was selected.")
        if "retry" in lowered and not any("retry" in symbol.name.lower() for symbol in context.selected_symbols):
            warnings.append("Query mentions retry, but no retry-related symbol was found.")
        if context.estimated_tokens >= context.token_budget:
            warnings.append("Token budget is tight; some lower-ranked context may have been compressed or excluded.")
        if len(selected_files) == 1 and len(context.selected_symbols) > 3:
            warnings.append("Selected context is dominated by one file; coverage may be narrow.")
        return warnings

    def _prepare_readiness(self, *, context: ContextPack, warnings: list[str]) -> dict[str, str]:
        if context.confidence >= 0.75 and not warnings:
            return {"level": "high", "reason": "Relevant symbols were found with no deterministic warnings."}
        if context.confidence >= 0.45:
            reason = "Relevant files or symbols were found"
            if warnings:
                reason += f", with {len(warnings)} warning(s)."
            else:
                reason += "."
            return {"level": "medium", "reason": reason}
        return {"level": "low", "reason": "Chronicle found weak matching evidence for this task."}

    def _render_agent_packet(
        self,
        *,
        query: str,
        target: str,
        context: ContextPack,
        selected_files: list[str],
        related_tests: list[str],
        warnings: list[str],
    ) -> str:
        lines = [
            "# Chronicle Context Packet",
            "",
            "## Task",
            query,
            "",
            "## Instructions for the Coding Agent",
            "Use this as the primary repo context.",
            "Prefer modifying selected files unless investigation proves another file is required.",
            "Pay attention to warnings and related tests.",
            f"Target agent: {target}",
            "",
            "## Primary Context",
        ]
        primary_symbols = self._packet_primary_symbols(query=query, context=context)
        supporting_symbols = [symbol for symbol in context.selected_symbols if symbol.name not in {item.name for item in primary_symbols}]
        lines.extend(
            [f"- {symbol.name} ({symbol.file_path}:{symbol.start_line})" for symbol in primary_symbols[:6]]
            or ["- none"]
        )
        lines.extend(["", "## Supporting Context"])
        lines.extend(
            [f"- {symbol.name} ({symbol.file_path}:{symbol.start_line})" for symbol in supporting_symbols[:8]]
            or ["- none"]
        )
        lines.extend(["", "## Verification Context"])
        lines.extend([f"- {file_path}" for file_path in related_tests] or ["- none found"])
        lines.extend(["", "## Selected Files"])
        lines.extend(f"- {file_path}" for file_path in selected_files)
        lines.extend(["", "## Functional Call Chain"])
        call_chain_warning = self._packet_call_chain_warning(query=query, context=context)
        if context.call_chain and context.call_chain.summary:
            lines.append(context.call_chain.summary)
        else:
            lines.append("- none")
        if call_chain_warning:
            lines.extend(["", "## Call Chain Warning", f"- {call_chain_warning}"])
        if context.patch_context and context.patch_context.summary:
            lines.extend(["", "## Dependency And Patch Hints", context.patch_context.summary])
        lines.extend(
            [
                "",
                "## Key Symbols",
            ]
        )
        lines.extend(f"- {symbol.name} ({symbol.file_path}:{symbol.start_line})" for symbol in context.selected_symbols[:12])
        lines.extend(["", "## Related Tests"])
        lines.extend([f"- {file_path}" for file_path in related_tests] or ["- none found"])
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
        lines.extend(["", "## Context", context.compressed_context.strip()])
        return "\n".join(lines).strip() + "\n"

    def _packet_primary_symbols(self, *, query: str, context: ContextPack) -> list[Any]:
        if not context.selected_symbols:
            return []
        primary: list[Any] = []
        if context.call_chain and context.call_chain.entry_symbol:
            for symbol in context.selected_symbols:
                if symbol.name == context.call_chain.entry_symbol:
                    primary.append(symbol)
                    break
        terms = self._packet_task_terms(query)
        for symbol in context.selected_symbols:
            if symbol in primary:
                continue
            haystack = self._packet_normalize(f"{symbol.name} {symbol.file_path}")
            if any(term in haystack for term in terms):
                primary.append(symbol)
            if len(primary) >= 4:
                break
        if not primary:
            primary.append(context.selected_symbols[0])
        return primary

    def _packet_call_chain_warning(self, *, query: str, context: ContextPack) -> str:
        if not (context.call_chain and context.call_chain.entry_symbol):
            return ""
        terms = self._packet_task_terms(query)
        if not terms:
            return ""
        entry = self._packet_normalize(context.call_chain.entry_symbol)
        if any(term in entry for term in terms):
            return ""
        strongest = sorted(terms, key=len, reverse=True)[0]
        return (
            f"Call chain starts at {context.call_chain.entry_symbol}, which does not directly match "
            f"task keyword `{strongest}`. Verify the flow before relying on it."
        )

    def _packet_task_terms(self, query: str) -> set[str]:
        stopwords = {
            "change",
            "changes",
            "done",
            "fine",
            "review",
            "current",
            "recent",
            "code",
            "flow",
            "call",
            "chain",
            "packet",
        }
        terms = {
            self._packet_normalize(token)
            for token in query.replace("_", " ").replace(".", " ").split()
            if len(token) > 2 and token.lower() not in stopwords
        }
        return {term for term in terms if len(term) > 2}

    def _packet_normalize(self, text: str) -> str:
        return "".join(character.lower() for character in text if character.isalnum())

    def _compact_prepare_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": run.get("run_id"),
            "task": run.get("task"),
            "target": run.get("target"),
            "selected_files": run.get("selected_files", []),
            "selected_symbols": run.get("selected_symbols", [])[:12],
            "related_tests": run.get("related_tests", []),
            "warnings": run.get("missing_context_warnings", []),
            "readiness": run.get("readiness"),
            "saved": run.get("output_files", {}),
        }

    def _compact_review(self, review: dict[str, Any]) -> dict[str, Any]:
        return {
            "review_id": review.get("review_id"),
            "query": review.get("query"),
            "changed_files": review.get("changed_files", []),
            "changed_symbols": review.get("changed_symbols", [])[:12],
            "related_files": review.get("related_files", [])[:8],
            "related_tests": review.get("related_tests", []),
            "warnings": review.get("warnings", []),
            "saved": review.get("output_files", {}),
        }

    def _compact_handoff(self, handoff: dict[str, Any]) -> dict[str, Any]:
        return {
            "handoff_id": handoff.get("handoff_id"),
            "task": handoff.get("task"),
            "changed_files": handoff.get("changed_files", []),
            "latest_prepare": handoff.get("latest_prepare"),
            "latest_review": handoff.get("latest_review"),
            "tests": handoff.get("tests"),
            "notes": handoff.get("notes", []),
            "warnings": handoff.get("warnings", []),
            "saved": handoff.get("output_files", {}),
        }

    def _status_next_steps(
        self,
        *,
        indexed: bool,
        changed_files: list[str],
        latest_prepare: dict[str, Any] | None,
    ) -> list[str]:
        if not indexed:
            return ["Run `chronicle prepare \"<task>\" --repo <repo>` to index and prepare context."]
        if changed_files:
            return ["Run tests, then `chronicle review --repo <repo>` to prepare a review packet."]
        if latest_prepare:
            return ["Run `chronicle replay --latest --repo <repo>` to reuse the latest prepared packet."]
        return ["Run `chronicle prepare \"<task>\" --repo <repo>` before asking a coding agent to edit."]

    def _symbols_for_files(self, snapshot: IndexSnapshot, files: list[str]) -> list[Any]:
        file_set = set(files)
        return [symbol for symbol in snapshot.symbols if symbol.file_path in file_set]

    def _related_files_for_symbols(
        self,
        snapshot: IndexSnapshot,
        symbols: list[Any],
        changed_files: list[str],
    ) -> list[str]:
        changed_set = set(changed_files)
        symbol_index = {symbol.id: symbol for symbol in snapshot.symbols}
        related: list[str] = []
        for symbol in symbols:
            for callee_id in snapshot.call_graph.get(symbol.id, []):
                callee = symbol_index.get(callee_id)
                if callee and callee.file_path not in changed_set and callee.file_path not in related:
                    related.append(callee.file_path)
            for caller_id, callees in snapshot.call_graph.items():
                if symbol.id not in callees:
                    continue
                caller = symbol_index.get(caller_id)
                if caller and caller.file_path not in changed_set and caller.file_path not in related:
                    related.append(caller.file_path)
        for file_path in changed_files:
            for imported in snapshot.dependency_graph.get(file_path, []):
                maybe_file = imported.replace(".", "/") + ".py"
                if maybe_file not in changed_set and maybe_file not in related:
                    related.append(maybe_file)
        return related[:8]

    def _review_warnings(
        self,
        *,
        changed_files: list[str],
        related_tests: list[str],
        latest_prepare: dict[str, Any] | None,
    ) -> list[str]:
        warnings: list[str] = []
        if not changed_files:
            warnings.append("No changed Python files detected.")
        if changed_files and not related_tests:
            warnings.append("No related tests found for the changed files.")
        if changed_files and not any("test" in file_path.lower() for file_path in changed_files):
            warnings.append("No test files changed.")
        if latest_prepare:
            prepared_files = set(latest_prepare.get("selected_files", []))
            outside = [file_path for file_path in changed_files if file_path not in prepared_files]
            if outside:
                warnings.append(
                    "Changed files outside the latest prepared packet: " + ", ".join(outside[:6])
                )
        else:
            warnings.append("No prior prepare run found; review cannot compare against original packet scope.")
        return warnings

    def _handoff_warnings(
        self,
        *,
        task: str,
        status: dict[str, Any],
        prepare_payload: dict[str, Any] | None,
        review_payload: dict[str, Any] | None,
        tests: str | None,
    ) -> list[str]:
        warnings: list[str] = []
        changed_files = set(status.get("changed_files", []))
        if not tests:
            warnings.append("Tests were not recorded for this handoff.")
        if prepare_payload is None:
            warnings.append("No prepare packet is available for this handoff.")
        else:
            prepared_task = str(prepare_payload.get("task") or "")
            if prepared_task and task and prepared_task.lower() != task.lower():
                warnings.append(f"Handoff task differs from latest prepare task: {prepared_task}")
            prepared_files = set(prepare_payload.get("selected_files", []))
            missing_prepare_files = sorted(changed_files.difference(prepared_files))
            if missing_prepare_files:
                warnings.append(
                    "Changed files outside latest prepare packet: " + ", ".join(missing_prepare_files[:8])
                )
        if review_payload is None:
            warnings.append("No review packet is available for this handoff.")
        else:
            reviewed_files = set(review_payload.get("changed_files", []))
            missing_review_files = sorted(changed_files.difference(reviewed_files))
            if missing_review_files:
                warnings.append(
                    "Latest review does not cover changed files: " + ", ".join(missing_review_files[:8])
                )
            if not review_payload.get("related_tests"):
                warnings.append("Latest review has no related tests.")
        return warnings

    def _render_review_packet(
        self,
        *,
        query: str,
        changed_files: list[str],
        changed_symbols: list[str],
        related_files: list[str],
        related_tests: list[str],
        warnings: list[str],
        context: ContextPack,
    ) -> str:
        lines = [
            "# Chronicle Review Packet",
            "",
            "## Review Goal",
            query,
            "",
            "## Changed Files",
        ]
        lines.extend([f"- {file_path}" for file_path in changed_files] or ["- none detected"])
        lines.extend(["", "## Changed Symbols"])
        lines.extend([f"- {symbol}" for symbol in changed_symbols[:12]] or ["- none detected"])
        lines.extend(["", "## Related Files To Inspect"])
        lines.extend([f"- {file_path}" for file_path in related_files] or ["- none found"])
        lines.extend(["", "## Related Tests"])
        lines.extend([f"- {file_path}" for file_path in related_tests] or ["- none found"])
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
        lines.extend(["", "## Grounded Context", context.compressed_context.strip()])
        return "\n".join(lines).strip() + "\n"

    def _render_handoff_packet(
        self,
        *,
        task: str,
        status: dict[str, Any],
        prepare_payload: dict[str, Any] | None,
        review_payload: dict[str, Any] | None,
        tests: str | None,
        notes: list[str],
        warnings: list[str],
    ) -> str:
        lines = [
            "# Chronicle Handoff",
            "",
            "## Task",
            task,
            "",
            "## Repo Status",
            f"- Index status: {status.get('index_status')}",
            f"- Changed files: {', '.join(status.get('changed_files', [])) or 'none'}",
            "",
            "## Prepared Context",
        ]
        if prepare_payload:
            lines.extend(
                [
                    f"- Run: {prepare_payload.get('run_id')}",
                    f"- Files: {', '.join(prepare_payload.get('selected_files', [])[:8]) or 'none'}",
                    f"- Warnings: {', '.join(prepare_payload.get('missing_context_warnings', [])) or 'none'}",
                ]
            )
        else:
            lines.append("- none")
        lines.extend(["", "## Review Context"])
        if review_payload:
            lines.extend(
                [
                    f"- Review: {review_payload.get('review_id')}",
                    f"- Changed files: {', '.join(review_payload.get('changed_files', [])[:8]) or 'none'}",
                    f"- Related tests: {', '.join(review_payload.get('related_tests', [])) or 'none'}",
                    f"- Warnings: {', '.join(review_payload.get('warnings', [])) or 'none'}",
                ]
            )
        else:
            lines.append("- none")
        lines.extend(["", "## Handoff Warnings"])
        lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
        lines.extend(["", "## Tests"])
        lines.append(tests or "Not recorded.")
        lines.extend(["", "## Notes"])
        lines.extend([f"- {note}" for note in notes] or ["- none"])
        return "\n".join(lines).strip() + "\n"

    def _latest_artifact(self, folder: str) -> dict[str, Any] | None:
        latest_path = self.config.index_dir / folder / "latest.json"
        if not latest_path.exists():
            return None
        try:
            pointer = json.loads(latest_path.read_text(encoding="utf-8"))
            payload = self._load_artifact_payload(pointer)
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        if payload is None:
            return None
        return {
            **pointer,
            "task": payload.get("task") or payload.get("query"),
            "generated_at": payload.get("generated_at"),
            "selected_files": payload.get("selected_files", []),
            "changed_files": payload.get("changed_files", []),
        }

    def _write_latest_artifact(self, folder: str, artifact_id: str, json_path: Path) -> None:
        latest_path = self.config.index_dir / folder / "latest.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(
            json.dumps({"id": artifact_id, "json": str(json_path)}, indent=2),
            encoding="utf-8",
        )

    def _load_artifact_payload(self, pointer: dict[str, Any] | None) -> dict[str, Any] | None:
        if not pointer:
            return None
        path_value = pointer.get("json") or pointer.get("run_json") or pointer.get("review_json") or pointer.get("handoff_json")
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _load_snapshot(self) -> IndexSnapshot | None:
        return self.snapshot_store.load(index_dir=self.config.index_dir)

    def _ensure_snapshot(self) -> IndexSnapshot:
        snapshot = self._load_snapshot()
        if snapshot is None or not snapshot.symbols or self._has_uncommitted_python_changes():
            snapshot = self.index()
        if not snapshot.symbols:
            raise ValueError(
                "Chronicle did not index any Python symbols for this repository. "
                "Check that the repo contains `.py` files and that the query targets Python code."
            )
        return snapshot

    def _sample_symbols(self, snapshot: IndexSnapshot, limit: int = 8) -> list[str]:
        preferred = [symbol.name for symbol in snapshot.symbols if symbol.file_path.startswith("src/")]
        fallback = [symbol.name for symbol in snapshot.symbols]
        names = preferred or fallback
        return names[:limit]

    def _warnings(self, snapshot: IndexSnapshot, file_count: int) -> list[str]:
        warnings: list[str] = []
        if file_count == 0:
            warnings.append("No Python files were found under the repository root.")
        if file_count > 0 and not snapshot.symbols:
            warnings.append("Python files were found, but no symbols were extracted.")
        if snapshot.symbols and not snapshot.commit_changes:
            warnings.append("No git change history was indexed; the repo may have no commits or no Python-file history.")
        return warnings

    def _index_health(self, snapshot: IndexSnapshot, file_count: int) -> dict[str, Any]:
        ready = file_count > 0 and bool(snapshot.symbols)
        if not ready:
            status = "not_ready"
        elif snapshot.symbols and snapshot.call_graph and snapshot.dependency_graph:
            status = "healthy"
        else:
            status = "partial"
        return {
            "status": status,
            "ready_for_eval": ready,
            "message": (
                "Indexing looks healthy and ready for evaluation."
                if status == "healthy"
                else "Indexing is partial but usable."
                if status == "partial"
                else "Indexing is incomplete; Chronicle cannot evaluate this repository yet."
            ),
        }

    def _top_matches(self, context: ContextPack) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for symbol in context.selected_symbols[:5]:
            matches.append(
                {
                    "name": symbol.name,
                    "location": f"{symbol.file_path}:{symbol.start_line}",
                    "score": context.ranking_scores.get(symbol.id, 0.0),
                }
            )
        return matches

    def _query_diagnosis(self, plan: QueryPlan, context: ContextPack) -> dict[str, Any]:
        exact_candidates = {self._normalize_identifier(candidate) for candidate in plan.candidate_symbols}
        matched_symbol_names = {self._normalize_identifier(symbol.name.split(".")[-1]) for symbol in context.selected_symbols}
        exact_match = bool(exact_candidates & matched_symbol_names) if exact_candidates else False
        top_name = context.selected_symbols[0].name if context.selected_symbols else None
        if exact_match:
            status = "strong_match"
            message = "Chronicle found an exact or near-exact symbol match for your query."
        elif context.selected_symbols:
            status = "nearby_match"
            message = "Chronicle found related code, but not an exact symbol match. Use this for exploration, not final judgment."
        else:
            status = "no_match"
            message = "Chronicle did not find relevant code for this query."
        return {
            "status": status,
            "message": message,
            "good_for_token_eval": status in {"strong_match", "nearby_match"},
            "top_match": top_name,
        }

    def _human_summary(self, health: dict[str, Any], query_diagnosis: dict[str, Any] | None) -> str:
        if query_diagnosis is None:
            return health["message"]
        if health["status"] != "healthy":
            return f"{health['message']} {query_diagnosis['message']}"
        return f"{health['message']} {query_diagnosis['message']}"

    def _score_files_for_baseline(self, plan: Any, snapshot: IndexSnapshot) -> dict[str, float]:
        scores: dict[str, float] = {}
        normalized_candidates = {self._normalize_identifier(candidate) for candidate in plan.candidate_symbols}
        normalized_keywords = {self._normalize_identifier(keyword) for keyword in plan.keywords}
        for symbol in snapshot.symbols:
            file_path = symbol.file_path
            score = scores.get(file_path, 0.0)
            normalized_leaf = self._normalize_identifier(symbol.name.split(".")[-1])
            if normalized_leaf and normalized_leaf in normalized_candidates:
                score += 6.0
            if any(keyword in symbol.body.lower() for keyword in plan.keywords):
                score += 0.3
            if normalized_leaf and normalized_leaf in normalized_keywords:
                score += 2.0
            if file_path in plan.candidate_files:
                score += 4.0
            if any(keyword in file_path.lower() for keyword in plan.keywords):
                score += 0.5
            scores[file_path] = score
        return scores

    def _read_repo_file(self, file_path: str) -> str:
        target = self.config.repo_path / file_path
        if not target.exists():
            return ""
        return target.read_text(encoding="utf-8")

    def _arm_report(
        self,
        label: str,
        context: ContextPack,
        answer: str,
        validation: ValidationResult,
        guardrails: dict[str, Any],
        *,
        repaired: bool = False,
        repair_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "label": label,
            "estimated_input_tokens": context.estimated_tokens,
            "estimated_output_tokens": self.budget_manager.estimate_tokens(answer),
            "selected_symbol_count": len(context.selected_symbols),
            "selected_symbols": [
                {
                    "name": symbol.name,
                    "file_path": symbol.file_path,
                    "start_line": symbol.start_line,
                }
                for symbol in context.selected_symbols[:8]
            ],
            "guardrails": guardrails,
            "validation": validation.model_dump(),
            "repaired": repaired,
            "repair_notes": repair_notes or [],
            "answer": answer,
        }

    def _answer_with_optional_repair(
        self,
        *,
        provider: OllamaProvider,
        model: str,
        query: str,
        context: ContextPack,
        label: str,
    ) -> dict[str, Any]:
        guardrails = self.guardrails.inspect(context.compressed_context)
        answer = provider.generate_text(
            model,
            build_answer_prompt(
                query=self._llm_query_envelope(
                    query=query,
                    context=context,
                    response_policy=self._response_policy(query=query, context=context),
                ),
                context=guardrails.redacted_text,
            ),
        )
        if answer is None:
            raise RuntimeError(
                "Chronicle did not receive an answer from Ollama. "
                "Verify the model name, local Ollama server, and timeout settings."
            )
        validation = self.validate_output(answer, context)
        if self._should_attempt_repair(context=context, validation=validation):
            repaired_answer = provider.generate_text(
                model,
                build_repair_prompt(
                    query=self._llm_query_envelope(
                        query=query,
                        context=context,
                        response_policy=self._response_policy(query=query, context=context),
                    ),
                    context=guardrails.redacted_text,
                    draft_answer=answer,
                    issues=validation.issues,
                ),
            )
            if repaired_answer:
                repaired_validation = self.validate_output(repaired_answer, context)
                if self._repair_improved(repaired_validation=repaired_validation, original_validation=validation):
                    return self._arm_report(
                        label=label,
                        context=context,
                        answer=repaired_answer,
                        validation=repaired_validation,
                        guardrails=guardrails.model_dump(),
                        repaired=True,
                        repair_notes=[
                            "Applied grounded repair loop after weak initial validation.",
                            f"Initial confidence={validation.confidence:.2f}, repaired confidence={repaired_validation.confidence:.2f}",
                        ],
                    )
        return self._arm_report(
            label=label,
            context=context,
            answer=answer,
            validation=validation,
            guardrails=guardrails.model_dump(),
        )

    def _should_attempt_repair(self, *, context: ContextPack, validation: ValidationResult) -> bool:
        if not context.selected_symbols:
            return False
        if validation.valid and validation.confidence >= 0.7:
            return False
        return context.confidence >= 0.5 and bool(validation.issues or validation.confidence < 0.7)

    def _repair_improved(
        self,
        *,
        repaired_validation: ValidationResult,
        original_validation: ValidationResult,
    ) -> bool:
        if repaired_validation.valid and not original_validation.valid:
            return True
        if repaired_validation.grounded and not original_validation.grounded:
            return True
        if repaired_validation.confidence >= original_validation.confidence + 0.08:
            return True
        if repaired_validation.ungrounded_references < original_validation.ungrounded_references:
            return True
        return False

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = set(self.query_planner._keywords(left.lower()))
        right_tokens = set(self.query_planner._keywords(right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 2)

    def _normalize_identifier(self, text: str) -> str:
        return "".join(character.lower() for character in text if character.isalnum())

    def _ab_winner_summary(
        self,
        baseline_validation: ValidationResult,
        chronicle_validation: ValidationResult,
        reduction_percent: float,
    ) -> str:
        if baseline_validation.grounded and not chronicle_validation.grounded:
            return (
                f"Baseline answer appears more grounded. Chronicle still reduced input tokens by {reduction_percent:.2f}%, "
                "but this run should not be treated as a quality win."
            )
        if chronicle_validation.grounded and not baseline_validation.grounded:
            return (
                f"Chronicle reduced input tokens by {reduction_percent:.2f}% and produced the more grounded answer."
            )
        if chronicle_validation.confidence > baseline_validation.confidence and reduction_percent > 0:
            return (
                f"Chronicle reduced input tokens by {reduction_percent:.2f}% and improved grounding confidence."
            )
        if chronicle_validation.confidence >= baseline_validation.confidence and reduction_percent > 0:
            return (
                f"Chronicle reduced input tokens by {reduction_percent:.2f}% while keeping grounding roughly comparable."
            )
        return (
            "Chronicle reduced context size, but this benchmark still needs manual review before claiming equal answer quality."
        )

    def _refine_context_confidence(self, *, plan: QueryPlan, context: ContextPack) -> float:
        confidence = context.confidence
        if not context.selected_symbols:
            return 0.0
        normalized_candidates = {self._normalize_identifier(candidate) for candidate in plan.candidate_symbols}
        normalized_symbols = {
            self._normalize_identifier(symbol.name) for symbol in context.selected_symbols
        } | {
            self._normalize_identifier(symbol.name.split(".")[-1]) for symbol in context.selected_symbols
        }
        if normalized_candidates and not (normalized_candidates & normalized_symbols):
            confidence -= 0.2

        keyword_hits = 0
        lowered_context = context.compressed_context.lower()
        for keyword in plan.keywords:
            if keyword in lowered_context:
                keyword_hits += 1
        keyword_coverage = keyword_hits / max(len(plan.keywords), 1) if plan.keywords else 1.0
        if keyword_coverage < 0.35:
            confidence -= 0.18
        elif keyword_coverage < 0.55:
            confidence -= 0.08

        top_score = max(context.ranking_scores.values(), default=0.0)
        if top_score < 4.0:
            confidence -= 0.15
        elif top_score < 7.0:
            confidence -= 0.08

        if plan.intent in {"explain", "architecture", "edit", "refactor", "performance", "dataflow"} and len(context.selected_symbols) < 2:
            confidence -= 0.1
        if context.excluded_symbols and context.estimated_tokens >= int(context.token_budget * 0.9):
            confidence -= 0.05
        return round(max(0.0, min(0.95, confidence)), 2)

    def _demo_human_summary(self, *, query: str, context: ContextPack, report: EvaluationReport) -> str:
        repo_insight = self._demo_repo_insight(query=query, context=context)
        if context.llm_decision and context.llm_decision.call_llm:
            return f"{repo_insight} Chronicle recommends an LLM call and estimates a {report.token_reduction_percent:.2f}% prompt reduction versus the baseline."
        return f"{repo_insight} The current match is still exploratory, so Chronicle recommends tightening retrieval before spending model tokens."

    def _demo_repo_insight(self, *, query: str, context: ContextPack) -> str:
        image_summary = self._image_repo_summary(query=query, context=context)
        if image_summary:
            return image_summary
        performance_summary = self._performance_repo_summary(query=query, context=context)
        if performance_summary:
            return performance_summary
        selected = ", ".join(symbol.name for symbol in context.selected_symbols[:4]) or "no strong symbols"
        return f"Chronicle mapped this question to {selected}."

    def _image_repo_summary(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_image_repo(query=query, context=context):
            return None
        stages = self._image_pipeline_stages(self._image_summary_symbols(context))
        detection = self._format_symbol_names(stages["detection"], limit=2)
        board_labeling = self._format_symbol_names(stages["board_labeling"], limit=2)
        classification = self._format_symbol_names(stages["classification"], limit=2)
        feature_extraction = self._format_symbol_names(stages["feature_extraction"], limit=2)
        preprocessing = self._format_symbol_names(stages["preprocessing"], limit=2)

        details: list[str] = []
        if detection:
            details.append(f"the likely detection logic lives in {detection}")
        if board_labeling:
            details.append(f"{board_labeling} appears downstream in board labeling")
        elif classification:
            details.append(f"{classification} looks closest to labeling or classification")
        if feature_extraction and feature_extraction != detection:
            details.append(f"{feature_extraction} looks like feature extraction")
        elif preprocessing and preprocessing != detection:
            details.append(f"{preprocessing} handles early image preparation")
        if not details:
            return None
        return f"This looks like a computer-vision pipeline, and {'; '.join(details)}."

    def _llm_payload_preview(self, *, query: str, context: ContextPack) -> dict[str, Any]:
        selected = [
            {
                "name": symbol.name,
                "file_path": symbol.file_path,
                "start_line": symbol.start_line,
            }
            for symbol in context.selected_symbols[:6]
        ]
        focus_summary = self._payload_focus_summary(context=context)
        response_policy = self._response_policy(query=query, context=context)
        context_preview = context.compressed_context[:460]
        if len(context.compressed_context) > 460:
            context_preview += "\n\n...[truncated]"
        if context.llm_decision and context.llm_decision.call_llm:
            prompt_preview = build_answer_prompt(
                query=self._llm_query_envelope(query=query, context=context, response_policy=response_policy),
                context=context_preview,
            )
            if len(prompt_preview) > 900:
                prompt_preview = prompt_preview[:900] + "\n\n...[truncated]"
        else:
            prompt_preview = None
        return {
            "query": query,
            "focus_summary": focus_summary,
            "response_policy": response_policy,
            "selected_symbols": selected,
            "context_preview": context_preview,
            "prompt_preview": prompt_preview,
        }

    def _response_policy(self, *, query: str, context: ContextPack) -> dict[str, Any]:
        lowered = query.lower()
        decision = context.llm_decision
        if "where is" in lowered or "which file" in lowered or "defined" in lowered:
            output_format = "short locator"
            verbosity = "minimal"
            section_budget = 2
            target_output_tokens = 180
        elif any(keyword in lowered for keyword in ("how", "improve", "better", "refactor", "upgrade")):
            if decision and decision.expected_value in {"cross-cutting explanation", "low-cost synthesis"}:
                output_format = "grounded explanation"
                section_budget = 5
                target_output_tokens = 360
            else:
                output_format = "compact plan"
                section_budget = 4
                target_output_tokens = 260
            verbosity = "concise"
        else:
            output_format = "grounded summary"
            verbosity = "concise"
            section_budget = 3
            target_output_tokens = 240

        max_output_tokens = target_output_tokens
        if decision and decision.max_output_tokens:
            max_output_tokens = min(decision.max_output_tokens, target_output_tokens)
        if not (decision and decision.call_llm):
            max_output_tokens = min(max_output_tokens, 180)

        return {
            "output_format": output_format,
            "verbosity": verbosity,
            "max_output_tokens": max_output_tokens,
            "section_budget": section_budget,
            "citation_scope": "selected symbols only",
            "render_shape": "short blocks",
        }

    def _llm_query_envelope(self, *, query: str, context: ContextPack, response_policy: dict[str, Any]) -> str:
        return "\n".join([query.strip(), "", "Response policy:", self._response_policy_text(response_policy)])

    def _response_policy_text(self, policy: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"- Output format: {policy['output_format']}",
                f"- Verbosity: {policy['verbosity']}",
                f"- Max output tokens: {policy['max_output_tokens']}",
                f"- Section budget: {policy['section_budget']}",
                f"- Cite using: {policy['citation_scope']}",
                f"- Render shape: {policy['render_shape']}",
            ]
        )

    def _demo_llm_reason(self, *, context: ContextPack) -> str:
        if context.llm_decision is None:
            return "Chronicle did not produce an LLM routing decision."
        if context.llm_decision.call_llm:
            return "Chronicle has enough grounded context to justify a model call."
        if context.confidence < 0.5:
            return "The current match is too broad for a confident LLM call."
        if len(context.selected_symbols) <= 2:
            return "Chronicle found only a narrow slice of the code, so an LLM would likely add little value."
        return "Chronicle thinks deterministic repo signals are enough for now."

    def _demo_query_strategy(self, *, query: str, context: ContextPack) -> str:
        if context.llm_decision and context.llm_decision.call_llm:
            return "Send the user's original question together with Chronicle's grounded context."
        image_strategy = self._image_query_strategy(query=query, context=context)
        if image_strategy:
            return image_strategy
        performance_strategy = self._performance_query_strategy(query=query, context=context)
        if performance_strategy:
            return performance_strategy
        return "Refine the question toward a specific function, stage, or failure point before using an LLM."

    def _demo_context_strategy(self, *, context: ContextPack) -> str:
        if context.llm_decision and context.llm_decision.call_llm:
            return "Send only Chronicle's compressed context pack, not the full repository."
        return "Use the selected symbols as exploration hints and avoid sending the current context to a model yet."

    def _demo_next_step(self, *, query: str, context: ContextPack) -> str:
        if context.llm_decision and context.llm_decision.call_llm:
            return "Proceed with an LLM using the context preview below."
        image_follow_up = self._image_follow_up(query=query, context=context)
        if image_follow_up:
            return image_follow_up
        performance_follow_up = self._performance_follow_up(query=query, context=context)
        if performance_follow_up:
            return performance_follow_up
        top = context.selected_symbols[0].name if context.selected_symbols else "the top symbol"
        return f"Ask a narrower follow-up around {top} or run call-chain to clarify the exact code path first."

    def _payload_focus_summary(self, *, context: ContextPack) -> str:
        names = [symbol.name for symbol in context.selected_symbols[:4]]
        if not names:
            return "Chronicle does not have a stable context pack yet."
        if context.call_chain and context.call_chain.entry_symbol:
            return f"Anchor the model on {names[0]} and preserve the call path from {context.call_chain.entry_symbol}."
        return f"Anchor the model on {self._format_symbol_names(names, limit=min(3, len(names)))} before generalizing."

    def _image_query_strategy(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_image_query(query):
            return None
        stages = self._image_pipeline_stages(self._image_summary_symbols(context))
        detection = self._format_symbol_names(stages["detection"], limit=2)
        board_labeling = self._format_symbol_names(stages["board_labeling"], limit=2)
        classification = self._format_symbol_names(stages["classification"], limit=2)
        if detection and board_labeling:
            return f"Refine the question toward a concrete stage, like the detection stage in {detection} or downstream labeling in {board_labeling}."
        if detection:
            return f"Refine the question toward the detection stage around {detection} before using an LLM."
        if classification:
            return f"Refine the question toward classification or labeling around {classification} before using an LLM."
        return None

    def _image_follow_up(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_image_query(query):
            return None
        stages = self._image_pipeline_stages(self._image_summary_symbols(context))
        detection = self._format_symbol_names(stages["detection"], limit=1)
        board_labeling = self._format_symbol_names(stages["board_labeling"], limit=1)
        classification = self._format_symbol_names(stages["classification"], limit=1)
        if detection and board_labeling:
            return f"Run call-chain on {detection} first, then ask whether {board_labeling} should stay downstream or be replaced."
        if detection:
            return f"Ask a narrower follow-up around {detection} or run call-chain to isolate the exact detection path."
        if classification:
            return f"Ask a narrower follow-up around {classification} to validate whether it owns labeling or piece classification."
        return None

    def _looks_like_image_repo(self, *, query: str, context: ContextPack) -> bool:
        symbols = self._image_summary_symbols(context)
        text = " ".join(
            [query]
            + [symbol.name for symbol in symbols]
            + [symbol.file_path for symbol in symbols]
            + [symbol.body[:200] for symbol in symbols[:4]]
        ).lower()
        image_terms = (
            "cv2",
            "opencv",
            "image",
            "frame",
            "pixel",
            "contour",
            "threshold",
            "histogram",
            "centroid",
            "board",
            "square",
            "piece",
            "camera",
        )
        return any(term in text for term in image_terms)

    def _looks_like_image_query(self, query: str) -> bool:
        lowered = query.lower()
        image_terms = (
            "image",
            "vision",
            "detect",
            "classification",
            "classify",
            "segmentation",
            "square",
            "piece",
            "board",
            "frame",
            "opencv",
            "cv2",
        )
        return any(term in lowered for term in image_terms)

    def _performance_repo_summary(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_performance_query(query):
            return None
        stages = self._pipeline_stages(context)
        prioritized = [
            ("orchestration", "the main orchestration path appears to live in"),
            ("io", "I/O-heavy work appears to cluster around"),
            ("batching", "batching or chunked work appears in"),
            ("compute", "the main compute path appears to live in"),
        ]
        details: list[str] = []
        for stage, prefix in prioritized:
            names = self._format_symbol_names(stages[stage], limit=2)
            if names:
                details.append(f"{prefix} {names}")
        if not details:
            return None
        return f"This looks like a performance-oriented pipeline, and {'; '.join(details)}."

    def _performance_query_strategy(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_performance_query(query):
            return None
        stages = self._pipeline_stages(context)
        orchestration = self._format_symbol_names(stages["orchestration"], limit=1)
        io_bound = self._format_symbol_names(stages["io"], limit=2)
        batching = self._format_symbol_names(stages["batching"], limit=2)
        if orchestration and io_bound:
            return f"Refine the question toward a concrete latency stage, like orchestration in {orchestration} or I/O work in {io_bound}."
        if batching:
            return f"Refine the question toward chunking, batching, or queue boundaries around {batching} before using an LLM."
        if orchestration:
            return f"Refine the question toward the main execution path around {orchestration} before using an LLM."
        return "Refine the question toward scheduling, batching, I/O, or compute hotspots before using an LLM."

    def _performance_follow_up(self, *, query: str, context: ContextPack) -> str | None:
        if not self._looks_like_performance_query(query):
            return None
        stages = self._pipeline_stages(context)
        orchestration = self._format_symbol_names(stages["orchestration"], limit=1)
        io_bound = self._format_symbol_names(stages["io"], limit=1)
        batching = self._format_symbol_names(stages["batching"], limit=1)
        if orchestration and io_bound:
            return f"Run call-chain on {orchestration} first, then ask whether {io_bound} is the main latency boundary."
        if batching:
            return f"Ask a narrower follow-up around {batching} to validate chunking, batching, or queue overhead."
        if orchestration:
            return f"Ask a narrower follow-up around {orchestration} or run call-chain to isolate the main latency path."
        return None

    def _looks_like_performance_query(self, query: str) -> bool:
        lowered = query.lower()
        performance_terms = (
            "latency",
            "slow",
            "faster",
            "speed",
            "performance",
            "throughput",
            "optimiz",
            "bottleneck",
            "async",
            "asynchronous",
            "concurrency",
            "parallel",
            "pipeline",
            "queue",
            "batch",
        )
        return any(term in lowered for term in performance_terms)

    def _pipeline_stages(self, context: ContextPack) -> dict[str, list[str]]:
        stage_keywords = {
            "orchestration": ("pipeline", "orchestr", "async", "await", "schedule", "dispatch", "worker"),
            "io": ("fetch", "load", "read", "write", "download", "upload", "request", "stream", "io"),
            "batching": ("batch", "chunk", "queue", "buffer", "window"),
            "compute": ("train", "process", "compute", "transform", "encode", "decode"),
        }
        stages = {stage: [] for stage in stage_keywords}
        symbols = context.selected_symbols[:6] if context.selected_symbols else self._ensure_snapshot().symbols[:20]
        for symbol in symbols:
            text = f"{symbol.name} {symbol.file_path} {symbol.body[:240]}".lower()
            for stage, keywords in stage_keywords.items():
                if any(keyword in text for keyword in keywords):
                    stages[stage].append(symbol.name)
        return {stage: self._unique_preserve_order(names) for stage, names in stages.items()}

    def _image_summary_symbols(self, context: ContextPack) -> list[Any]:
        if context.selected_symbols:
            return context.selected_symbols[:6]
        snapshot = self._ensure_snapshot()
        image_terms = (
            "cv2",
            "opencv",
            "image",
            "frame",
            "contour",
            "threshold",
            "histogram",
            "centroid",
            "board",
            "square",
            "piece",
        )
        ranked = []
        for symbol in snapshot.symbols:
            text = f"{symbol.name} {symbol.file_path} {symbol.body[:240]}".lower()
            score = sum(1 for term in image_terms if term in text)
            if score > 0:
                ranked.append((score, symbol))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [symbol for _, symbol in ranked[:6]]

    def _image_pipeline_stages(self, symbols: list[Any]) -> dict[str, list[str]]:
        stage_keywords = {
            "detection": ("detect", "centroid", "contour", "crop", "threshold", "square", "bbox"),
            "segmentation": ("segment", "mask", "morph", "watershed"),
            "classification": ("classif", "predict", "histogram", "piece", "svm", "cnn"),
            "feature_extraction": ("feature", "descriptor", "histogram", "edge", "centroid"),
            "post_processing": ("normalize", "filter", "cleanup", "draw", "annotate"),
            "board_labeling": ("label", "board", "grid", "square"),
            "preprocessing": ("frame", "image", "load", "read", "crop", "normalize", "exposure"),
        }
        stages = {stage: [] for stage in stage_keywords}
        for symbol in symbols:
            text = f"{symbol.name} {symbol.file_path} {symbol.body[:240]}".lower()
            for stage, keywords in stage_keywords.items():
                if any(keyword in text for keyword in keywords):
                    stages[stage].append(symbol.name)
        return {stage: self._unique_preserve_order(names) for stage, names in stages.items()}

    def _format_symbol_names(self, names: list[str], limit: int) -> str:
        trimmed = names[:limit]
        if not trimmed:
            return ""
        if len(trimmed) == 1:
            return trimmed[0]
        return ", ".join(trimmed[:-1]) + f" and {trimmed[-1]}"

    def _unique_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _prefer_query_entry_symbols(self, symbols: list[Any], plan: QueryPlan) -> list[Any]:
        if not symbols or not plan.candidate_symbols:
            return symbols
        normalized_candidates = [
            (
                self._normalize_identifier(candidate),
                self._normalize_identifier(candidate.split(".")[-1]),
            )
            for candidate in plan.candidate_symbols
        ]

        def rank(symbol: Any) -> tuple[int, int]:
            normalized_name = self._normalize_identifier(symbol.name)
            normalized_leaf = self._normalize_identifier(symbol.name.split(".")[-1])
            for index, (candidate_full, candidate_leaf) in enumerate(normalized_candidates):
                if normalized_name == candidate_full:
                    return (0, index)
                if normalized_leaf == candidate_leaf:
                    return (1, index)
            return (2, 9999)

        ordered = list(symbols)
        ordered.sort(key=rank)
        return ordered

    def _session_hints(self, session_id: str, create_if_missing: bool) -> SessionMemoryHints:
        session = (
            self.session_store.get_or_create(self.config.index_dir, self.config.repo_path, session_id)
            if create_if_missing
            else self.session_store.load(self.config.index_dir, session_id)
        )
        if session is None:
            return SessionMemoryHints(
                session_id=session_id,
                prior_turn_count=0,
            )
        recent_queries = [turn.query.lower() for turn in session.turns[-self.config.session_query_recall_limit :]]
        validated_facts = [
            note
            for turn in session.turns
            if turn.grounded and turn.notes
            for note in turn.notes
        ][: self.config.session_query_recall_limit]
        preferred_symbols = self._most_common(
            [symbol for turn in session.turns for symbol in turn.selected_symbols],
            limit=self.config.session_symbol_recall_limit,
        )
        preferred_files = self._most_common(
            [file_path for turn in session.turns for file_path in turn.selected_files],
            limit=self.config.session_file_recall_limit,
        )
        return SessionMemoryHints(
            session_id=session.session_id,
            prior_turn_count=len(session.turns),
            preferred_symbols=preferred_symbols,
            preferred_files=preferred_files,
            recent_queries=recent_queries,
            validated_facts=validated_facts,
        )

    def _remember_context(self, session_id: str, plan: QueryPlan, context: ContextPack) -> None:
        self.session_store.get_or_create(self.config.index_dir, self.config.repo_path, session_id)
        turn = SessionTurn(
            turn_id=self._new_turn_id(),
            query=context.query,
            intent=plan.intent,
            token_budget=context.token_budget,
            estimated_tokens=context.estimated_tokens,
            selected_symbols=[symbol.name for symbol in context.selected_symbols],
            selected_files=list(dict.fromkeys(symbol.file_path for symbol in context.selected_symbols)),
            excluded_symbols=list(context.excluded_symbols),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.session_store.append_turn(self.config.index_dir, session_id, turn)

    def _remember_validation(self, session_id: str, validation: ValidationResult, notes: list[str] | None = None) -> None:
        self.session_store.update_latest_turn(
            self.config.index_dir,
            session_id,
            validation_confidence=validation.confidence,
            grounded=validation.grounded,
            notes=notes,
        )

    def _session_payload(self, session_id: str) -> dict[str, Any]:
        session = self.session_store.load(self.config.index_dir, session_id)
        if session is None:
            return {
                "session_id": session_id,
                "status": "not_found",
                "message": "Chronicle has not recorded any turns for this session yet.",
            }
        hints = self._session_hints(session_id=session_id, create_if_missing=False)
        return {
            "session_id": session.session_id,
            "status": "active",
            "turn_count": len(session.turns),
            "updated_at": session.updated_at,
            "recalled_symbols": hints.preferred_symbols,
            "recalled_files": hints.preferred_files,
            "recent_queries": hints.recent_queries,
            "validated_facts": hints.validated_facts,
        }

    def _most_common(self, items: list[str], limit: int) -> list[str]:
        counts: dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item for item, _ in ranked[:limit]]

    def _new_session_id(self) -> str:
        return f"session-{uuid4().hex[:12]}"

    def _new_turn_id(self) -> str:
        return f"turn-{uuid4().hex[:12]}"

    def _new_bus_id(self) -> str:
        return f"bus-{uuid4().hex[:12]}"

    def _has_uncommitted_python_changes(self) -> bool:
        return bool(self._changed_python_files(include_untracked=False))

    def _changed_python_files(self, *, include_untracked: bool) -> list[str]:
        if not (self.config.repo_path / ".git").exists():
            return []
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", "*.py"],
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        files = [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".py")]
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--", "*.py"],
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if staged.returncode == 0:
            files.extend(line.strip() for line in staged.stdout.splitlines() if line.strip().endswith(".py"))
        if include_untracked:
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if untracked.returncode == 0:
                files.extend(line.strip() for line in untracked.stdout.splitlines() if line.strip().endswith(".py"))
        return list(dict.fromkeys(files))
