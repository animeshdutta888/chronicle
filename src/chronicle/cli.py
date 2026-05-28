from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys

from .api import Chronicle
from .remote_repo import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chronicle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("index", "Build and persist repository intelligence."),
        ("build", "Alias for index."),
    ):
        index_parser = subparsers.add_parser(command, help=help_text)
        index_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
        index_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before indexing.")
        index_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
        index_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
        index_parser.add_argument("--index-dir", default=None, help="Directory to persist Chronicle artifacts.")

    for command, help_text in (
        ("context", "Retrieve grounded context for a coding-agent query."),
        ("ask", "Alias for context."),
    ):
        context_parser = subparsers.add_parser(command, help=help_text)
        context_parser.add_argument("query", help="Question to plan and retrieve context for.")
        context_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
        context_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
        context_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
        context_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
        context_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
        context_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
        context_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
        context_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare a grounded context packet for a coding agent.")
    prepare_parser.add_argument("query", help="Coding task to prepare context for.")
    prepare_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    prepare_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    prepare_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    prepare_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    prepare_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    prepare_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    prepare_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    prepare_parser.add_argument("--target", choices=["codex", "claude", "cursor", "generic"], default="generic", help="Agent packet target.")
    prepare_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")
    prepare_parser.add_argument("--no-auto-index", action="store_true", help="Fail instead of auto-indexing when no index exists.")
    prepare_parser.add_argument("--force-reindex", action="store_true", help="Rebuild the index before preparing context.")

    run_parser = subparsers.add_parser("run", help="Prepare a full Chronicle manual agent run.")
    run_parser.add_argument("task", help="Coding task to prepare for an agent.")
    run_parser.add_argument("--manual", action="store_true", help="Prepare files for manual paste into an agent.")
    run_parser.add_argument("--target", choices=["codex", "claude", "cursor", "generic"], default="generic", help="Agent prompt target.")
    run_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    run_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    run_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    run_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    run_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    run_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    run_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    run_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    finish_parser = subparsers.add_parser("finish", help="Finish a Chronicle run by capturing diff, review, and report artifacts.")
    finish_parser.add_argument("--run", default=None, help="Run id to finish. Defaults to the active run.")
    finish_parser.add_argument("--base", default="main", help="Base branch or ref for PR review.")
    finish_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    finish_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    finish_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    finish_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    finish_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    finish_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    report_parser = subparsers.add_parser("report", help="Print or regenerate a Chronicle run report.")
    report_parser.add_argument("--latest", action="store_true", help="Use the latest active run.")
    report_parser.add_argument("--run", default=None, help="Run id to report.")
    report_parser.add_argument("--format", choices=["markdown"], default="markdown", help="Report format.")
    report_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    report_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    report_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    report_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    report_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    report_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    replay_parser = subparsers.add_parser("replay", help="Replay saved Chronicle prepare runs without recomputing context.")
    replay_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    replay_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    replay_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    replay_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    replay_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    replay_parser.add_argument("--latest", action="store_true", help="Replay the latest prepared run.")
    replay_parser.add_argument("--list", action="store_true", help="List saved prepared runs.")
    replay_parser.add_argument("--run", default=None, help="Replay a specific run id.")
    replay_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    explain_parser = subparsers.add_parser("explain", help="Explain why a prepared run selected its context.")
    explain_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    explain_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    explain_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    explain_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    explain_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    explain_parser.add_argument("--latest", action="store_true", help="Explain the latest prepared run.")
    explain_parser.add_argument("--run", default=None, help="Explain a specific run id.")
    explain_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect an indexed file or symbol.")
    inspect_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    inspect_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    inspect_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    inspect_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    inspect_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    inspect_parser.add_argument("--file", default=None, help="File path to inspect.")
    inspect_parser.add_argument("--symbol", default=None, help="Symbol name to inspect.")
    inspect_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    status_parser = subparsers.add_parser("status", help="Show Chronicle index, change, and artifact status.")
    status_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    status_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    status_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    status_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    status_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    status_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    review_parser = subparsers.add_parser("review", help="Prepare a grounded review packet for recent code changes.")
    review_parser.add_argument("query", nargs="?", default="Review recent code changes and impacted tests", help="Review goal.")
    review_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    review_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    review_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    review_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    review_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    review_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    review_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    review_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    handoff_parser = subparsers.add_parser("handoff", help="Create a concise handoff packet from latest prepare/review state.")
    handoff_parser.add_argument("task", nargs="?", default=None, help="Optional handoff task title.")
    handoff_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    handoff_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    handoff_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    handoff_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    handoff_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    handoff_parser.add_argument("--tests", default=None, help="Optional tests run or test result summary.")
    handoff_parser.add_argument("--note", action="append", default=None, help="Optional handoff note.")
    handoff_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    setup_parser = subparsers.add_parser("setup", help="Add Chronicle workflow instructions for Codex, Claude, or Cursor.")
    setup_parser.add_argument("agent", choices=["codex", "claude", "cursor", "all"], help="Agent workflow file to configure.")
    setup_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    setup_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before setup.")
    setup_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    setup_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    setup_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    setup_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")
    setup_parser.add_argument("--no-mcp", action="store_true", help="Only update agent instruction files; do not register MCP.")
    setup_parser.add_argument("--mcp-only", action="store_true", help="Only register MCP; do not update agent instruction files.")

    pr_review_parser = subparsers.add_parser("pr-review", help="Create a local PR review artifact from git diff.")
    pr_review_parser.add_argument("--base", default="main", help="Base branch or ref to diff against.")
    pr_review_parser.add_argument("--format", choices=["markdown"], default="markdown", help="Output format.")
    pr_review_parser.add_argument("--output", default=None, help="Optional output markdown path.")
    pr_review_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    pr_review_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    pr_review_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    pr_review_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    pr_review_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    pr_review_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare Chronicle context against a baseline.")
    evaluate_parser.add_argument("query", help="Question to evaluate.")
    evaluate_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    evaluate_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    evaluate_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    evaluate_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    evaluate_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    evaluate_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    evaluate_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    evaluate_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    doctor_parser = subparsers.add_parser("doctor", help="Diagnose repository indexing and retrieval readiness.")
    doctor_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    doctor_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    doctor_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    doctor_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    doctor_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    doctor_parser.add_argument("--query", default=None, help="Optional query to test retrieval.")
    doctor_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    doctor_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id to inspect memory-aware retrieval.")
    doctor_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    demo_parser = subparsers.add_parser("demo", help="Run index + context + evaluate as an end-to-end token-savings demo.")
    demo_parser.add_argument("query", help="Question to run end-to-end.")
    demo_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    demo_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    demo_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    demo_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    demo_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    demo_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    demo_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    demo_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    ab_parser = subparsers.add_parser("ab-test", help="Run baseline vs Chronicle context through an LLM and compare outputs.")
    ab_parser.add_argument("query", help="Question to compare.")
    ab_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    ab_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    ab_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    ab_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    ab_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    ab_parser.add_argument("--token-budget", type=int, default=None, help="Chronicle token budget.")
    ab_parser.add_argument("--baseline-token-budget", type=int, default=None, help="Baseline full-file token budget.")
    ab_parser.add_argument("--model", required=True, help="Ollama model to use for both arms.")
    ab_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    ab_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    call_chain_parser = subparsers.add_parser("call-chain", help="Render a functional call chain for a repo question.")
    call_chain_parser.add_argument("query", help="Question to trace.")
    call_chain_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    call_chain_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    call_chain_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    call_chain_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    call_chain_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    call_chain_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    call_chain_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id for multi-turn memory.")
    call_chain_parser.add_argument("--max-depth", type=int, default=4, help="Maximum chain depth to render.")
    call_chain_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    session_start_parser = subparsers.add_parser("session-start", help="Create or reuse a Chronicle session for multi-turn memory.")
    session_start_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    session_start_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    session_start_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    session_start_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    session_start_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    session_start_parser.add_argument("--session-id", default=None, help="Optional custom session id.")
    session_start_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    session_show_parser = subparsers.add_parser("session-show", help="Inspect Chronicle session memory.")
    session_show_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    session_show_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    session_show_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    session_show_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    session_show_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    session_show_parser.add_argument("--session-id", required=True, help="Chronicle session id to inspect.")
    session_show_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    bus_start_parser = subparsers.add_parser("bus-start", help="Create a shared Chronicle context bus for multi-agent workflows.")
    bus_start_parser.add_argument("query", help="Root query for the multi-agent workflow.")
    bus_start_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    bus_start_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    bus_start_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    bus_start_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    bus_start_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    bus_start_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id to attach.")
    bus_start_parser.add_argument("--bus-id", default=None, help="Optional custom bus id.")
    bus_start_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    bus_context_parser = subparsers.add_parser("bus-context", help="Add a grounded phase to a Chronicle multi-agent bus.")
    bus_context_parser.add_argument("query", help="Query for this agent phase.")
    bus_context_parser.add_argument("--role", required=True, choices=["planner", "coder", "reviewer", "critic", "governance", "retriever"])
    bus_context_parser.add_argument("--bus-id", required=True, help="Chronicle bus id.")
    bus_context_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    bus_context_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    bus_context_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    bus_context_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    bus_context_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    bus_context_parser.add_argument("--token-budget", type=int, default=None, help="Override token budget.")
    bus_context_parser.add_argument("--session-id", default=None, help="Optional Chronicle session id to reuse memory.")
    bus_context_parser.add_argument("--note", action="append", default=None, help="Optional note for this phase.")
    bus_context_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    bus_handoff_parser = subparsers.add_parser("bus-handoff", help="Record a deterministic handoff between agents on a Chronicle bus.")
    bus_handoff_parser.add_argument("--bus-id", required=True, help="Chronicle bus id.")
    bus_handoff_parser.add_argument("--from-role", required=True, choices=["planner", "coder", "reviewer", "critic", "governance", "retriever"])
    bus_handoff_parser.add_argument("--to-role", required=True, choices=["planner", "coder", "reviewer", "critic", "governance", "retriever"])
    bus_handoff_parser.add_argument("--reason", required=True, help="Reason for the handoff.")
    bus_handoff_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    bus_handoff_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    bus_handoff_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    bus_handoff_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    bus_handoff_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    bus_handoff_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")

    bus_show_parser = subparsers.add_parser("bus-show", help="Inspect a Chronicle multi-agent context bus.")
    bus_show_parser.add_argument("--bus-id", required=True, help="Chronicle bus id.")
    bus_show_parser.add_argument("--repo", default=".", help="Path to the git repository or local codebase.")
    bus_show_parser.add_argument("--repo-url", default=None, help="Git URL to clone/pull before analysis.")
    bus_show_parser.add_argument("--repos-dir", default=None, help="Directory used for cloned remote repositories.")
    bus_show_parser.add_argument("--branch", default=None, help="Optional branch to checkout/pull.")
    bus_show_parser.add_argument("--index-dir", default=None, help="Directory containing Chronicle artifacts.")
    bus_show_parser.add_argument("--view", choices=["compact", "full"], default="compact", help="Compact human view or full machine detail.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_path = resolve_repo_path(
        repo=args.repo,
        repo_url=getattr(args, "repo_url", None),
        repos_dir=getattr(args, "repos_dir", None),
        branch=getattr(args, "branch", None),
    )
    chronicle = Chronicle(repo_path=repo_path, index_dir=getattr(args, "index_dir", None))

    try:
        if args.command in {"index", "build"}:
            snapshot = chronicle.index()
            payload = {
                "repo": str(repo_path),
                "index_dir": str(chronicle.config.index_dir),
                "symbol_count": len(snapshot.symbols),
                "commit_change_count": len(snapshot.commit_changes),
                "call_graph_nodes": len(snapshot.call_graph),
                "dependency_graph_nodes": len(snapshot.dependency_graph),
            }
            print(_success(command=args.command, data=payload))
            return 0

        if args.command in {"context", "ask"}:
            context = chronicle.context(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
            )
            payload = context.model_dump() if args.view == "full" else _compact_context(context.model_dump())
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "prepare":
            payload = chronicle.prepare(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
                target=getattr(args, "target", "generic"),
                view=getattr(args, "view", "compact"),
                auto_index=not getattr(args, "no_auto_index", False),
                force_reindex=getattr(args, "force_reindex", False),
            )
            if args.view == "compact":
                print(_prepare_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "run":
            payload = chronicle.run(
                task=args.task,
                target=getattr(args, "target", "generic"),
                manual=getattr(args, "manual", False),
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
                view=getattr(args, "view", "compact"),
            )
            if args.view == "compact":
                print(_run_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "finish":
            payload = chronicle.finish(
                run_id=getattr(args, "run", None),
                base=getattr(args, "base", "main"),
                view=getattr(args, "view", "compact"),
            )
            if args.view == "compact":
                print(_finish_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "report":
            payload = chronicle.report(
                run_id=getattr(args, "run", None),
                latest=getattr(args, "latest", False) or getattr(args, "run", None) is None,
                view=getattr(args, "view", "compact"),
            )
            if args.view == "compact":
                print(payload.get("report", ""))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "replay":
            payload = chronicle.replay(
                run_id=getattr(args, "run", None),
                latest=getattr(args, "latest", False),
                list_runs=getattr(args, "list", False),
                view=getattr(args, "view", "compact"),
            )
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "explain":
            payload = chronicle.explain(
                run_id=getattr(args, "run", None),
                latest=getattr(args, "latest", False),
                view=getattr(args, "view", "compact"),
            )
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "inspect":
            if bool(getattr(args, "file", None)) == bool(getattr(args, "symbol", None)):
                raise ValueError("Use exactly one of `--file` or `--symbol`.")
            if getattr(args, "file", None):
                payload = chronicle.inspect_file(args.file, view=args.view)
            else:
                payload = chronicle.inspect_symbol(args.symbol, view=args.view)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "status":
            payload = chronicle.status(view=args.view)
            if args.view == "compact":
                print(_status_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "review":
            payload = chronicle.review(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
                view=args.view,
            )
            if args.view == "compact":
                print(_review_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "handoff":
            payload = chronicle.handoff(
                task=getattr(args, "task", None),
                tests=getattr(args, "tests", None),
                notes=getattr(args, "note", None),
                view=args.view,
            )
            if args.view == "compact":
                print(_handoff_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "setup":
            if args.no_mcp and args.mcp_only:
                raise ValueError("Use only one of `--no-mcp` or `--mcp-only`.")
            payload = chronicle.setup_agent(
                args.agent,
                configure_mcp=not args.no_mcp,
                instructions=not args.mcp_only,
            )
            if args.view == "compact":
                print(_setup_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "pr-review":
            payload = chronicle.pr_review(base=args.base, output=args.output)
            if args.view == "compact":
                print(_pr_review_text(payload))
                return 0
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "evaluate":
            report = chronicle.evaluate(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
            )
            payload = {
                "repo": str(repo_path),
                "index_dir": str(chronicle.config.index_dir),
                "query": args.query,
                **report.model_dump(),
            }
            if args.view == "compact":
                payload = _compact_evaluate(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "doctor":
            payload = chronicle.diagnose(
                query=getattr(args, "query", None),
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
            )
            payload["next_steps"] = _doctor_next_steps(args)
            if args.view == "compact":
                payload = _compact_doctor(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "demo":
            payload = chronicle.demo(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
            )
            if args.view == "compact":
                payload = _compact_demo(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "ab-test":
            payload = chronicle.ab_test(
                query=args.query,
                model=args.model,
                token_budget=getattr(args, "token_budget", None),
                baseline_token_budget=getattr(args, "baseline_token_budget", None),
                session_id=getattr(args, "session_id", None),
            )
            if args.view == "compact":
                payload = _compact_ab_test(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "call-chain":
            payload = chronicle.call_chain(
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
                max_depth=getattr(args, "max_depth", 4),
            )
            if args.view == "compact":
                payload = _compact_call_chain(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "session-start":
            session = chronicle.start_session(session_id=getattr(args, "session_id", None))
            payload = session.model_dump() if args.view == "full" else _compact_session(session.model_dump())
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "session-show":
            session = chronicle.session(session_id=args.session_id)
            if session is None:
                raise ValueError(f"Chronicle could not find session `{args.session_id}` for this repository.")
            payload = {
                "session": session.model_dump(),
                "summary": chronicle.diagnose(session_id=args.session_id)["session"],
            }
            if args.view == "compact":
                payload = _compact_session_show(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "bus-start":
            bus = chronicle.start_agent_bus(
                root_query=args.query,
                bus_id=getattr(args, "bus_id", None),
                session_id=getattr(args, "session_id", None),
            )
            payload = bus.model_dump() if args.view == "full" else _compact_bus({"bus": bus.model_dump(), "summary": chronicle.bus_summary(bus.bus_id)})
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "bus-context":
            bus = chronicle.bus_context(
                bus_id=args.bus_id,
                role=args.role,
                query=args.query,
                token_budget=getattr(args, "token_budget", None),
                session_id=getattr(args, "session_id", None),
                notes=getattr(args, "note", None),
            )
            payload = {"bus": bus.model_dump(), "summary": chronicle.bus_summary(args.bus_id)}
            if args.view == "compact":
                payload = _compact_bus(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "bus-handoff":
            bus = chronicle.bus_handoff(
                bus_id=args.bus_id,
                from_role=args.from_role,
                to_role=args.to_role,
                reason=args.reason,
            )
            payload = {"bus": bus.model_dump(), "summary": chronicle.bus_summary(args.bus_id)}
            if args.view == "compact":
                payload = _compact_bus(payload)
            print(_success(command=args.command, data=payload))
            return 0

        if args.command == "bus-show":
            bus = chronicle.agent_bus(args.bus_id)
            if bus is None:
                raise ValueError(f"Chronicle could not find bus `{args.bus_id}` for this repository.")
            payload = {"bus": bus.model_dump(), "summary": chronicle.bus_summary(args.bus_id)}
            if args.view == "compact":
                payload = _compact_bus(payload)
            print(_success(command=args.command, data=payload))
            return 0
    except RuntimeError as exc:
        print(_error(command=getattr(args, "command", "chronicle"), message=str(exc)), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(_error(command=getattr(args, "command", "chronicle"), message=str(exc)), file=sys.stderr)
        return 1

    return 1


def _success(command: str, data: dict) -> str:
    return json.dumps(
        {
            "status": "ok",
            "command": command,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        },
        indent=2,
    )


def _error(command: str, message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "command": command,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error": {"message": message},
        },
        indent=2,
    )


def _prepare_text(data: dict) -> str:
    warnings = data.get("warnings") or []
    readiness = data.get("readiness") or {}
    saved = data.get("saved") or {}
    lines = [
        "Chronicle prepared agent context",
        "",
        f"Task: {data.get('task')}",
        f"Target: {data.get('target')}",
        f"Selected: {len(data.get('selected_files', []))} files, {len(data.get('selected_symbols', []))} symbols, {len(data.get('related_tests', []))} related tests",
        f"Readiness: {readiness.get('level', 'unknown')} - {readiness.get('reason', 'No readiness reason recorded.')}",
        "",
        "Selected files:",
    ]
    lines.extend(f"- {file_path}" for file_path in data.get("selected_files", []))
    lines.extend(["", "Key symbols:"])
    lines.extend(f"- {symbol}" for symbol in data.get("selected_symbols", [])[:8])
    lines.extend(["", "Warnings:"])
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "Saved:",
            f"- {saved.get('prepare_md')}",
            f"- {saved.get('prepare_json')}",
            "",
            f"Replay: chronicle replay --run {data.get('run_id')}",
            f"Explain: chronicle explain --run {data.get('run_id')}",
        ]
    )
    return "\n".join(lines)


def _run_text(data: dict) -> str:
    saved = data.get("saved") or {}
    lines = [
        "Chronicle prepared a token-optimized context packet.",
        "",
        f"Run ID: {data.get('run_id')}",
        f"Task: {data.get('task')}",
        f"Agent prompt saved: {saved.get('agent_prompt_md')}",
        "",
        "Next:",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(data.get("next", []), start=1))
    lines.extend(["", "Saved:"])
    for key in ("prepare_md", "context_packet_md", "agent_prompt_md", "run_json"):
        if saved.get(key):
            lines.append(f"- {saved[key]}")
    return "\n".join(lines)


def _finish_text(data: dict) -> str:
    saved = data.get("saved") or {}
    lines = [
        "Chronicle finished run",
        "",
        f"Run ID: {data.get('run_id')}",
        f"Status: {data.get('status')}",
        f"Risk level: {data.get('risk_level')}",
        f"Context Quality Score: {data.get('context_quality_score')}/100",
        f"Changed files: {len(data.get('changed_files', []))}",
        "",
        "Suggested tests:",
    ]
    lines.extend([f"- {test}" for test in data.get("suggested_tests", [])] or ["- none found"])
    lines.extend(["", f"Report saved: {saved.get('report_md')}", "", "Saved:"])
    for key in ("diff_patch", "review_md", "pr_review_md", "run_json"):
        if saved.get(key):
            lines.append(f"- {saved[key]}")
    lines.extend(["", f"To print it again: chronicle report --run {data.get('run_id')}"])
    return "\n".join(lines)


def _status_text(data: dict) -> str:
    lines = [
        "Chronicle status",
        "",
        f"Repo: {data.get('repo')}",
        f"Index: {data.get('index_status')} ({data.get('symbol_count')} symbols)",
        f"Changed Python files: {len(data.get('changed_files', []))}",
    ]
    if data.get("changed_files"):
        lines.extend(["", "Changed files:"])
        lines.extend(f"- {file_path}" for file_path in data.get("changed_files", []))
    latest_prepare = data.get("latest_prepare") or {}
    latest_review = data.get("latest_review") or {}
    if latest_prepare or latest_review:
        lines.extend(["", "Latest artifacts:"])
        if latest_prepare:
            lines.append(f"- prepare: {latest_prepare.get('run_id') or latest_prepare.get('id')}")
        if latest_review:
            lines.append(f"- review: {latest_review.get('id')}")
    lines.extend(["", "Next steps:"])
    lines.extend(f"- {step}" for step in data.get("next_steps", []))
    return "\n".join(lines)


def _review_text(data: dict) -> str:
    saved = data.get("saved") or {}
    warnings = data.get("warnings") or []
    lines = [
        "Chronicle reviewed recent changes",
        "",
        f"Run ID: {data.get('run_id') or data.get('review_id')}",
        f"Risk level: {data.get('risk_level', 'unknown')}",
        f"Changed: {len(data.get('changed_files', []))} files, {len(data.get('changed_symbols', []))} symbols",
        f"Related: {len(data.get('related_files', []))} files, {len(data.get('related_tests', []))} tests",
        "",
        "Changed files:",
    ]
    lines.extend([f"- {file_path}" for file_path in data.get("changed_files", [])] or ["- none"])
    lines.extend(["", "Related tests:"])
    lines.extend([f"- {file_path}" for file_path in data.get("related_tests", [])] or ["- none found"])
    lines.extend(["", "Warnings:"])
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(
        [
            "",
            "Saved:",
            f"- {saved.get('review_md')}",
            f"- {saved.get('review_json')}",
            "",
            "Handoff: chronicle handoff --tests \"<tests run>\"",
        ]
    )
    return "\n".join(lines)


def _handoff_text(data: dict) -> str:
    saved = data.get("saved") or {}
    latest_review = data.get("latest_review") or {}
    lines = [
        "Chronicle created handoff",
        "",
        f"Run ID: {data.get('run_id') or data.get('handoff_id')}",
        f"Task: {data.get('task')}",
        f"Changed files: {len(data.get('changed_files', []))}",
        f"Review summary: {latest_review.get('task') or latest_review.get('id') or 'none'}",
        f"Tests: {data.get('tests') or 'not recorded'}",
        "",
        "Risks and warnings:",
    ]
    lines.extend([f"- {warning}" for warning in data.get("warnings", [])] or ["- none"])
    lines.extend([
        "",
        "Suggested tests:",
        "- see review.md",
        "",
        "Saved:",
        f"- {saved.get('handoff_md')}",
        f"- {saved.get('handoff_json')}",
    ])
    return "\n".join(lines)


def _setup_text(data: dict) -> str:
    mcp_results = data.get("configured", [])
    skipped = [item for item in mcp_results if item.get("status") == "skipped"]
    title = "Chronicle agent setup partially complete" if skipped else "Chronicle agent setup complete"
    lines = [title, "", f"Repo: {data.get('repo')}"]
    if data.get("updated"):
        lines.extend(["", "Updated:"])
    for item in data.get("updated", []):
        lines.append(f"- {item.get('agent')}: {item.get('action')} {item.get('path')}")
    if mcp_results:
        lines.extend(["", "Configured:"])
    for item in mcp_results:
        if item.get("status") == "skipped":
            continue
        status = item.get("status")
        suffix = f" ({status})" if status else ""
        lines.append(f"- {item.get('agent')} MCP server: {item.get('name')}{suffix}")
    if skipped:
        lines.extend(["", "Skipped:"])
        for item in skipped:
            lines.append(f"- {item.get('agent')} MCP setup because {item.get('reason')}")
        lines.extend(["", "To add MCP later:"])
        for command in dict.fromkeys(item.get("manual_command") for item in skipped if item.get("manual_command")):
            lines.append(f"  {command}")
    lines.extend(["", "Next:", "  In your agent, ask it to use Chronicle before large code changes."])
    return "\n".join(lines)


def _pr_review_text(data: dict) -> str:
    saved = data.get("saved") or {}
    lines = [
        "Chronicle PR review",
        "",
        f"Run ID: {data.get('run_id')}",
        f"Base: {data.get('base')}",
        f"Risk level: {data.get('risk_level')}",
        f"Changed files: {len(data.get('changed_files', []))}",
        f"Impacted symbols: {len(data.get('impacted_symbols', []))}",
        "",
        "Suggested tests:",
    ]
    lines.extend([f"- {test}" for test in data.get("suggested_tests", [])] or ["- none found"])
    lines.extend(["", "Warnings:"])
    lines.extend([f"- {warning}" for warning in data.get("warnings", [])] or ["- none"])
    lines.extend(["", "Saved:", f"- {saved.get('pr_review_md')}"])
    return "\n".join(lines)


def _doctor_next_steps(args: argparse.Namespace) -> dict[str, str]:
    source = _source_flags(args)
    query = getattr(args, "query", None)
    token_budget = getattr(args, "token_budget", None)
    session_id = getattr(args, "session_id", None)
    budget_flag = f" --token-budget {token_budget}" if token_budget else ""
    session_flag = f" --session-id {session_id}" if session_id else ""
    if not query:
        return {
            "index": f"PYTHONPATH=src python3 -m chronicle.cli index {source}".strip(),
            "session_start": f"PYTHONPATH=src python3 -m chronicle.cli session-start {source}".strip(),
        }
    return {
        "evaluate": f'PYTHONPATH=src python3 -m chronicle.cli evaluate "{query}" {source}{budget_flag}{session_flag}'.strip(),
        "demo": f'PYTHONPATH=src python3 -m chronicle.cli demo "{query}" {source}{budget_flag}{session_flag}'.strip(),
        "ab_test": (
            f'PYTHONPATH=src python3 -m chronicle.cli ab-test "{query}" {source}{budget_flag}{session_flag} '
            "--baseline-token-budget 12000 --model <ollama-model>"
        ).strip(),
    }


def _source_flags(args: argparse.Namespace) -> str:
    if getattr(args, "repo_url", None):
        flags = [f'--repo-url {args.repo_url}']
        if getattr(args, "repos_dir", None):
            flags.append(f'--repos-dir {args.repos_dir}')
        if getattr(args, "branch", None):
            flags.append(f'--branch {args.branch}')
        return " ".join(flags)
    flags = [f'--repo {args.repo}']
    if getattr(args, "index_dir", None):
        flags.append(f'--index-dir {args.index_dir}')
    return " ".join(flags)


def _compact_context(data: dict) -> dict:
    selected_symbols = [
        {
            "name": symbol.get("name"),
            "file_path": symbol.get("file_path"),
            "start_line": symbol.get("start_line"),
        }
        for symbol in data.get("selected_symbols", [])[:6]
    ]
    focus_areas = ((data.get("llm_brief") or {}).get("focus_areas") or [])[:4]
    llm_decision = data.get("llm_decision") or {}
    decision_suffix = ""
    if llm_decision.get("call_llm") is False:
        decision_suffix = f" Chronicle recommends no LLM call: {llm_decision.get('reason')}"
    return {
        "query": data.get("query"),
        "session_id": data.get("session_id"),
        "token_budget": data.get("token_budget"),
        "estimated_tokens": data.get("estimated_tokens"),
        "confidence": data.get("confidence"),
        "human_summary": (
            f"Chronicle selected {len(data.get('selected_symbols', []))} symbols"
            f" and estimates {data.get('estimated_tokens')} input tokens."
            + (f" Focus first on {', '.join(focus_areas)}." if focus_areas else "")
            + decision_suffix
        ),
        "selected_symbols": selected_symbols,
        "call_chain_summary": ((data.get("call_chain") or {}).get("summary") or None),
        "patch_context_summary": ((data.get("patch_context") or {}).get("summary") or None),
        "llm_brief": data.get("llm_brief"),
        "llm_decision": llm_decision,
        "memory_summary": data.get("memory_summary"),
    }


def _compact_evaluate(data: dict) -> dict:
    reduction = data.get("token_reduction_percent")
    confidence = data.get("benchmark_confidence")
    return {
        "repo": data.get("repo"),
        "query": data.get("query"),
        "human_summary": (
            f"Chronicle uses {data.get('chronicle_tokens')} estimated tokens instead of {data.get('baseline_tokens')}"
            + (f" ({reduction}% reduction)." if reduction is not None else ".")
            + (f" Benchmark confidence is {confidence}." if confidence else "")
        ),
        "baseline_tokens": data.get("baseline_tokens"),
        "chronicle_tokens": data.get("chronicle_tokens"),
        "token_reduction_percent": data.get("token_reduction_percent"),
        "retrieval_hit_rate": data.get("retrieval_hit_rate"),
        "answer_grounding_score": data.get("answer_grounding_score"),
        "benchmark_confidence": data.get("benchmark_confidence"),
        "recommendation": data.get("recommendation"),
    }


def _compact_doctor(data: dict) -> dict:
    diagnosis = data.get("query_diagnosis") or {}
    status = diagnosis.get("status") or "unknown"
    human_summary = data.get("human_summary") or (
        f"Repository health is {((data.get('health') or {}).get('status') or 'unknown')} "
        f"and query match quality is {status}."
    )
    return {
        "repo": data.get("repo"),
        "health": data.get("health"),
        "query": data.get("query"),
        "human_summary": human_summary,
        "query_diagnosis": data.get("query_diagnosis"),
        "selected_symbols": data.get("selected_symbols", [])[:6],
        "top_matches": data.get("top_matches", [])[:5],
        "call_chain_summary": data.get("call_chain_summary"),
        "patch_context_summary": data.get("patch_context_summary"),
        "llm_brief": data.get("llm_brief"),
        "session": data.get("session"),
        "next_steps": data.get("next_steps"),
    }


def _compact_demo(data: dict) -> dict:
    evaluation = data.get("evaluation") or {}
    return {
        "repo": data.get("repo"),
        "query": data.get("query"),
        "human_summary": (
            f"Index built successfully and Chronicle reduced estimated input from "
            f"{evaluation.get('baseline_tokens')} to {evaluation.get('chronicle_tokens')} tokens."
            if evaluation
            else "Chronicle completed the end-to-end demo."
        ),
        "index_health": ((data.get("index") or {}).get("health")),
        "selected_symbols": ((data.get("context") or {}).get("selected_symbols", []))[:5],
        "call_chain_summary": (((data.get("context") or {}).get("call_chain") or {}).get("summary") if isinstance((data.get("context") or {}).get("call_chain"), dict) else None),
        "token_savings": {
            "baseline_tokens": ((data.get("evaluation") or {}).get("baseline_tokens")),
            "chronicle_tokens": ((data.get("evaluation") or {}).get("chronicle_tokens")),
            "token_reduction_percent": ((data.get("evaluation") or {}).get("token_reduction_percent")),
            "benchmark_confidence": ((data.get("evaluation") or {}).get("benchmark_confidence")),
        },
    }


def _compact_ab_test(data: dict) -> dict:
    baseline = data.get("baseline") or {}
    chronicle = data.get("chronicle") or {}
    comparison = data.get("comparison") or {}
    return {
        "query": data.get("query"),
        "model": data.get("model"),
        "human_summary": comparison.get("winner_summary")
        or (
            f"Chronicle reduced estimated input tokens by {comparison.get('input_token_reduction_percent')}% "
            f"for model {data.get('model')}."
        ),
        "baseline": {
            "estimated_input_tokens": baseline.get("estimated_input_tokens"),
            "estimated_output_tokens": baseline.get("estimated_output_tokens"),
            "selected_symbols": baseline.get("selected_symbols", [])[:5],
            "repaired": baseline.get("repaired"),
            "validation": baseline.get("validation"),
        },
        "chronicle": {
            "estimated_input_tokens": chronicle.get("estimated_input_tokens"),
            "estimated_output_tokens": chronicle.get("estimated_output_tokens"),
            "selected_symbols": chronicle.get("selected_symbols", [])[:5],
            "repaired": chronicle.get("repaired"),
            "validation": chronicle.get("validation"),
            "llm_brief": (((chronicle.get("context_pack") or {}).get("llm_brief")) if isinstance(chronicle.get("context_pack"), dict) else None),
        },
        "comparison": comparison,
    }


def _compact_call_chain(data: dict) -> dict:
    return {
        "query": data.get("query"),
        "entry_symbol": data.get("entry_symbol"),
        "human_summary": (
            f"Call chain starts at {data.get('entry_symbol') or 'the best matched symbol'} "
            f"and fits within about {data.get('context_estimated_tokens')} context tokens."
        ),
        "selected_symbols": data.get("selected_symbols", [])[:6],
        "summary": data.get("summary"),
        "mermaid": data.get("mermaid"),
        "context_estimated_tokens": data.get("context_estimated_tokens"),
    }


def _compact_session(data: dict) -> dict:
    return {
        "session_id": data.get("session_id"),
        "repo_path": data.get("repo_path"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "turn_count": len(data.get("turns", [])),
        "human_summary": f"Session {data.get('session_id')} is ready for multi-turn memory reuse.",
    }


def _compact_session_show(data: dict) -> dict:
    session = data.get("session") or {}
    summary = data.get("summary") or {}
    turns = session.get("turns", [])
    recent_turns = [
        {
            "query": turn.get("query"),
            "intent": turn.get("intent"),
            "selected_symbols": turn.get("selected_symbols", [])[:4],
            "validation_confidence": turn.get("validation_confidence"),
            "grounded": turn.get("grounded"),
        }
        for turn in turns[-3:]
    ]
    return {
        "summary": summary,
        "human_summary": (
            f"Session has {summary.get('turn_count', len(turns))} recorded turns and "
            f"{len(recent_turns)} recent turns are shown."
        ),
        "recent_turns": recent_turns,
    }


def _compact_bus(data: dict) -> dict:
    bus = data.get("bus") or {}
    summary = data.get("summary") or {}
    phases = bus.get("phases", [])
    latest = phases[-1] if phases else None
    compact_latest = None
    if latest:
        context_pack = latest.get("context_pack") or {}
        compact_latest = {
            "role": latest.get("role"),
            "query": latest.get("query"),
            "token_budget": latest.get("token_budget"),
            "selected_symbols": [
                {
                    "name": symbol.get("name"),
                    "file_path": symbol.get("file_path"),
                    "start_line": symbol.get("start_line"),
                }
                for symbol in context_pack.get("selected_symbols", [])[:6]
            ],
            "call_chain_summary": ((context_pack.get("call_chain") or {}).get("summary") or None),
            "patch_context_summary": ((context_pack.get("patch_context") or {}).get("summary") or None),
            "llm_brief": context_pack.get("llm_brief"),
            "llm_decision": latest.get("llm_decision"),
            "validation": latest.get("validation"),
        }
    return {
        "summary": summary,
        "human_summary": (
            f"Bus {summary.get('bus_id')} has {summary.get('phase_count', len(phases))} phases"
            f" and the latest role is {summary.get('latest_role') or 'unknown'}."
        ),
        "latest_phase": compact_latest,
        "recent_handoffs": (bus.get("handoffs") or [])[-3:],
    }


if __name__ == "__main__":
    raise SystemExit(main())
