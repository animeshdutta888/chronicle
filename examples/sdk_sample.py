from __future__ import annotations

import argparse
from pathlib import Path

from chronicle import Chronicle


DEFAULT_REPO = "/Users/animeshdutta/Projects/Nudge_git"
DEFAULT_TASK = "Improve ManagerAgent.run reminder orchestration safely"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chronicle SDK full-cycle example: prepare, status, review, and handoff."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Local repository to analyze. Defaults to the local Nudge repo path used during development.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task to prepare for the coding agent.",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Optional test command/result to include in the handoff, for example: 'pytest tests/test_manager.py passed'.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=3000,
        help="Chronicle token budget for prepare/review packets.",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()
    chronicle = Chronicle(repo_path=repo_path)

    print("1. Prepare context for the coding agent")
    prepared = chronicle.prepare(args.task, token_budget=args.token_budget)
    print(f"Prepared: {prepared['saved']['context_md']}")
    print(f"Selected files: {', '.join(prepared['selected_files']) or 'none'}")
    print()

    print("2. Check Chronicle status")
    status = chronicle.status()
    print(f"Index: {status['index_status']} ({status['symbol_count']} symbols)")
    print(f"Changed files: {', '.join(status['changed_files']) or 'none'}")
    print()

    print("3. Review recent changes")
    review = chronicle.review(
        "Review recent changes and impacted tests for this task",
        token_budget=args.token_budget,
    )
    print(f"Review: {review['saved']['review_md']}")
    print(f"Related tests: {', '.join(review['related_tests']) or 'none found'}")
    if review["warnings"]:
        print("Review warnings:")
        for warning in review["warnings"]:
            print(f"- {warning}")
    print()

    print("4. Create handoff")
    handoff = chronicle.handoff(
        task=args.task,
        tests=args.tests,
        notes=[
            "Use the prepare packet before implementation.",
            "Use the review packet after edits and tests.",
        ],
    )
    print(f"Handoff: {handoff['saved']['handoff_md']}")
    print()

    print("Cycle")
    print("prepare -> agent edits -> tests -> review -> handoff")


if __name__ == "__main__":
    main()
