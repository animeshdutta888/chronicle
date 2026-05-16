from __future__ import annotations

import argparse
from pathlib import Path

from chronicle import Chronicle
from chronicle.llm.providers import OllamaError, OllamaProvider


DEFAULT_QUERY = "How does ManagerAgent orchestrate reminders and memory?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chronicle against a local repo, then call a local Ollama model with the SDK packet."
    )
    parser.add_argument(
        "--repo",
        default="/Users/animeshdutta/Projects/Nudge_git/Nudge",
        help="Absolute path to the local repository Chronicle should index.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question to ask about the repository.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b-instruct",
        help="Local Ollama model name.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=3000,
        help="Chronicle token budget for the compressed context pack.",
    )
    parser.add_argument(
        "--baseline-token-budget",
        type=int,
        default=12000,
        help="Baseline token budget used for the direct full-context comparison.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline vs Chronicle comparison and print both answers plus token usage.",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()
    chronicle = Chronicle(repo_path=repo_path)
    provider = OllamaProvider()

    if args.compare:
        run_comparison(
            chronicle=chronicle,
            provider=provider,
            query=args.query,
            model=args.model,
            token_budget=args.token_budget,
            baseline_token_budget=args.baseline_token_budget,
        )
        return

    packet = chronicle.prepare_prompt_packet(
        query=args.query,
        token_budget=args.token_budget,
    )

    print("Chronicle summary:")
    print(packet.human_summary)
    print()
    print(f"Should call LLM: {packet.should_call_llm}")
    print(f"Selected symbols: {', '.join(packet.selected_symbols) or 'none'}")
    print(f"Estimated input tokens: {packet.estimated_input_tokens}")
    print()

    if not packet.should_call_llm or not packet.prompt:
        print("Chronicle recommends skipping the model call for this query.")
        print("Compressed context preview:")
        print(packet.compressed_context[:800])
        return

    try:
        response_text = provider.generate_text(args.model, packet.prompt)
    except OllamaError as exc:
        print(f"Ollama request failed: {exc}")
        return

    print("Model response:")
    print(response_text or "<empty response>")

    context = chronicle.context(args.query, token_budget=args.token_budget, remember=False)
    validation = chronicle.validate_output(response_text or "", context)
    print()
    print(
        f"Validation: valid={validation.valid} grounded={validation.grounded} "
        f"confidence={validation.confidence:.2f}"
    )
    if validation.issues:
        print("Issues:")
        for issue in validation.issues:
            print(f"- {issue}")


def run_comparison(
    *,
    chronicle: Chronicle,
    provider: OllamaProvider,
    query: str,
    model: str,
    token_budget: int,
    baseline_token_budget: int,
) -> None:
    try:
        report = chronicle.ab_test(
            query=query,
            model=model,
            token_budget=token_budget,
            baseline_token_budget=baseline_token_budget,
            llm_provider=provider,
        )
    except (OllamaError, RuntimeError) as exc:
        print(f"Comparison failed: {exc}")
        return

    baseline = report["baseline"]
    chronicle_arm = report["chronicle"]
    comparison = report["comparison"]

    print("Baseline")
    print(f"Input tokens: {baseline['estimated_input_tokens']}")
    print(f"Output tokens: {baseline['estimated_output_tokens']}")
    print(f"Grounded: {baseline['validation']['grounded']}")
    print(f"Confidence: {baseline['validation']['confidence']}")
    print()
    print(baseline["answer"])
    print()
    print("Chronicle")
    print(f"Input tokens: {chronicle_arm['estimated_input_tokens']}")
    print(f"Output tokens: {chronicle_arm['estimated_output_tokens']}")
    print(f"Grounded: {chronicle_arm['validation']['grounded']}")
    print(f"Confidence: {chronicle_arm['validation']['confidence']}")
    print()
    print(chronicle_arm["answer"])
    print()
    print("Comparison")
    print(f"Input token reduction: {comparison['input_token_reduction_percent']}%")
    print(f"Answer similarity: {comparison['answer_similarity']}")
    print(f"Same or better grounding: {comparison['same_or_better_grounding']}")
    print(f"Both grounded: {comparison['both_grounded']}")
    print(f"Winner summary: {comparison['winner_summary']}")


if __name__ == "__main__":
    main()
