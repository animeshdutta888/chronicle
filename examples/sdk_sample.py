from __future__ import annotations

import argparse
from pathlib import Path

from chronicle import Chronicle
from chronicle.llm.providers import OllamaError, OllamaProvider


DEFAULT_QUERY = "How does ManagerAgent orchestrate reminders and memory?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDK-first Chronicle example: build one prompt packet, call Ollama, and print packet stats."
    )
    parser.add_argument(
        "--repo",
        default=".",
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
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser().resolve()
    chronicle = Chronicle(repo_path=repo_path)
    provider = OllamaProvider()

    packet = chronicle.prepare_prompt_packet(
        query=args.query,
        token_budget=args.token_budget,
    )
    report = chronicle.evaluate(
        query=args.query,
        token_budget=args.token_budget,
    )

    print("Chronicle SDK packet")
    print(f"Query: {args.query}")
    print(f"Should call LLM: {packet.should_call_llm}")
    print(f"Selected symbols: {', '.join(packet.selected_symbols) or 'none'}")
    print(f"Estimated input tokens: {packet.estimated_input_tokens}")
    print(f"Response policy: {packet.response_policy}")
    print()
    print("Token summary")
    print(f"Baseline tokens: {report.baseline_tokens}")
    print(f"Chronicle tokens: {report.chronicle_tokens}")
    print(f"Reduction: {report.token_reduction_percent}%")
    print(f"Grounding score: {report.answer_grounding_score}")
    print(f"Recommendation: {report.recommendation}")
    print()

    if not packet.should_call_llm or not packet.prompt:
        print("Chronicle recommends skipping the model call for this query.")
        print()
        print("Compressed context preview:")
        print(packet.compressed_context[:1200])
        return

    try:
        response_text = provider.generate_text(args.model, packet.prompt)
    except OllamaError as exc:
        print(f"Ollama request failed: {exc}")
        return

    print("Prompt preview")
    print(packet.prompt[:1500])
    print()
    print("Model response")
    print(response_text or "<empty response>")
    print()

    context = chronicle.context(args.query, token_budget=args.token_budget, remember=False)
    validation = chronicle.validate_output(response_text or "", context)
    print("Validation")
    print(
        f"Valid: {validation.valid} | Grounded: {validation.grounded} | "
        f"Confidence: {validation.confidence:.2f}"
    )
    if validation.issues:
        print("Issues:")
        for issue in validation.issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
