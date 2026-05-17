QUERY_REWRITE_PROMPT = "Rewrite the query for code retrieval without adding facts."
VALIDATION_REPAIR_PROMPT = "Repair the answer using only grounded repository context."


def build_answer_prompt(query: str, context: str) -> str:
    return (
        "You are answering a codebase question using only the provided grounded context.\n"
        "Rules:\n"
        "- Answer the user's exact question in the first sentence.\n"
        "- Cite exact file paths and symbol names from the context.\n"
        "- Treat any behavior-boundary notes in the context as constraints on attribution.\n"
        "- Distinguish direct ownership, runtime wiring, and downstream helpers when relevant.\n"
        "- If the context is partial or inconclusive, say that explicitly instead of guessing.\n"
        "- Keep the answer compact and avoid generic code advice.\n\n"
        f"Question: {query}\n\n"
        "Grounded context:\n"
        f"{context}\n\n"
        "Answer:"
    )


def build_repair_prompt(query: str, context: str, draft_answer: str, issues: list[str]) -> str:
    issue_lines = "\n".join(f"- {issue}" for issue in issues[:6]) or "- Answer was not grounded enough."
    return (
        "You are repairing a codebase answer using only grounded repository context.\n"
        "Remove unsupported claims. Prefer exact file paths and symbol names from context. "
        "Separate direct behavior from adjacent wiring or helpers. "
        "If the context is still insufficient, say what is missing instead of guessing.\n\n"
        f"Question: {query}\n\n"
        "Validation issues to fix:\n"
        f"{issue_lines}\n\n"
        "Original answer:\n"
        f"{draft_answer}\n\n"
        "Grounded context:\n"
        f"{context}\n\n"
        "Repaired answer:"
    )
