QUERY_REWRITE_PROMPT = "Rewrite the query for code retrieval without adding facts."
VALIDATION_REPAIR_PROMPT = "Repair the answer using only grounded repository context."


def build_answer_prompt(query: str, context: str) -> str:
    return (
        "You are answering a codebase question using only the provided grounded context.\n"
        "Cite exact file paths when possible. If the answer is not supported by the context, say so.\n\n"
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
