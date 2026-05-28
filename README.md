# Chronicle

Local-first ContextOps for AI coding agents.

Chronicle prepares grounded repo context before agents code, reviews changes after they code, and creates replayable handoffs for humans or the next agent.

> Alpha release: Chronicle is currently strongest on Python `.py` repositories.

## Quickstart

```bash
pip install chronicle-sdk
chronicle setup codex
chronicle run "Fix auth token refresh bug"
# paste the saved agent_prompt.md into Codex, Claude, or Cursor
chronicle finish
```

Chronicle saves artifacts under `chronicle_logs/runs/<run_id>/`:

```text
prepare.md
context_packet.md
agent_prompt.md
diff.patch
review.md
pr-review.md
report.md
prepare.json
run.json
```

Use `agent_prompt.md` as the ready-to-paste prompt for Codex, Claude, Cursor, or another coding agent.

## Agent Setup

```bash
chronicle setup codex
chronicle setup claude
chronicle setup cursor
chronicle setup all
chronicle setup codex --no-mcp
chronicle setup codex --mcp-only
```

Setup adds Chronicle workflow instructions and attempts MCP registration:

- Codex: appends `AGENTS.md` and runs `codex mcp add chronicle -- chronicle-mcp --repo <repo>` when `codex` is installed.
- Claude: appends `CLAUDE.md` and runs `claude mcp add chronicle -- chronicle-mcp --repo <repo>` when `claude` is installed.
- Cursor: writes `.cursor/rules/chronicle.mdc` and project `.cursor/mcp.json`.

If Codex or Claude is not installed, setup still updates the workflow file and prints the manual MCP command. Use `--no-mcp` for instruction files only, or `--mcp-only` to skip instruction files and only configure MCP.

## Two-Step Run Loop

```bash
chronicle run "Fix auth token refresh bug"
# agent edits code
chronicle finish --base main
```

- `run` prepares the smallest useful context, writes `context_packet.md`, and creates `agent_prompt.md`.
- `finish` captures `diff.patch`, runs review and PR review logic, and writes `report.md`.
- `finish` prints the report path; use `chronicle report --latest` later to print or regenerate it.
- `replay` shows the run timeline and saved artifacts.

If you run Chronicle from inside Codex, the same CLI flow works. `chronicle setup codex` also tries to register Chronicle MCP so Codex can call Chronicle tools directly, but the two-step CLI flow stays the simplest path.

The lower-level commands still exist when you want each step separately:

```bash
chronicle prepare "Fix auth token refresh bug"
chronicle review
chronicle handoff --tests "pytest passed"
```

## Local PR Review

```bash
chronicle pr-review --base main
chronicle pr-review --base main --format markdown
chronicle pr-review --base main --output pr-review.md
```

`pr-review` stays local. It uses `git diff` against the base ref, summarizes changed files and impacted symbols, suggests related tests, and writes a markdown review artifact.

## Common Commands

```bash
chronicle status
chronicle replay --latest
chronicle explain --latest
chronicle inspect --file src/auth/service.py
chronicle inspect --symbol AuthService.refresh_token
chronicle index
```

Use `--repo /path/to/repo` from outside the target repository. Use `--view full` when you need machine-readable JSON.

## SDK

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
prepared = chronicle.prepare("Fix auth token refresh bug")
review = chronicle.review()
handoff = chronicle.handoff(tests="pytest passed")

print(prepared["saved"]["prepare_md"])
print(review["saved"]["review_md"])
print(handoff["saved"]["handoff_md"])
```

## MCP

Start a Chronicle MCP server for one repo:

```bash
chronicle-mcp --repo /path/to/repo
```

Codex CLI setup:

```bash
codex mcp add chronicle -- chronicle-mcp --repo /path/to/repo
```

Claude Desktop config example:

```json
{
  "mcpServers": {
    "chronicle": {
      "command": "chronicle-mcp",
      "args": ["--repo", "/path/to/repo"]
    }
  }
}
```

## Capabilities

- Python AST indexing for functions, classes, and methods.
- Symbol graph and import dependency graph construction.
- Deterministic query planning, context ranking, and provenance.
- Token-budget-aware context compression.
- Patch-aware review context for edited symbols, related files, and tests.
- Local markdown artifacts for context, review, handoff, and PR review.
- SQLite-backed snapshot persistence and session memory.
- MCP tools for agent workflows.

## Notes

- Chronicle is local-first and does not send repository code anywhere by default.
- Index artifacts are stored in `chronicle_logs/index.sqlite3` unless `--index-dir` is provided.
- Remote repos cloned via `--repo-url` are stored in `chronicle_logs/repos/` by default.
- Non-Python language support is still partial.
