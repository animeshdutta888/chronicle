# Chronicle

Local-first ContextOps for AI coding agents.

Chronicle prepares grounded repo context before agents code, reviews changes after they code, and creates replayable handoffs for humans or the next agent.

> Alpha release: Chronicle is currently strongest on Python `.py` repositories.

## Quickstart

```bash
pip install chronicle-sdk
chronicle setup codex
chronicle prepare "Fix auth token refresh bug"
chronicle review
chronicle handoff
```

Chronicle saves artifacts under `chronicle_logs/runs/<run_id>/`:

```text
prepare.md
review.md
handoff.md
pr-review.md
prepare.json
```

Use `prepare.md` as the primary packet for Codex, Claude, Cursor, or another coding agent.

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

## Daily Loop

```bash
chronicle prepare "Fix auth token refresh bug"
# agent edits code
chronicle review
chronicle handoff --tests "pytest passed"
```

- `prepare` selects relevant files, symbols, tests, warnings, and writes `prepare.md`.
- `review` inspects changed files, impacted symbols, related tests, warnings, and writes `review.md`.
- `handoff` summarizes the latest prepare/review state and writes `handoff.md`.

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
