# Chronicle

Chronicle prepares grounded, explainable context packets for AI coding agents before they write code.

> **Alpha release:** Chronicle is an early Python-first alpha. The `prepare` workflow is the recommended entry point; language coverage, hosted workflows, and evaluation depth are still expanding.

> **Current limitation:** Chronicle is currently best for structured Python `.py` repositories.

## Start Here

```bash
pip install chronicle-sdk
chronicle prepare "Fix auth token refresh bug" --repo ./repo
```

Chronicle auto-indexes on first prepare run, selects relevant files/symbols/tests, warns about missing context, and saves a replayable context packet under `chronicle_logs/runs/`.

Example output:

```text
Chronicle prepared agent context

Task: Fix auth token refresh bug
Target: generic
Selected: 4 files, 8 symbols, 1 related tests
Readiness: medium - Relevant files or symbols were found.

Saved:
- chronicle_logs/runs/run_123/context.md
- chronicle_logs/runs/run_123/run.json
```

Use the generated `context.md` as the packet for Codex, Claude, Cursor, or another coding agent.

Optional target formatting:

```bash
chronicle prepare "Fix auth token refresh bug" --repo ./repo --target codex
```

`--target codex|claude|cursor|generic` does not change retrieval. It only labels/tailors packet instructions for the receiving tool. If omitted, Chronicle uses `generic`.

## Why Chronicle

- Reduce prompt tokens before an LLM call.
- Keep answers grounded in real files, symbols, tests, and provenance.
- Reuse context across CLI, SDK, MCP, hosted, and agent workflows.
- Block low-signal model calls before they waste budget.

## Main Commands

```bash
chronicle prepare "Fix auth token refresh bug" --repo ./repo
chronicle status --repo ./repo
chronicle review --repo ./repo
chronicle handoff --repo ./repo --tests "pytest passed"
chronicle replay --repo ./repo --latest
chronicle explain --repo ./repo --latest
chronicle inspect --repo ./repo --file src/auth/service.py
chronicle inspect --repo ./repo --symbol AuthService.refresh_token
```

Recommended daily loop:

```text
prepare -> agent edits -> tests -> status -> review -> handoff
```

- `prepare` creates the packet to give a coding agent before implementation.
- `status` shows whether the index is ready, what Python files changed, and the latest Chronicle artifacts.
- `review` creates a grounded packet for changed files, related symbols, and impacted tests.
- `handoff` creates a concise summary for a reviewer or the next agent.

For large repos or CI, you can index explicitly:

```bash
chronicle index --repo /path/to/repo
```

Use `--view full` on commands when you need the full machine/audit payload.

## SDK

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
prepared = chronicle.prepare("Fix auth token refresh bug")
status = chronicle.status()
review = chronicle.review()
handoff = chronicle.handoff(tests="pytest passed")

print(prepared["saved"]["context_md"])
print(review["saved"]["review_md"])
print(handoff["saved"]["handoff_md"])
```

Lower-level context retrieval is still available:

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
context = chronicle.context(
    query="Where is auth token refresh handled?",
    token_budget=3000,
)

print(context.to_markdown())
```

For custom LLM calls:

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
packet = chronicle.prepare_prompt_packet(
    query="How should I refactor the retry path?",
    token_budget=3000,
)

prompt = packet.prompt if packet.should_call_llm and packet.prompt else packet.compressed_context
```

## MCP

Start a Chronicle MCP server for one repo:

```bash
chronicle-mcp --repo /path/to/repo
```

If running from a local checkout:

```bash
python -m chronicle.mcp_stdio --repo /path/to/repo
```

Codex CLI setup:

```bash
codex mcp add chronicle -- chronicle-mcp --repo /path/to/repo
```

Codex config example:

```toml
[mcp_servers.chronicle]
command = "chronicle-mcp"
args = ["--repo", "/path/to/repo"]
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

After setup, ask your MCP client to call Chronicle `prepare`:

```json
{
  "query": "Fix auth token refresh bug"
}
```

Optional:

```json
{
  "query": "Fix auth token refresh bug",
  "target": "codex"
}
```

For the full loop, MCP clients can also call:

```json
{ "tool": "status" }
{ "tool": "review", "query": "Review recent changes and impacted tests" }
{ "tool": "handoff", "tests": "pytest passed" }
```

MCP exposes the same core implementations as the CLI and SDK. Other tools are available for advanced workflows: `index`, `context`, `evaluate`, `doctor`, `call_chain`, `prepare_prompt_packet`, `session_start`, `session_show`, `bus_start`, `bus_context`, `bus_handoff`, `bus_show`, `bus_validate_latest`, and `bus_summary`.

Chronicle also exposes `chronicle://server-info` so MCP clients can verify the server is connected.

## Capabilities

- Python AST indexing for functions, classes, and methods.
- Symbol graph and import dependency graph construction.
- Git evolution summaries for churn, risky changes, and symbol-to-file history.
- Deterministic query planning, context ranking, and provenance.
- Token-budget-aware context compression.
- LLM routing decisions without forcing an LLM call.
- Output validation for grounded file and symbol references.
- Secret redaction guardrails before external LLM calls.
- SQLite-backed snapshot persistence and session memory.
- Patch-aware contexting for edited symbols, callers/callees, tests, and interfaces.
- Multi-agent context bus for planner/coder/reviewer/critic handoffs.
- Evaluation metrics with benchmark confidence and recommendation output.

## Language Support

- Production-usable now: Python `.py` repositories.
- Partial only: notebook-heavy Python repos where most logic lives in `.ipynb`.
- Planned next: Go, Rust, TypeScript / JavaScript.
- Later: C / C++.

Chronicle should not be treated as benchmark-grade on non-Python repos until symbol extraction, dependency understanding, patch-aware retrieval, and grounding are implemented and validated for that language.

## Advanced CLI

```bash
chronicle context "Where is auth token refresh handled?" --repo /path/to/repo --token-budget 3000
chronicle evaluate "Where is auth token refresh handled?" --repo /path/to/repo --token-budget 3000
chronicle doctor --repo /path/to/repo --query "Where is RequestContext defined?"
chronicle call-chain "Trace how ManagerAgent.run reaches retry logic" --repo /path/to/repo
chronicle session-start --repo /path/to/repo
chronicle session-show --repo /path/to/repo --session-id session-abc123
chronicle ab-test "Where is full_dispatch_request defined?" --repo-url https://github.com/pallets/flask.git --model qwen2.5:14b-instruct
```

Multi-agent context bus:

```bash
chronicle bus-start "Improve ManagerAgent.run flow" --repo /path/to/repo --bus-id feature-bus
chronicle bus-context "Plan ManagerAgent.run enhancement" --repo /path/to/repo --bus-id feature-bus --role planner
chronicle bus-handoff --repo /path/to/repo --bus-id feature-bus --from-role planner --to-role coder --reason "Context is grounded"
chronicle bus-show --repo /path/to/repo --bus-id feature-bus
```

Run the SDK full-cycle example against the local Nudge repo:

```bash
PYTHONPATH=src python3 examples/sdk_sample.py \
  --repo /Users/animeshdutta/Projects/Nudge_git \
  --task "Improve ManagerAgent.run reminder orchestration safely" \
  --tests "pytest passed"
```

## Hosted Alpha

Run the hosted alpha API locally:

```bash
pip install -e .[hosted]
chronicle-api
```

Available endpoints: `GET /health`, `POST /index`, `POST /doctor`, `POST /demo`, `POST /context`, `POST /evaluate`, and `POST /call-chain`.


## Architecture

```text
User Query
  ↓
Query Planner
  ↓
Repository Intelligence Layer
  ├── AST Index
  ├── Symbol Graph
  ├── Dependency Graph
  └── Git Evolution Map
  ↓
Retrieval Orchestrator
  ↓
Context Compression Engine
  ↓
Token Budget Manager
  ↓
LLM Router
  ↓
Output Validator
  ↓
Evaluation Layer
```

## Notes

- Chronicle is local-first and does not send repository code anywhere by default.
- Index artifacts are stored in `chronicle_logs/index.sqlite3` and mirrored to `chronicle_logs/index.json` unless `index_dir` is overridden.
- Prepared agent packets are stored in `chronicle_logs/runs/`.
- Session memory is stored locally in `chronicle_logs/sessions.sqlite3`.
- Remote repos cloned via `--repo-url` are stored in `chronicle_logs/repos/` by default.
- Compact CLI output is the default for `prepare`; use `--view full` when you want JSON.
- For best MVP retrieval, prefer exact symbol queries like `Where is RequestContext defined?` over broad semantic questions.
