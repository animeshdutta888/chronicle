# Chronicle

Chronicle is an **AI Context Operating System for Coding Agents**.

> **Alpha Release**  
> Chronicle is live as an early Python-first alpha. The core context engine is usable today, while language coverage, hosted workflows, and evaluation depth are still expanding.

> **Current limitation:** Chronicle is currently best for structured Python `.py` repositories.

It indexes repository structure, ranks the smallest useful context for a coding task, compresses that context to fit a token budget, records provenance for every chunk, and decides whether an LLM is needed at all.

## Why Chronicle

- Reduce prompt tokens before the LLM call
- Keep answers grounded in real files, symbols, and flow
- Reuse context across CLI, SDK, hosted, and agent workflows
- Block low-signal model calls before they waste budget

## What Chronicle optimizes

- Accuracy per token
- Deterministic retrieval before probabilistic reasoning
- Grounded context with file and symbol provenance
- Lower token spend without lowering answer quality

## Current MVP capabilities

- Python AST indexing for functions, classes, and methods
- Symbol graph and import dependency graph construction
- Git evolution summaries for churn, risky changes, and symbol-to-file history
- Deterministic query planning and context ranking
- Ownership-aware context shaping that separates direct behavior, runtime wiring, and adjacent helpers
- Token-budget-aware context compression
- LLM routing decisions without forcing an LLM call
- Output validation for grounded file and symbol references
- Secret redaction guardrails before external LLM calls
- SQLite-backed snapshot persistence with JSON compatibility
- SQLite-backed session memory for multi-turn context recall
- Patch-aware contexting for edited symbols, callers/callees, tests, and interfaces
- Multi-agent context bus for planner/coder/reviewer/critic handoffs
- Confidence gating to block low-signal LLM calls before they waste tokens
- Grounded repair loop to retry weak answers using the same validated context
- MCP-compatible integration scaffold for external agent systems
- Evaluation metrics with benchmark confidence and recommendation output

## Current language support

- Production-usable now: Python `.py` repositories
- Partial only: notebook-heavy Python repos where most logic lives in `.ipynb`
- Planned next: Go, Rust, TypeScript / JavaScript
- Later: C / C++

Chronicle should not be treated as benchmark-grade on non-Python repos until symbol extraction, dependency understanding, patch-aware retrieval, and grounding are implemented and validated for that language.

## Python SDK

Chronicle is ready to ship as a public Python package while still working against private codebases in your own environment before LLM calls.

### Install from PyPI

```bash
pip install chronicle-sdk
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install the public SDK directly:

```bash
pip install chronicle-sdk
```

Run the hosted alpha API locally:

```bash
pip install -e .[hosted]
chronicle-api
```

Index a repository:

```bash
chronicle index --repo /path/to/repo
```

Retrieve grounded context:

```bash
chronicle context "Where is auth token refresh handled?" --repo /path/to/repo --token-budget 3000
```

Get the full machine-auditable payload when needed:

```bash
chronicle context "Where is auth token refresh handled?" --repo /path/to/repo --token-budget 3000 --view full
```

Evaluate Chronicle against a baseline:

```bash
chronicle evaluate "Where is auth token refresh handled?" --repo /path/to/repo --token-budget 3000
```

Diagnose whether a repo indexed correctly:

```bash
chronicle doctor --repo /path/to/repo --query "Where is RequestContext defined?" --token-budget 2500
```

Run an end-to-end token-savings demo:

```bash
chronicle demo "Where is RequestContext defined?" --repo-url https://github.com/pallets/flask.git --token-budget 2500
```

Run an A/B LLM comparison with and without Chronicle context:

```bash
chronicle ab-test "Where is full_dispatch_request defined?" \
  --repo-url https://github.com/pallets/flask.git \
  --token-budget 2500 \
  --baseline-token-budget 12000 \
  --model qwen2.5:14b-instruct
```

Render a functional call chain as text plus Mermaid:

```bash
chronicle call-chain "Trace how ManagerAgent.run reaches retry logic" \
  --repo /path/to/repo \
  --token-budget 2200
```

Use patch-aware contexting after local code changes:

```bash
chronicle doctor \
  --repo /path/to/repo \
  --query "Enhance ManagerAgent.run to support retry and update impacted tests" \
  --token-budget 2200
```

Start a reusable Chronicle session for multi-turn memory:

```bash
chronicle session-start --repo /path/to/repo
```

Use session-aware contexting across turns:

```bash
chronicle context "Where is ManagerAgent.run defined?" \
  --repo /path/to/repo \
  --token-budget 2200 \
  --session-id session-abc123

chronicle context "How does ManagerAgent.run call retry logic?" \
  --repo /path/to/repo \
  --token-budget 2200 \
  --session-id session-abc123
```

Inspect recorded session memory:

```bash
chronicle session-show --repo /path/to/repo --session-id session-abc123
```

Create and use a shared multi-agent context bus:

```bash
chronicle bus-start "Improve ManagerAgent.run flow" --repo /path/to/repo --bus-id feature-bus

chronicle bus-context "Plan ManagerAgent.run enhancement" \
  --repo /path/to/repo \
  --bus-id feature-bus \
  --role planner \
  --token-budget 2200

chronicle bus-handoff \
  --repo /path/to/repo \
  --bus-id feature-bus \
  --from-role planner \
  --to-role coder \
  --reason "Context is grounded"

chronicle bus-show --repo /path/to/repo --bus-id feature-bus
```

Python SDK:

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
chronicle.index()

context = chronicle.context(
    query="Where is auth token refresh handled?",
    token_budget=3000,
)

print(context.to_markdown())
```

SDK packet for your own LLM call:

```python
from chronicle import Chronicle

chronicle = Chronicle(repo_path="./repo")
packet = chronicle.prepare_prompt_packet(
    query="How should I refactor the retry path?",
    token_budget=3000,
)

if packet.should_call_llm and packet.prompt:
    prompt = packet.prompt
else:
    prompt = packet.compressed_context
```

The SDK packet gives you:

- `compressed_context` for the smallest grounded repo slice
- `response_policy` for output length and format control
- `should_call_llm` to block weak model calls
- `behavior boundaries` inside the context pack so LLMs can avoid misattributing nearby code

## Retrieval architecture

Chronicle’s retrieval path is now deliberately quality-first for synthesis queries:

1. **Intent and concept planning**
   - classify the query (`locator`, `performance`, `dataflow`, `refactor`, etc.)
   - extract stable query concepts, not just raw keywords

2. **Deterministic ranking**
   - score symbols using exact matches, normalized concept matches, file proximity, graph proximity, patch hints, and session memory

3. **Coverage-aware diversification**
   - avoid collapsing onto only one strong symbol
   - preserve cross-concept coverage when the question spans multiple behaviors or layers

4. **Ownership-aware enrichment**
   - expand anchor classes into key methods
   - surface helper evidence from selected execution paths
   - add boundary notes that distinguish direct ownership from adjacent runtime wiring

5. **Focused compression**
   - keep fuller bodies for anchor surfaces
   - use query-aware excerpts for long support methods so relevant branches survive the token cut

This means Chronicle is no longer just “top-k symbols under a budget.” It is trying to preserve the smallest grounded packet that still keeps behavior boundaries intact.

## Comparing Chronicle vs baseline

When you run `chronicle ab-test` or the sample Ollama comparison, read the results in this order:

1. `Winner summary`
2. `Grounded` / `Both grounded`
3. `Input token reduction`
4. `Answer similarity`

High token reduction alone is not a quality win. Chronicle should only be treated as better when the answer stays on-task, grounded, and materially as useful as the baseline.
- `prompt` when Chronicle recommends a model call
- `selected_symbols` and `selected_files` for tracing and logging

Run the local SDK example against the Nudge repo with Ollama:

```bash
PYTHONPATH=src python3 examples/sample_nudge_sdk_ollama.py \
  --repo /Users/animeshdutta/Projects/Nudge_git/Nudge \
  --model qwen2.5:14b-instruct
```

Run the SDK-first single-packet example to print packet stats, token reduction, and one Ollama response:

```bash
PYTHONPATH=src python3 examples/sdk_sample.py \
  --repo /Users/animeshdutta/Projects/Nudge_git/Nudge \
  --model qwen2.5:14b-instruct
```

Run the same example in comparison mode to print baseline vs Chronicle token usage and both model responses:

```bash
PYTHONPATH=src python3 examples/sample_nudge_sdk_ollama.py \
  --repo /Users/animeshdutta/Projects/Nudge_git/Nudge \
  --model qwen2.5:14b-instruct \
  --compare
```

LangGraph-style integration:

```python
from chronicle.integrations.langgraph_node import ChronicleContextNode

node = ChronicleContextNode(repo_path="./repo", token_budget=4000)
result = node({"query": "Trace checkout retries"})
```

## Hosted alpha deployment

Chronicle now includes a minimal FastAPI service for a Python-only hosted alpha.

For the full deployment runbook, see `chronicle/DEPLOYMENT.md:1`.
For the shortest founder shipping path, see `chronicle/LAUNCH_CHECKLIST.md:1`.

Available endpoints:

- `GET /health`
- `POST /index`
- `POST /doctor`
- `POST /demo`
- `POST /context`
- `POST /evaluate`
- `POST /call-chain`

Customers can use Chronicle either from the landing-page demo form or through direct API calls.
If you set `CHRONICLE_API_KEY`, hosted endpoints require the `X-API-Key` header.

### Fastest free-host path

#### Render

- Push this repo to GitHub
- Create a new Web Service on Render
- Render can use `render.yaml` directly
- Build command: `pip install -e '.[hosted]'`
- Start command: `chronicle-api`
- Set environment variable: `CHRONICLE_API_KEY=replace-with-a-secret-key`

#### Railway

- Create a new project from the repo
- Use the same commands:
  - build: `pip install -e '.[hosted]'`
  - start: `chronicle-api`

#### Fly.io

- Use the included `Dockerfile`
- Then deploy with normal Fly Docker flow

### Example request

```bash
curl -X POST http://localhost:8000/doctor \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask.git",
    "query": "Where is full_dispatch_request defined?",
    "token_budget": 2500
  }'
```

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

- Chronicle is currently an alpha release: strongest on Python-repo contexting, still maturing in hosted workflows, language coverage, and benchmark depth.
- Chronicle is local-first and does not send repository code anywhere by default.
- Index artifacts are stored in `.chronicle/index.sqlite3` and mirrored to `.chronicle/index.json` unless `index_dir` is overridden.
- Session memory is stored locally in `.chronicle/sessions.sqlite3`.
- Remote repos cloned via `--repo-url` are stored in `.chronicle/repos/` by default.
- For best MVP retrieval, prefer exact symbol queries like `Where is RequestContext defined?` over broad semantic questions.
- CLI output is wrapped in a stable envelope with `status`, `command`, `generated_at`, and `data`.
- Compact CLI output is the default; use `--view full` when you want the full machine/audit payload.
- `evaluate` now includes `benchmark_confidence` and `recommendation` so token-savings reports are easier to trust.
- Session-aware `context`, `doctor`, `demo`, `evaluate`, and `ab-test` calls can reuse prior retrieved symbols, files, and validated facts.
- Flow, trace, and edit-style queries can include a compact functional call chain in the grounded context before an LLM call.
- Edit and enhancement queries can automatically include context priorities, a coverage checklist, patch-aware summaries, and an LLM task brief so the model sees changed symbols, related tests, interfaces, and likely flow with fewer tokens.
- If retrieval confidence is weak, Chronicle can recommend skipping the LLM call instead of paying for speculation.
- If an LLM answer is weak or ungrounded, Chronicle can run a grounded repair pass using the same validated context before returning the result.
- Multi-agent workflows can persist shared grounded context, deterministic handoffs, and per-phase validation in `.chronicle/agent_bus.sqlite3`.
