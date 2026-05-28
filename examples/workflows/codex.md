# Codex Workflow

```bash
pip install chronicle-sdk
chronicle setup codex
chronicle run "Fix auth token refresh bug"
chronicle finish
```

`chronicle setup codex` updates `AGENTS.md` and attempts to register the `chronicle` MCP server with Codex. Give Codex `chronicle_logs/runs/<run_id>/agent_prompt.md` before it edits code.
