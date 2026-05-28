# Claude Workflow

```bash
pip install chronicle-sdk
chronicle setup claude
chronicle run "Fix auth token refresh bug"
chronicle finish
```

`chronicle setup claude` updates `CLAUDE.md` and attempts to register the `chronicle` MCP server with Claude Code. Use `chronicle_logs/runs/<run_id>/agent_prompt.md` as the primary Claude prompt.
