from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server

from .integrations import ChronicleMCPServer


class ChronicleMCPStdioServer:
    """Chronicle MCP server backed by the official Python MCP SDK."""

    def __init__(self, repo_path: str | Path, index_dir: str | Path | None = None) -> None:
        self.adapter = ChronicleMCPServer(repo_path=repo_path, index_dir=index_dir)
        self.repo_path = Path(repo_path)
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.server: Server[None, None] = Server(
            name="chronicle-mcp",
            version="0.1.0",
            instructions=(
                "Use Chronicle to index Python repositories, retrieve grounded context, "
                "trace call chains, and manage multi-agent context buses."
            ),
        )
        self._register_resources()
        self._register_tools()

    def _register_resources(self) -> None:
        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    name="chronicle-server-info",
                    title="Chronicle Server Info",
                    uri="chronicle://server-info",
                    description="Chronicle MCP server metadata and default repository settings.",
                    mimeType="application/json",
                )
            ]

        @self.server.read_resource()
        async def read_resource(uri: Any) -> str:
            if str(uri) != "chronicle://server-info":
                raise ValueError(f"Unsupported Chronicle resource: {uri}")
            payload = {
                "name": "chronicle-mcp",
                "repo_path": str(self.repo_path),
                "index_dir": str(self.index_dir) if self.index_dir is not None else None,
                "tool_count": len(self.adapter.tool_definitions()),
            }
            return json.dumps(payload, indent=2, sort_keys=True)

    def _register_tools(self) -> None:
        tool_specs = self.adapter.tool_definitions()

        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=spec["name"],
                    description=spec["description"],
                    inputSchema=spec["inputSchema"],
                )
                for spec in tool_specs
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return self.adapter.handle(name, arguments)

    async def run_async(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )

    def run(self) -> None:
        anyio.run(self.run_async)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chronicle-mcp")
    parser.add_argument("--repo", default=".", help="Path to the local repository Chronicle should analyze.")
    parser.add_argument("--index-dir", default=None, help="Optional Chronicle index directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ChronicleMCPStdioServer(repo_path=args.repo, index_dir=args.index_dir)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
