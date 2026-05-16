from __future__ import annotations

import ast
from collections import defaultdict
import textwrap

from ..core.models import Symbol


class CallGraphBuilder:
    def build(self, symbols: list[Symbol]) -> dict[str, list[str]]:
        name_index = defaultdict(list)
        for symbol in symbols:
            name_index[symbol.name.split(".")[-1]].append(symbol.id)

        graph: dict[str, list[str]] = {}
        for symbol in symbols:
            try:
                tree = ast.parse(textwrap.dedent(symbol.body))
            except SyntaxError:
                graph[symbol.id] = []
                continue
            collector = _CallCollector()
            collector.visit(tree)
            targets: list[str] = []
            for callee_name in collector.calls:
                matches = name_index.get(callee_name, [])
                targets.extend(matches or [callee_name])
            symbol.calls = list(dict.fromkeys(targets))
            graph[symbol.id] = list(symbol.calls)
        return graph


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        name = self._name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)

    def _name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None
