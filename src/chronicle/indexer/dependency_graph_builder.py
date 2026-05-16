from __future__ import annotations

import ast

from .ast_parser import ParsedModule


class DependencyGraphBuilder:
    def build(self, modules: list[ParsedModule]) -> dict[str, list[str]]:
        graph: dict[str, list[str]] = {}
        for module in modules:
            if module.tree is None:
                continue
            imports: list[str] = []
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    imports.extend(
                        f"{module_name}.{alias.name}".strip(".")
                        for alias in node.names
                    )
            graph[module.relative_path] = list(dict.fromkeys(imports))
        return graph
