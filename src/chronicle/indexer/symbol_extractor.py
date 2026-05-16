from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from ..core.models import Symbol
from .ast_parser import ParsedModule


@dataclass(slots=True)
class _Scope:
    name: str
    kind: str


class SymbolExtractor:
    def extract(self, modules: list[ParsedModule]) -> list[Symbol]:
        symbols: list[Symbol] = []
        for module in modules:
            if module.tree is None:
                continue
            visitor = _SymbolVisitor(module)
            visitor.visit(module.tree)
            symbols.extend(visitor.symbols)
        return symbols


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, module: ParsedModule) -> None:
        self.module = module
        self.symbols: list[Symbol] = []
        self.scope: list[_Scope] = []
        self.lines = module.source.splitlines()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        self._record_symbol(node=node, symbol_type="class")
        self.scope.append(_Scope(name=node.name, kind="class"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        symbol_type = "method" if self.scope and self.scope[-1].kind == "class" else "function"
        self._record_symbol(node=node, symbol_type=symbol_type)
        self.scope.append(_Scope(name=node.name, kind="function"))
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        symbol_type = "method" if self.scope and self.scope[-1].kind == "class" else "function"
        self._record_symbol(node=node, symbol_type=symbol_type)
        self.scope.append(_Scope(name=node.name, kind="function"))
        self.generic_visit(node)
        self.scope.pop()

    def _record_symbol(self, node: ast.AST, symbol_type: str) -> None:
        name = getattr(node, "name", "unknown")
        qualified_parts = [scope.name for scope in self.scope] + [name]
        qualified_name = ".".join(qualified_parts)
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        body = "\n".join(self.lines[start_line - 1 : end_line]).strip()
        signature = self._signature(node)
        docstring = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None
        content_hash = hashlib.sha1(body.encode("utf-8")).hexdigest()
        symbol_id = f"{self.module.relative_path}:{qualified_name}"
        parent = ".".join(qualified_parts[:-1]) or None
        self.symbols.append(
            Symbol(
                id=symbol_id,
                name=qualified_name,
                type=symbol_type,
                file_path=self.module.relative_path,
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                docstring=docstring,
                content_hash=content_hash,
                body=body,
                parent=parent,
            )
        )

    def _signature(self, node: ast.AST) -> str | None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        arguments = [argument.arg for argument in node.args.args]
        if node.args.vararg:
            arguments.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            arguments.append(f"**{node.args.kwarg.arg}")
        return f"{node.name}({', '.join(arguments)})"
