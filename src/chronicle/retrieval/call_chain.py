from __future__ import annotations

from collections import deque
import re

from ..core.models import CallChainReport, CallChainStep, IndexSnapshot, Symbol


class CallChainBuilder:
    def build(
        self,
        *,
        query: str,
        snapshot: IndexSnapshot,
        selected_symbols: list[Symbol],
        max_depth: int = 4,
        max_chains: int = 4,
        preferred_terms: set[str] | None = None,
    ) -> CallChainReport | None:
        if not selected_symbols:
            return None
        symbol_index = {symbol.id: symbol for symbol in snapshot.symbols}
        entry = self._entry_symbol(selected_symbols=selected_symbols, preferred_terms=preferred_terms or set())
        chain_ids = self._chains(
            start_id=entry.id,
            call_graph=snapshot.call_graph,
            max_depth=max_depth,
            max_chains=max_chains,
        )
        chains: list[list[CallChainStep]] = []
        for chain in chain_ids:
            steps: list[CallChainStep] = []
            for symbol_id in chain:
                symbol = symbol_index.get(symbol_id)
                if symbol is None:
                    continue
                steps.append(
                    CallChainStep(
                        symbol_id=symbol.id,
                        name=symbol.name,
                        file_path=symbol.file_path,
                        start_line=symbol.start_line,
                    )
                )
            if steps:
                chains.append(steps)
        if not chains:
            return None
        return CallChainReport(
            query=query,
            entry_symbol=entry.name,
            max_depth=max_depth,
            chains=chains,
            mermaid=self._mermaid(chains),
            summary=self._summary(
                chains=chains,
                direct_calls=self._direct_calls(entry=entry, call_graph=snapshot.call_graph, symbol_index=symbol_index),
            ),
        )

    def _entry_symbol(self, *, selected_symbols: list[Symbol], preferred_terms: set[str]) -> Symbol:
        if not preferred_terms:
            return selected_symbols[0]

        def score(symbol: Symbol) -> tuple[int, int, int]:
            normalized_name = self._normalize(symbol.name)
            leaf = self._normalize(symbol.name.split(".")[-1])
            file_path = self._normalize(symbol.file_path)
            exact_leaf = any(term == leaf for term in preferred_terms)
            exact_name = any(term == normalized_name for term in preferred_terms)
            contained = sum(1 for term in preferred_terms if term and (term in normalized_name or term in file_path))
            callable_bonus = 1 if symbol.type in {"function", "method"} else 0
            return (
                8 if exact_name else 0,
                6 if exact_leaf else 0,
                contained + callable_bonus,
            )

        best = max(selected_symbols, key=score)
        return best if score(best) > (0, 0, 0) else selected_symbols[0]

    def _chains(
        self,
        *,
        start_id: str,
        call_graph: dict[str, list[str]],
        max_depth: int,
        max_chains: int,
    ) -> list[list[str]]:
        discovered: list[list[str]] = []
        queue: deque[list[str]] = deque([[start_id]])
        seen_paths: set[tuple[str, ...]] = set()

        while queue and len(discovered) < max_chains:
            path = queue.popleft()
            path_key = tuple(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            current = path[-1]
            neighbors = [neighbor for neighbor in call_graph.get(current, []) if neighbor in call_graph]
            if not neighbors or len(path) >= max_depth:
                discovered.append(path)
                continue
            expanded = False
            for neighbor in neighbors[:3]:
                if neighbor in path:
                    continue
                queue.append(path + [neighbor])
                expanded = True
            if not expanded:
                discovered.append(path)
        if not discovered:
            discovered.append([start_id])
        return discovered[:max_chains]

    def _direct_calls(
        self,
        *,
        entry: Symbol,
        call_graph: dict[str, list[str]],
        symbol_index: dict[str, Symbol],
    ) -> list[Symbol]:
        generic = {"str", "dict", "list", "set", "len", "print", "now", "strftime", "isoformat", "write_text", "mkdir"}
        direct: list[Symbol] = []
        for symbol_id in call_graph.get(entry.id, []):
            symbol = symbol_index.get(symbol_id)
            if symbol is None:
                continue
            leaf = symbol.name.split(".")[-1]
            if leaf in generic:
                continue
            if symbol.file_path != entry.file_path and leaf in {"get", "dumps", "loads"}:
                continue
            if symbol not in direct:
                direct.append(symbol)
            if len(direct) >= 8:
                break
        return direct

    def _summary(self, *, chains: list[list[CallChainStep]], direct_calls: list[Symbol]) -> str:
        rendered = []
        if chains and direct_calls:
            rendered.append(
                "Direct calls from "
                + chains[0][0].name
                + ": "
                + ", ".join(f"{symbol.name} ({symbol.file_path}:{symbol.start_line})" for symbol in direct_calls)
            )
        for chain in chains:
            rendered.append(" -> ".join(f"{step.name} ({step.file_path}:{step.start_line})" for step in chain))
        return "\n".join(f"- {line}" for line in rendered)

    def _mermaid(self, chains: list[list[CallChainStep]]) -> str:
        lines = ["flowchart TD"]
        emitted_edges: set[tuple[str, str]] = set()
        for chain in chains:
            for left, right in zip(chain, chain[1:]):
                edge = (left.symbol_id, right.symbol_id)
                if edge in emitted_edges:
                    continue
                emitted_edges.add(edge)
                left_id = self._node_id(left.symbol_id)
                right_id = self._node_id(right.symbol_id)
                lines.append(f'  {left_id}["{left.name}"] --> {right_id}["{right.name}"]')
        return "\n".join(lines)

    def _node_id(self, symbol_id: str) -> str:
        return "node_" + "".join(character if character.isalnum() else "_" for character in symbol_id)

    def _normalize(self, text: str) -> str:
        return "".join(part.lower() for part in re.findall(r"[A-Za-z0-9]+", text))
