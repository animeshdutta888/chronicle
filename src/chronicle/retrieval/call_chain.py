from __future__ import annotations

from collections import deque

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
    ) -> CallChainReport | None:
        if not selected_symbols:
            return None
        symbol_index = {symbol.id: symbol for symbol in snapshot.symbols}
        entry = selected_symbols[0]
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
            summary=self._summary(chains),
        )

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

    def _summary(self, chains: list[list[CallChainStep]]) -> str:
        rendered = []
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
