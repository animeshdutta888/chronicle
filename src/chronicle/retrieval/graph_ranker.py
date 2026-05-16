from __future__ import annotations


class GraphRanker:
    def proximity_bonus(self, symbol_id: str, call_graph: dict[str, list[str]], seed_ids: set[str]) -> float:
        if not seed_ids:
            return 0.0
        direct_neighbors = set(call_graph.get(symbol_id, []))
        reverse_neighbors = {
            caller
            for caller, callees in call_graph.items()
            if symbol_id in callees
        }
        if direct_neighbors & seed_ids or reverse_neighbors & seed_ids:
            return 1.5
        if symbol_id in seed_ids:
            return 1.0
        return 0.0
