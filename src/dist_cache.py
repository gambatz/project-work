from __future__ import annotations

from dataclasses import dataclass, field
import networkx as nx


@dataclass
class MoveCache:
    
    graph: nx.Graph
    beta: float
    _cache: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)

    def _stats(self, u: int, v: int) -> tuple[float, float]:
        if u == v:
            return 0.0, 0.0

        key = (u, v)
        if key in self._cache:
            return self._cache[key]

        
        if self.graph.has_edge(u, v):
            d = float(self.graph[u][v]["dist"])
            s1 = d
            sb = d ** self.beta
            
            self._cache[(u, v)] = (s1, sb)
            self._cache[(v, u)] = (s1, sb)
            return s1, sb

        
        path = nx.dijkstra_path(self.graph, u, v, weight="dist")

        s1 = 0.0
        sb = 0.0
        for a, b in zip(path, path[1:]):
            d = float(self.graph[a][b]["dist"])
            s1 += d
            sb += d ** self.beta

        self._cache[(u, v)] = (s1, sb)
        self._cache[(v, u)] = (s1, sb)
        return s1, sb

    def move_cost(self, u: int, v: int, load: float, alpha: float) -> float:
       
        s1, sb = self._stats(u, v)
        if load <= 0.0:
            return s1
        return s1 + ((alpha * load) ** self.beta) * sb