# src/dist_cache.py
from __future__ import annotations

from dataclasses import dataclass, field
import networkx as nx


@dataclass
class MoveCache:
    """
    Cache per calcolare velocemente il costo di spostamento u->v con carico 'load'
    rispettando la definizione piecewise sul cammino più corto (weight='dist').

    Precalcolo per ogni coppia (u,v):
      S1 = sum(dist_e)
      Sb = sum(dist_e ** beta)

    Costo per carico load:
      cost = S1 + (alpha * load) ** beta * Sb
    """
    graph: nx.Graph
    beta: float
    _cache: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)

    def _stats(self, u: int, v: int) -> tuple[float, float]:
        if u == v:
            return 0.0, 0.0

        key = (u, v)
        if key in self._cache:
            return self._cache[key]

        # Fast path: arco diretto
        if self.graph.has_edge(u, v):
            d = float(self.graph[u][v]["dist"])
            s1 = d
            sb = d ** self.beta
            self._cache[key] = (s1, sb)
            return s1, sb

        # Cammino più corto secondo la metrica corretta: weight='dist'
        path = nx.dijkstra_path(self.graph, u, v, weight="dist")

        s1 = 0.0
        sb = 0.0
        for a, b in zip(path, path[1:]):
            d = float(self.graph[a][b]["dist"])
            s1 += d
            sb += d ** self.beta

        self._cache[key] = (s1, sb)
        return s1, sb

    def move_cost(self, u: int, v: int, load: float, alpha: float) -> float:
        """
        Costo di spostamento u->v portando 'load' oro.
        """
        s1, sb = self._stats(u, v)
        if load <= 0.0:
            return s1
        return s1 + ((alpha * load) ** self.beta) * sb
