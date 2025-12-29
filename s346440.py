from __future__ import annotations

from typing import List, Tuple
import os
import sys

# Permette import dei file dentro src/ senza aggiungere __init__.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dist_cache import MoveCache
from routing import InstanceData, build_routes_multistart


def _to_output(routes: List[List[int]], gold: List[float]) -> List[Tuple[int, float]]:
    """
    Converte una lista di giri (liste di città, senza 0) nel formato richiesto:
    [(c1,g1), ..., (0,0), (cK,gK), ..., (0,0)]
    """
    out: List[Tuple[int, float]] = []
    visited = set()

    for r in routes:
        load = 0.0
        for c in r:
            if c in visited:
                continue
            visited.add(c)
            load += gold[c]
            out.append((c, load))
        out.append((0, 0.0))

    if not out or out[-1][0] != 0:
        out.append((0, 0.0))
    return out


def solution(problem) -> List[Tuple[int, float]]:
    """
    Richiesta dal prof:
    input: Problem
    output: [(c1,g1), ..., (0,0)]
    """
    G = problem.graph
    alpha = float(problem.alpha)
    beta = float(problem.beta)

    # oro
    gold = [0.0] * G.number_of_nodes()
    for n in G.nodes:
        gold[n] = float(G.nodes[n].get("gold", 0.0))

    # posizioni (per sweep)
    pos = {n: tuple(G.nodes[n]["pos"]) for n in G.nodes}

    data = InstanceData(
        move=MoveCache(G, beta=beta),
        gold=gold,
        alpha=alpha,
        beta=beta,
        pos=pos,
    )

    cities = [n for n in G.nodes if n != 0]

    # multi-start + big-gold + insertion + 2-opt
    routes = build_routes_multistart(cities, data, starts=12, seed=0)

    return _to_output(routes, gold)


# -----------------------
# Test locale (quando runni)
# -----------------------
if __name__ == "__main__":
    # Usa SEMPRE il Problem del prof (generatore ufficiale)
    from Problem import Problem

    # valori “significativi” per fare esperimenti
    ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
    BETAS  = [0.5, 1.0, 1.5, 2.0, 4.0]

    N = 200
    DENSITY = 0.3
    SEED = 42

    for a in ALPHAS:
        for b in BETAS:
            p = Problem(N, density=DENSITY, alpha=a, beta=b, seed=SEED)

            sol = solution(p)

            # Se il Problem del prof ha un evaluator, usalo (nomi possibili)
            cost = None
            if hasattr(p, "evaluate"):
                cost = p.evaluate(sol)
            elif hasattr(p, "cost"):
                try:
                    cost = p.cost(sol)
                except TypeError:
                    cost = None

            base = None
            if hasattr(p, "baseline"):
                try:
                    base = p.baseline()
                except TypeError:
                    base = None

            print(f"N={N} dens={DENSITY} alpha={a} beta={b} | my_cost={cost} | baseline={base} | len={len(sol)}")
