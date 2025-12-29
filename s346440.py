# s123456.py
from __future__ import annotations

from typing import List, Tuple
import os
import sys

# Permette import dei file dentro src/ senza aggiungere __init__.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.dist_cache import MoveCache
from src.routing import InstanceData, build_routes_multistart


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
# Evaluator locale per test
# -----------------------
def evaluate_solution(problem, sol: List[Tuple[int, float]]) -> float:
    """
    Valuta una soluzione nel formato [(city, load_after_pickup), ...]
    coerentemente con le regole:
    - spostamenti tra due city consecutive: shortest path su weight='dist'
    - costo piecewise sugli edge del path:
        d + (alpha * d * load)^beta
    - raccogli oro solo se city non è mai stata visitata prima (city != 0)
    - quando arrivi a 0: scarichi (load=0)
    """
    import networkx as nx

    G = problem.graph
    alpha = float(problem.alpha)
    beta = float(problem.beta)

    visited = set([0])
    load = 0.0
    total = 0.0
    cur = 0

    def edge_cost(d: float, cur_load: float) -> float:
        return d + (alpha * d * cur_load) ** beta

    for city, _reported_load in sol:
        # 1) spostamento cur -> city tramite shortest path
        if cur != city:
            path = nx.dijkstra_path(G, cur, city, weight="dist")
            for a, b in zip(path, path[1:]):
                d = float(G[a][b]["dist"])
                total += edge_cost(d, load)

        cur = city

        # 2) arrivo: aggiorna carico
        if city == 0:
            load = 0.0
        else:
            if city not in visited:
                load += float(G.nodes[city]["gold"])
                visited.add(city)

    return total


# -----------------------
# Test locale (quando runni)
# -----------------------
if __name__ == "__main__":
    # Usa SEMPRE il Problem del prof (generatore ufficiale)
    from Problem import Problem

    # valori “significativi” per fare esperimenti
    ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
    BETAS  = [0.5, 1.0, 1.5, 2.0, 4.0]

    # Parametri di generazione per i test (puoi modificarli a piacere)
    N = 200
    DENSITY = 0.3
    SEED = 42

    print("----- BASELINE sanity checks (alcuni esempi) -----")
    print("Problem(100, density=0.2, alpha=1, beta=1).baseline():", Problem(100, density=0.2, alpha=1, beta=1, seed=SEED).baseline())
    print("Problem(100, density=1.0, alpha=1, beta=1).baseline():", Problem(100, density=1.0, alpha=1, beta=1, seed=SEED).baseline())
    print()

    print("----- GRID TEST (my_cost vs baseline) -----")
    for a in ALPHAS:
        for b in BETAS:
            p = Problem(N, density=DENSITY, alpha=a, beta=b, seed=SEED)

            sol = solution(p)
            my_cost = evaluate_solution(p, sol)
            baseline_cost = p.baseline()

            ratio = baseline_cost / my_cost if my_cost > 0 else float("inf")

            # info utile: quante volte torni a base (approssimato contando gli zeri)
            num_returns = sum(1 for c, _ in sol if c == 0)

            print(
                f"N={N} dens={DENSITY} alpha={a} beta={b} | "
                f"my_cost={my_cost:.6g} | baseline={baseline_cost:.6g} | "
                f"baseline/my={ratio:.4f} | len={len(sol)} | returns={num_returns}"
            )
