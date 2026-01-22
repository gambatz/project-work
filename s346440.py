from __future__ import annotations

from typing import List, Tuple
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dist_cache import MoveCache
from routing import InstanceData, build_routes_multistart


def _to_output(routes: List[List[int]], gold: List[float]) -> List[Tuple[int, float]]:
 
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
    G = problem.graph
    alpha = float(problem.alpha)
    beta = float(problem.beta)

    gold = [0.0] * G.number_of_nodes()
    for n in G.nodes:
        gold[n] = float(G.nodes[n].get("gold", 0.0))

    pos = {n: tuple(G.nodes[n]["pos"]) for n in G.nodes}
    total_gold = sum(gold) - gold[0]

    data = InstanceData(
        move=MoveCache(G, beta=beta),
        gold=gold,
        alpha=alpha,
        beta=beta,
        pos=pos,
        total_gold=total_gold,
    )

    cities = [n for n in G.nodes if n != 0]
    routes = build_routes_multistart(cities, data, starts=12, seed=0)
    return _to_output(routes, gold)


def evaluate_solution(problem, sol: List[Tuple[int, float]]) -> float:

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
        if cur != city:
            path = nx.dijkstra_path(G, cur, city, weight="dist")
            for a, b in zip(path, path[1:]):
                d = float(G[a][b]["dist"])
                total += edge_cost(d, load)

        cur = city

        if city == 0:
            load = 0.0
        else:
            if city not in visited:
                load += float(G.nodes[city]["gold"])
                visited.add(city)

    return total


def print_header() -> None:
    title = "TEST (tabella stile collega - versione ridotta)"
    print("\n" + title)
    print("-" * 120)
    header = f"{'N':>5} | {'Alp':>4} | {'Bet':>4} | {'Den':>4} | {'Baseline':>12} | {'Mio Costo':>12} | {'Delta':>12} | {'Delta %':>8} | {'Time(s)':>8} | {'Ret':>4}"
    print(header)
    print("-" * 120)


def print_row(r: dict) -> None:
    print(
        f"{r['N']:>5} | {r['alpha']:>4.1f} | {r['beta']:>4.1f} | {r['density']:>4.1f} | "
        f"{r['baseline']:>12.2f} | {r['my_cost']:>12.2f} | {r['delta']:>12.2f} | "
        f"{r['delta_pct']:>7.2f}% | {r['time_s']:>8.3f} | {r['returns']:>4}"
    )


if __name__ == "__main__":
    from Problem import Problem

    SEED = 42


    TESTS = []

    for N in [100, 200, 500]:
        for den in [0.2, 1.0]:
            for beta in [0.5, 1.0]:
                TESTS.append((N, den, 1.0, beta))

    for N in [100, 200, 500]:
        den = 0.3
        for alpha in [0.5, 2.0]:
            for beta in [1.5, 2.0, 4.0]:
                TESTS.append((N, den, alpha, beta))


    print_header()

    for (N, den, alpha, beta) in TESTS:
        p = Problem(N, density=den, alpha=alpha, beta=beta, seed=SEED)

        t0 = time.perf_counter()
        sol = solution(p)
        my_cost = evaluate_solution(p, sol)
        t1 = time.perf_counter()

        baseline = p.baseline()
        delta = baseline - my_cost
        delta_pct = (delta / baseline) * 100.0 if baseline != 0 else 0.0
        returns = sum(1 for c, _ in sol if c == 0)

        row = {
            "N": N,
            "density": den,
            "alpha": alpha,
            "beta": beta,
            "baseline": float(baseline),
            "my_cost": float(my_cost),
            "delta": float(delta),
            "delta_pct": float(delta_pct),
            "time_s": float(t1 - t0),
            "returns": int(returns),
        }
        print_row(row)

    print("-" * 120)