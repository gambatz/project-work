# src/routing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import random

from dist_cache import MoveCache


@dataclass
class InstanceData:
    move: MoveCache
    gold: List[float]
    alpha: float
    beta: float
    pos: dict[int, tuple[float, float]]


# -------------------------
# Costi
# -------------------------
def route_cost(route: List[int], data: InstanceData) -> float:
    """
    Costo di un giro: 0 -> route -> 0
    route NON contiene lo 0.
    """
    if not route:
        return 0.0

    total = 0.0
    load = 0.0
    prev = 0

    for c in route:
        total += data.move.move_cost(prev, c, load, data.alpha)
        load += data.gold[c]
        prev = c

    total += data.move.move_cost(prev, 0, load, data.alpha)
    return total


def solution_cost(routes: List[List[int]], data: InstanceData) -> float:
    return sum(route_cost(r, data) for r in routes)


# -------------------------
# Sweep ordering
# -------------------------
def sweep_order(cities: List[int], data: InstanceData, angle_offset: float) -> List[int]:
    bx, by = data.pos[0]

    def ang(i: int) -> float:
        x, y = data.pos[i]
        a = math.atan2(y - by, x - bx) + angle_offset
        # normalizzazione per ordine stabile
        if a < -math.pi:
            a += 2 * math.pi
        if a > math.pi:
            a -= 2 * math.pi
        return a

    return sorted(cities, key=ang)


# -------------------------
# Insertion (più veloce)
# -------------------------
def prefix_state(route: List[int], data: InstanceData) -> Tuple[List[float], List[float]]:
    """
    prefix_cost[k] = costo accumulato PRIMA di visitare route[k]
    prefix_load[k] = load PRIMA di visitare route[k]
    lunghezza = len(route)+1; indice len(route) è "dopo l'ultimo nodo"
    """
    n = len(route)
    prefix_cost = [0.0] * (n + 1)
    prefix_load = [0.0] * (n + 1)

    cost = 0.0
    load = 0.0
    prev = 0

    for i, c in enumerate(route):
        prefix_cost[i] = cost
        prefix_load[i] = load
        cost += data.move.move_cost(prev, c, load, data.alpha)
        load += data.gold[c]
        prev = c

    prefix_cost[n] = cost
    prefix_load[n] = load
    return prefix_cost, prefix_load


def cost_from_tail(tail: List[int], start_prev: int, start_load: float, data: InstanceData) -> float:
    """
    Costo aggiuntivo per fare start_prev -> tail -> 0 partendo con start_load.
    """
    cost = 0.0
    load = start_load
    prev = start_prev

    for c in tail:
        cost += data.move.move_cost(prev, c, load, data.alpha)
        load += data.gold[c]
        prev = c

    cost += data.move.move_cost(prev, 0, load, data.alpha)
    return cost


def best_insertion(route: List[int], city: int, data: InstanceData) -> Tuple[List[int], float]:
    """
    Inserisce city nella posizione migliore.
    Usa prefix per non ricalcolare tutta la route per ogni k.
    """
    if not route:
        r = [city]
        return r, route_cost(r, data)

    pref_c, pref_l = prefix_state(route, data)

    best_r = None
    best_cost = float("inf")

    for k in range(len(route) + 1):
        # route[:k] + [city] + route[k:]
        if k == 0:
            prev_node = 0
        else:
            prev_node = route[k - 1]

        start_load = pref_l[k]
        tail = [city] + route[k:]  # city + vecchia coda

        cost_here = pref_c[k] + cost_from_tail(tail, prev_node, start_load, data)
        if cost_here < best_cost:
            best_cost = cost_here
            best_r = route[:k] + [city] + route[k:]

    return best_r, best_cost


# -------------------------
# 2-opt (first improvement)
# -------------------------
def two_opt_first_improvement(route: List[int], data: InstanceData, max_moves: int = 25) -> List[int]:
    """
    2-opt veloce: accetta la prima mossa che migliora, fino a max_moves.
    """
    if len(route) < 4:
        return route[:]

    best = route[:]
    best_cost = route_cost(best, data)
    n = len(best)

    moves = 0
    improved = True

    while improved and moves < max_moves:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                c = route_cost(cand, data)
                if c + 1e-12 < best_cost:
                    best = cand
                    best_cost = c
                    moves += 1
                    improved = True
                    break
            if improved:
                break

    return best


# -------------------------
# Big-gold policy (beta>1 / alpha alto)
# -------------------------
def big_gold_partition(cities: List[int], data: InstanceData) -> Tuple[List[int], List[int]]:
    """
    Se la penalità del carico è alta (beta>1 e/o alpha alto),
    isola una parte delle città con oro alto servendole singolarmente.

    Ritorna (heavy, normal).
    """
    if len(cities) <= 2:
        return [], cities[:]

    pain = max(0.0, data.beta - 1.0) * max(0.0, data.alpha)
    if pain <= 0.25:
        return [], cities[:]

    golds = sorted(data.gold[c] for c in cities)
    # più pain alto => soglia più bassa => più "heavy"
    p = 0.90 - min(0.10, 0.02 * pain)  # ~[0.80..0.90]
    idx = int(p * (len(golds) - 1))
    thr = golds[idx]

    heavy = [c for c in cities if data.gold[c] >= thr]
    heavy_set = set(heavy)
    normal = [c for c in cities if c not in heavy_set]
    return heavy, normal


# -------------------------
# Costruzione routes (merge vs split)
# -------------------------
def build_routes_from_order(ordered: List[int], data: InstanceData) -> List[List[int]]:
    routes: List[List[int]] = []
    current: List[int] = []

    pain = max(0.0, data.beta - 1.0) * max(0.0, data.alpha)
    conserv = 1.0 + 0.20 * pain  # più alto => meno merge

    for city in ordered:
        if not current:
            current = [city]
            continue

        curr_cost = route_cost(current, data)
        solo_cost = route_cost([city], data)

        inserted, ins_cost = best_insertion(current, city, data)
        split_cost = curr_cost + solo_cost

        if ins_cost <= split_cost / conserv:
            current = inserted
        else:
            routes.append(current)
            current = [city]

    if current:
        routes.append(current)

    return routes


# -------------------------
# Multi-start (random restarts)
# -------------------------
def build_routes_multistart(
    cities: List[int],
    data: InstanceData,
    starts: int = 12,
    seed: int = 0
) -> List[List[int]]:
    rng = random.Random(seed)

    heavy, normal = big_gold_partition(cities, data)
    heavy_routes = [[c] for c in heavy]

    # adattamento per N grandi
    n = len(normal)
    if n > 2000:
        starts = min(starts, 4)
    elif n > 1000:
        starts = min(starts, 6)

    best_routes: Optional[List[List[int]]] = None
    best_cost = float("inf")

    for t in range(max(1, starts)):
        angle_offset = (2 * math.pi) * (t / max(1, starts))
        ordered = sweep_order(normal, data, angle_offset)

        # piccola perturbazione (coerente con iterated local search)
        if n >= 20:
            k = max(1, n // 50)  # ~2% swap
            for _ in range(k):
                i = rng.randrange(n)
                j = rng.randrange(n)
                ordered[i], ordered[j] = ordered[j], ordered[i]

        routes = build_routes_from_order(ordered, data)

        # local search su ogni route
        routes = [two_opt_first_improvement(r, data, max_moves=25) for r in routes]

        # aggiungi heavy routes
        routes = routes + heavy_routes

        c = solution_cost(routes, data)
        if c < best_cost:
            best_cost = c
            best_routes = routes

    return best_routes if best_routes is not None else heavy_routes
