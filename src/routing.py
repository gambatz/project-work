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
    total_gold: float


def sp_dist(u: int, v: int, data: InstanceData) -> float:

    return data.move._stats(u, v)[0]  


def route_cost(route: List[int], data: InstanceData) -> float:
   
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


def solo_city_cost(city: int, data: InstanceData) -> float:
   
    g = data.gold[city]
    return data.move.move_cost(0, city, 0.0, data.alpha) + data.move.move_cost(city, 0, g, data.alpha)



def two_opt_first_improvement(route: List[int], data: InstanceData, max_moves: int) -> List[int]:
    if len(route) < 4 or max_moves <= 0:
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


def two_opt_policy(route_len: int) -> int:
 
    if route_len <= 25:
        return 20
    if route_len <= 45:
        return 10
    if route_len <= 70:
        return 4
    return 0


def sweep_order(cities: List[int], data: InstanceData, angle_offset: float) -> List[int]:
    bx, by = data.pos[0]

    def ang(i: int) -> float:
        x, y = data.pos[i]
        a = math.atan2(y - by, x - bx) + angle_offset
        if a < -math.pi:
            a += 2 * math.pi
        if a > math.pi:
            a -= 2 * math.pi
        return a

    return sorted(cities, key=ang)


def prefix_state(route: List[int], data: InstanceData) -> Tuple[List[float], List[float]]:
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
    if not route:
        r = [city]
        return r, route_cost(r, data)

    pref_c, pref_l = prefix_state(route, data)

    best_r: Optional[List[int]] = None
    best_cost = float("inf")

    for k in range(len(route) + 1):
        prev_node = 0 if k == 0 else route[k - 1]
        start_load = pref_l[k]
        tail = [city] + route[k:]
        cost_here = pref_c[k] + cost_from_tail(tail, prev_node, start_load, data)
        if cost_here < best_cost:
            best_cost = cost_here
            best_r = route[:k] + [city] + route[k:]

    return best_r if best_r is not None else route + [city], best_cost



def heavy_fraction(alpha: float, beta: float, n: int) -> float:
    if n < 20 or beta <= 1.25:
        return 0.0

    pain = max(0.0, beta - 1.0) * max(0.0, alpha)

    if beta >= 2.0:
        frac = 0.05 + min(0.05, 0.01 * pain)
        return min(0.10, max(0.05, frac))

    if beta >= 1.45:
        frac = 0.02 + min(0.03, 0.01 * pain)
        return min(0.05, max(0.02, frac))

    frac = 0.01 + min(0.01, 0.005 * pain)
    return min(0.02, max(0.0, frac))


def big_gold_partition(cities: List[int], data: InstanceData) -> Tuple[List[int], List[int]]:
    frac = heavy_fraction(data.alpha, data.beta, len(cities))
    if frac <= 0.0:
        return [], cities[:]
    k = max(1, int(round(frac * len(cities))))
    sorted_c = sorted(cities, key=lambda c: data.gold[c], reverse=True)
    heavy = sorted_c[:k]
    heavy_set = set(heavy)
    normal = [c for c in cities if c not in heavy_set]
    return heavy, normal


def compute_conserv(data: InstanceData) -> float:
    pain = max(0.0, data.beta - 1.0) * max(0.0, data.alpha)
    conserv = 1.0 + 0.20 * pain
    return max(1.0, conserv)


def build_routes_from_order_beta_ge_1(ordered: List[int], data: InstanceData) -> List[List[int]]:
    routes: List[List[int]] = []
    current: List[int] = []
    conserv = compute_conserv(data)

    for city in ordered:
        if not current:
            current = [city]
            continue

        curr_cost = route_cost(current, data)
        solo_cost = solo_city_cost(city, data)

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


def nn_tour(cities: List[int], data: InstanceData, start_city: int) -> List[int]:
    unvisited = set(cities)
    tour: List[int] = []
    cur = start_city
    tour.append(cur)
    unvisited.remove(cur)

    while unvisited:
        nxt = min(unvisited, key=lambda v: sp_dist(cur, v, data))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    return tour


def tour_length(tour: List[int], data: InstanceData) -> float:
    if not tour:
        return 0.0
    total = sp_dist(0, tour[0], data)
    for a, b in zip(tour, tour[1:]):
        total += sp_dist(a, b, data)
    total += sp_dist(tour[-1], 0, data)
    return total


def two_opt_tour_limited(tour: List[int], data: InstanceData, max_moves: int, sample_pairs: int, rng: random.Random) -> List[int]:
    
    if len(tour) < 6 or max_moves <= 0:
        return tour[:]

    best = tour[:]
    best_len = tour_length(best, data)
    n = len(best)

    moves = 0
    while moves < max_moves:
        improved = False
       
        for _ in range(sample_pairs):
            i = rng.randrange(0, n - 2)
            j = rng.randrange(i + 2, n)
            cand = best[:i] + list(reversed(best[i:j])) + best[j:]
            L = tour_length(cand, data)
            if L + 1e-12 < best_len:
                best = cand
                best_len = L
                moves += 1
                improved = True
                break
        if not improved:
            break

    return best


def split_tour_greedy(tour: List[int], data: InstanceData) -> List[List[int]]:

    routes: List[List[int]] = []
    current: List[int] = []
    cost_so_far = 0.0   
    load = 0.0
    prev = 0


    n = len(tour)
    max_len = max(25, min(70, int(0.30 * n) + 15))
    
    max_load = max(0.18 * data.total_gold, min(0.45 * data.total_gold, (0.35 / (1.0 + 0.03 * data.alpha)) * data.total_gold))

    for city in tour:
       
        add_edge = data.move.move_cost(prev, city, load, data.alpha)
        new_load = load + data.gold[city]
        cost_close_after_add = (cost_so_far + add_edge) + data.move.move_cost(city, 0, new_load, data.alpha)

        
        cost_close_now = cost_so_far + data.move.move_cost(prev, 0, load, data.alpha)
        cost_split = cost_close_now + solo_city_cost(city, data)

        
        force_split = (len(current) >= max_len) or (load >= max_load)

        if (not current) or (cost_close_after_add <= cost_split and not force_split):
            
            current.append(city)
            cost_so_far += add_edge
            load = new_load
            prev = city
        else:
            
            routes.append(current)
           
            current = [city]
            cost_so_far = data.move.move_cost(0, city, 0.0, data.alpha)
            load = data.gold[city]
            prev = city

    if current:
        routes.append(current)

    return routes


def build_routes_beta_lt_1(cities: List[int], data: InstanceData, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    n = len(cities)
    if n == 0:
        return []

   
    starts = 6 if n <= 300 else 3
    cities_sorted = sorted(cities, key=lambda c: sp_dist(0, c, data), reverse=True)
    start_candidates = cities_sorted[: min(3, n)] + rng.sample(cities, k=min(starts, n))

    best_tour: Optional[List[int]] = None
    best_len = float("inf")

    for s in start_candidates:
        tour = nn_tour(cities, data, start_city=s)
        L = tour_length(tour, data)
        if L < best_len:
            best_len = L
            best_tour = tour

    tour = best_tour if best_tour is not None else nn_tour(cities, data, start_city=cities[0])

 
    if n <= 200:
        tour = two_opt_tour_limited(tour, data, max_moves=30, sample_pairs=80, rng=rng)
    elif n <= 500:
        tour = two_opt_tour_limited(tour, data, max_moves=18, sample_pairs=60, rng=rng)
    else:
        tour = two_opt_tour_limited(tour, data, max_moves=10, sample_pairs=40, rng=rng)

    routes = split_tour_greedy(tour, data)

    improved_routes = []
    for r in routes:
        mm = two_opt_policy(len(r))
        improved_routes.append(two_opt_first_improvement(r, data, max_moves=mm))
    return improved_routes


def auto_starts_beta_ge_1(n: int, alpha: float, beta: float, default: int) -> int:
    starts = default
    if n > 2000:
        return min(starts, 4)
    if n > 1000:
        return min(starts, 6)
    if n <= 300:
        if 0.95 <= beta <= 1.8:
            return max(starts, 18)
        if beta >= 3.0:
            return min(starts, 10)
    return starts


def build_routes_multistart(cities: List[int], data: InstanceData, starts: int = 12, seed: int = 0) -> List[List[int]]:
    
    if data.beta < 1.0:
        
        return build_routes_beta_lt_1(cities, data, seed=seed)

    rng = random.Random(seed)

    
    heavy, normal = big_gold_partition(cities, data)
    heavy_routes = [[c] for c in heavy]

    n = len(normal)
    starts = auto_starts_beta_ge_1(n, data.alpha, data.beta, starts)

    best_routes: Optional[List[List[int]]] = None
    best_cost = float("inf")

    for t in range(max(1, starts)):
        angle_offset = (2 * math.pi) * (t / max(1, starts))
        ordered = sweep_order(normal, data, angle_offset)

     
        if n >= 20:
            k = max(1, n // 50)
            for _ in range(k):
                i = rng.randrange(n)
                j = rng.randrange(n)
                ordered[i], ordered[j] = ordered[j], ordered[i]

        routes = build_routes_from_order_beta_ge_1(ordered, data)

        improved_routes = []
        for r in routes:
            mm = two_opt_policy(len(r))
            improved_routes.append(two_opt_first_improvement(r, data, max_moves=mm))
        routes = improved_routes + heavy_routes

        c = solution_cost(routes, data)
        if c < best_cost:
            best_cost = c
            best_routes = routes

    return best_routes if best_routes is not None else heavy_routes