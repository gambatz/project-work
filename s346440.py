from Problem import Problem
import networkx as nx


def solution(p: Problem):
    graph = p.graph
    alpha = p.alpha
    beta = p.beta
    num_nodes = len(graph.nodes())
    
    remaining_gold = {}
    for node in graph.nodes():
        remaining_gold[node] = graph.nodes[node]['gold']
    
    all_paths = {}
    all_dists = {}
    for source in graph.nodes():
        all_paths[source] = nx.single_source_dijkstra_path(graph, source, weight='dist')
        all_dists[source] = nx.single_source_dijkstra_path_length(graph, source, weight='dist')
    
    def edge_cost(dist, carried_gold):
        if carried_gold == 0:
            return dist
        return dist + (alpha * dist * carried_gold) ** beta
    
    def path_cost(node_path, carried_gold):
        if len(node_path) <= 1:
            return 0.0
        total_cost = 0.0
        for i in range(len(node_path) - 1):
            a, b = node_path[i], node_path[i + 1]
            if graph.has_edge(a, b):
                dist = graph[a][b]['dist']
            else:
                dist = all_dists[a][b]
            total_cost += edge_cost(dist, carried_gold)
        return total_cost
    
    path = []
    current_position = 0
    carried_gold = 0.0
    
    def move_along(node_path):
        nonlocal current_position
        for i in range(1, len(node_path)):
            city = node_path[i]
            path.append((city, 0))
            current_position = city
    
    def go_to_base():
        nonlocal current_position, carried_gold
        if current_position != 0:
            base_path = all_paths[current_position][0]
            move_along(base_path)
        carried_gold = 0.0
    
    def calculate_optimal_load_cap(distance_from_base, beta_val, alpha_val):
        if beta_val <= 1.0:
            return float('inf')
        else:
            denominator = (beta_val - 1) * alpha_val * beta_val * (distance_from_base ** (beta_val - 1))
            if denominator <= 0:
                return float('inf')
            w_star = (2.0 / denominator) ** (1.0 / beta_val)
            return max(1.0, w_star)
    
    max_gold = max(remaining_gold[i] for i in remaining_gold if i != 0) if num_nodes > 1 else 0
    
    if beta >= 1.5:
        max_weight_per_trip = max(200.0, max_gold * 1.2)
        heavy_gold_penalty_multiplier = 0.02
    elif beta >= 1.2:
        max_weight_per_trip = max(300.0, max_gold * 1.5)
        heavy_gold_penalty_multiplier = 0.015
    else:
        max_weight_per_trip = max(500.0, max_gold * 2.0)
        heavy_gold_penalty_multiplier = 0.01
    
    def filter_candidates(current_pos, remaining_gold_dict):
        K = 6
        candidates = set()
        cities_with_gold = [c for c in remaining_gold_dict if c != 0 and remaining_gold_dict[c] > 0]
        if not cities_with_gold:
            return []
        distances_from_current = [(c, all_dists[current_pos][c]) for c in cities_with_gold]
        distances_from_current.sort(key=lambda x: x[1])
        k_nearest = [c for c, _ in distances_from_current[:K]]
        candidates.update(k_nearest)
        for city in cities_with_gold:
            dist_current_to_city = all_dists[current_pos][city]
            dist_base_to_city = all_dists[0][city]
            if dist_current_to_city <= 0.8 * dist_base_to_city:
                candidates.add(city)
        return list(candidates)
    
    while any(remaining_gold[c] > 0 for c in remaining_gold if c != 0):
        if current_position != 0:
            go_to_base()
        carried_gold = 0.0
        collections_in_trip = 0
        
        while True:
            candidates = filter_candidates(current_position, remaining_gold)
            if not candidates:
                break
            
            best_city = None
            best_score = float('inf')
            best_delta = float('inf')
            best_take_amount = 0
            
            for city in candidates:
                gold_at_city = remaining_gold[city]
                if gold_at_city <= 0:
                    continue
                
                dist_to_base = all_dists[city][0]
                optimal_load_cap = calculate_optimal_load_cap(dist_to_base, beta, alpha)
                take_amount = min(gold_at_city, max(1.0, optimal_load_cap))
                
                if carried_gold + take_amount > max_weight_per_trip:
                    take_amount = max(0, max_weight_per_trip - carried_gold)
                    if take_amount < 1.0:
                        continue
                
                path_to_base_from_current = all_paths[current_position][0]
                cost_return_now = path_cost(path_to_base_from_current, carried_gold)
                
                path_base_to_city = all_paths[0][city]
                cost_base_to_city = path_cost(path_base_to_city, 0)
                
                path_city_to_base = all_paths[city][0]
                cost_city_to_base = path_cost(path_city_to_base, take_amount)
                
                option_a = cost_return_now + cost_base_to_city + cost_city_to_base
                
                path_current_to_city = all_paths[current_position][city]
                cost_to_city = path_cost(path_current_to_city, carried_gold)
                
                path_city_to_base_after = all_paths[city][0]
                cost_return_from_city = path_cost(path_city_to_base_after, carried_gold + take_amount)
                
                option_b = cost_to_city + cost_return_from_city
                
                delta = option_b - option_a
                penalty = heavy_gold_penalty_multiplier * gold_at_city * dist_to_base
                score = delta + penalty
                
                if score < best_score:
                    best_score = score
                    best_delta = delta
                    best_city = city
                    best_take_amount = take_amount
            
            if best_city is not None and best_delta < 0:
                city_path = all_paths[current_position][best_city]
                if len(city_path) > 1:
                    for i in range(1, len(city_path) - 1):
                        path.append((city_path[i], 0))
                        current_position = city_path[i]
                
                path.append((best_city, best_take_amount))
                current_position = best_city
                carried_gold += best_take_amount
                remaining_gold[best_city] -= best_take_amount
                collections_in_trip += 1
                
            elif current_position == 0 and collections_in_trip == 0:
                if candidates:
                    nearest_city = min(candidates, key=lambda c: all_dists[0][c])
                    gold_at_nearest = remaining_gold[nearest_city]
                    
                    dist_to_base = all_dists[nearest_city][0]
                    optimal_load_cap = calculate_optimal_load_cap(dist_to_base, beta, alpha)
                    take_amount = min(gold_at_nearest, max(1.0, optimal_load_cap))
                    
                    city_path = all_paths[current_position][nearest_city]
                    
                    if len(city_path) > 1:
                        for i in range(1, len(city_path) - 1):
                            path.append((city_path[i], 0))
                            current_position = city_path[i]
                    
                    path.append((nearest_city, take_amount))
                    current_position = nearest_city
                    carried_gold += take_amount
                    remaining_gold[nearest_city] -= take_amount
                    collections_in_trip += 1
                else:
                    break
            else:
                break
    
    if current_position != 0:
        go_to_base()
    
    if not path or path[-1] != (0, 0):
        path.append((0, 0))
    
    validated_path = _validate_and_fix_path(path, graph, all_paths)
    return validated_path


def _validate_and_fix_path(path, graph, all_paths):
    if not path:
        return [(0, 0)]
    validated = []
    prev_city = 0
    for city, gold in path:
        if city == prev_city:
            if gold > 0:
                if validated and validated[-1][0] == city:
                    old_city, old_gold = validated[-1]
                    validated[-1] = (old_city, old_gold + gold)
                else:
                    validated.append((city, gold))
            continue
        if graph.has_edge(prev_city, city):
            validated.append((city, gold))
        else:
            if city in all_paths[prev_city]:
                intermediate_path = all_paths[prev_city][city]
                for i in range(1, len(intermediate_path) - 1):
                    validated.append((intermediate_path[i], 0))
                validated.append((city, gold))
            else:
                validated.append((city, gold))
        
        prev_city = city
    if not validated or validated[-1][0] != 0:
        if prev_city != 0 and 0 in all_paths[prev_city]:
            return_path = all_paths[prev_city][0]
            for i in range(1, len(return_path)):
                validated.append((return_path[i], 0))
        if not validated or validated[-1] != (0, 0):
            validated.append((0, 0))
    
    if validated[-1] != (0, 0):
        if validated[-1][0] == 0:
            validated[-1] = (0, 0)
        else:
            validated.append((0, 0))
    
    return validated


def is_valid_path(problem, path):
    if not path:
        return False
    graph = problem.graph
    prev_city = 0
    for city, gold in path:
        if city != prev_city:
            if not graph.has_edge(prev_city, city):
                return False
        prev_city = city
    return True


def evaluate_solution(problem, sol):
    graph = problem.graph
    alpha = float(problem.alpha)
    beta = float(problem.beta)

    visited = set([0])
    load = 0.0
    total = 0.0
    cur = 0

    def edge_cost(d, cur_load):
        return d + (alpha * d * cur_load) ** beta

    for city, gold_pickup in sol:
        if cur != city:
            path = nx.dijkstra_path(graph, cur, city, weight="dist")
            for a, b in zip(path, path[1:]):
                d = float(graph[a][b]["dist"])
                total += edge_cost(d, load)
        cur = city
        if city == 0:
            load = 0.0
        else:
            load += gold_pickup

    return total


if __name__ == "__main__":
    import time
    
    SEED = 42
    
    TEST_CASES = [
        (100, 1.0, 0.5, 0.2),
        (100, 1.0, 1.0, 0.2),
        (100, 1.0, 0.5, 1.0),
        (100, 1.0, 1.0, 1.0),
        (200, 1.0, 0.5, 0.2),
        (200, 1.0, 1.0, 0.2),
        (200, 1.0, 0.5, 1.0),
        (200, 1.0, 1.0, 1.0),
        (500, 1.0, 0.5, 0.2),
        (500, 1.0, 1.0, 0.2),
    ]
    
    header = " N  |Alp|Bet|Den|   Baseline |  Mio Costo |      Delta|Delta %|Time(s)"
    sep = "=" * len(header)
    
    print(header)
    print(sep)
    
    for (N, alpha, beta, density) in TEST_CASES:
        p = Problem(N, density=density, alpha=alpha, beta=beta, seed=SEED)
        
        t0 = time.perf_counter()
        sol = solution(p)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        
        baseline = p.baseline()
        my_cost = evaluate_solution(p, sol)
        delta = baseline - my_cost
        delta_pct = (delta / baseline) * 100.0 if baseline > 0 else 0.0
        
        delta_str = f"{delta_pct:+.2f}%"
        
        print(f"{N:>3} |{alpha:.1f}|{beta:.1f}|{density:.1f}|{baseline:>11.2f} |{my_cost:>11.2f} |{delta:>10.2f}|{delta_str:>7}|{elapsed:.4f}")