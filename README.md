Gold collection on a graph with load-dependent travel cost
1. Problem description
The problem involves N cities placed in the unit square ((0,1)\times(0,1)), with a special node 0 (the depot/base) fixed at (0.5,0.5). Each city (i>0) contains a non-negative amount of gold (g_i \ge 0). Cities are connected by an undirected graph with a controllable edge density, and each edge ((i,j)) has weight equal to the geometric distance between the two cities (stored as attribute dist).

The objective is to start at the base and bring all gold back to the depot, by visiting all cities in some order, possibly returning to the base multiple times.

The cost of moving from (i) to (j) while carrying a load (w) is:

[ c_{ij}(w) = d_{ij} + (\alpha , d_{ij}, w)^{\beta} ]

where:

(d_{ij}) is the distance from (i) to (j),
(\alpha \ge 0) and (\beta \ge 0) are problem parameters,
(w) is the current carried gold (load).
Important detail: (d_{ij}) is interpreted as a shortest-path distance on the graph (using dist as edge weight). Therefore, costs must be computed piecewise along the edges of the shortest path, summing the contributions of each traversed edge, rather than aggregating distances incorrectly.

2. Role of (\alpha) and (\beta): practical observations
(\alpha) scales the load-dependent penalty. Larger (\alpha) makes transporting gold more expensive even for moderate travel distances.
(\beta) controls the non-linearity of the penalty term:
(\beta > 1): the penalty grows super-linearly → carrying large loads becomes very expensive → it is often beneficial to make more trips and avoid accumulating too much gold.
(\beta = 1): “balanced” case; in practice, it is difficult to significantly outperform the baseline.
(\beta < 1): the penalty grows sub-linearly → additional load is “relatively cheaper” → longer combined tours can be advantageous, and the solution quality depends strongly on geometric routing quality.
These behaviors motivate a two-regime heuristic.

3. Proposed solution: two-regime hybrid heuristic
The implemented method uses two different strategies depending on (\beta), while sharing common evaluation primitives (shortest-path distance and load-aware edge cost).

3.1 Regime A ((\beta < 1)): “TSP-first + split”
When (\beta < 1), the problem behaves similarly to a geometric routing problem (TSP/VRP-like) with mild load penalty. The adopted strategy is:

Build a global tour to minimize travel distance (shortest-path distances using dist):

Multi-start Nearest Neighbor (NN) from several candidate start cities.
Limited, sampled tour-level 2-opt to improve the global visiting order without excessive runtime.
Split the tour into multiple depot trips using the true load-dependent cost:

Greedy scan of the tour, deciding whether to keep extending the current trip or return to the base.
Soft caps on trip length and/or carried load to avoid degenerate behaviors.
This regime was introduced to fix the earlier inconsistency where (\beta<1) could produce “mega-tours” or suboptimal splits.

3.2 Regime B ((\beta \ge 1)): “Sweep + insertion + safe 2-opt”
When (\beta \ge 1), accumulating gold becomes risky. The strategy is:

Heavy-city handling (optional): a small fraction of high-gold cities is served with dedicated trips, especially for larger (\beta), to avoid large load spikes in longer routes.

Sweep ordering: cities are ordered by polar angle around the depot; multiple angular offsets are tested (multi-start).

Insertion-based route construction:

Compare the cost of inserting a city into the current route vs. starting a new route.
Use a split/merge criterion influenced by (\alpha,\beta) to control how aggressively routes are merged.
Safe route-level 2-opt:

Limited intra-route local search to refine routes.
Disabled or strongly reduced for long routes to prevent runtime blow-ups.
4. Experimental results (qualitative summary)
Experiments were run on multiple combinations of:

(N \in {100,200,500})
density (\in {0.2, 1.0}) (plus 0.3 for additional tests)
(\alpha) values such as 0.5, 1.0, 2.0
(\beta) values such as 0.5, 1.0, 1.5, 2.0, 4.0
Key observations:

(\beta=0.5): the dedicated “TSP-first + split” regime yields large and consistent improvements over the baseline (often in the ~14%–30% range depending on instance size and density).
(\beta=1.0): improvements are typically small (often close to 0%–0.2%), reflecting the difficulty of beating a strong baseline in this regime.
(\beta=4.0): improvements can be very large (often >30%), because controlling carried load becomes crucial and the heuristic is explicitly designed for that.
5. Strengths
Adaptivity to (\beta)
The method changes behavior across regimes, preventing degenerate solutions and exploiting the structure of the cost function.

Strong performance for (\beta>1)
Split/merge decisions, optional heavy-city handling, and local search yield significant gains where load penalties dominate.

Robust behavior for (\beta<1)
The “TSP-first + split” approach addresses the main weakness of naive merging strategies and improves solution quality substantially.

Better scalability than full evolutionary approaches
Compared to GA/memetic methods, the heuristic avoids population-based iterations and typically runs faster on many instances.

Shortest-path caching via MoveCache
Reduces repeated Dijkstra computations, improving runtime during repeated cost evaluations and local improvements.

6. Weaknesses and limitations
Runtime outliers on large ((N)) cases, especially for (\beta<1)
Tour-level improvement (2-opt) can become expensive if not carefully budgeted. This is mitigated by limiting moves/samples or disabling tour-level improvements for large (N).

No optimality guarantee
As a heuristic, the method does not guarantee global optimality; quality depends on initial construction and local decisions.

Sensitivity to internal parameters
The number of multi-start trials, 2-opt budget, and split caps can trade solution quality for runtime. Proper tuning is needed.

Intrinsic difficulty of (\beta=1)
Even sophisticated heuristics often yield only marginal improvements over the baseline.

7. Possible future improvements
Time-aware adaptive local search budgets as a function of (N) and density.
Dynamic programming (optimal splitting) given a fixed tour to find the best segmentation into trips, which may improve (\beta<1) further.
Inter-route neighborhood moves (swap/relocate across routes) in addition to intra-route 2-opt.
More refined heavy-city selection, considering not only gold but also spatial/graph structure.
8. Conclusions
This project proposes a two-regime hybrid heuristic for gold collection on a weighted graph with load-dependent travel costs. The solution leverages the qualitative differences across (\beta): it behaves like geometric routing for (\beta<1), and focuses on load control for (\beta\ge1). Experimental results show significant improvements over the baseline particularly for (\beta<1) and large (\beta), with robust performance across different (N) and densities. The main remaining concern is occasional runtime outliers on large instances in the (\beta<1) regime, which can be addressed by stricter local-search budgets with limited impact on solution quality.