"""Energy-efficient multi-hop routing in a wireless sensor network (WSN).

Requirements: Python 3.9+ and SciPy (`pip install scipy`).

LP model: maximize network lifetime T. f[i,j] is total bits sent over a
 directed link during T; r[i] is generated traffic rate; B[i] is battery energy.
The model omits link capacities, collisions, retransmissions, and packet scheduling.
"""
from __future__ import annotations
import argparse
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog

Node = int
Arc = Tuple[Node, Node]

@dataclass
class Network:
    positions: np.ndarray
    sink: int
    arcs: List[Arc]
    tx_energy: Dict[Arc, float]
    rx_energy: float
    generation_rate: Dict[Node, float]
    battery: Dict[Node, float]


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def make_network(n_sensors: int = 10, seed: int = 7, area_size_m: float = 100.0,
                 extra_link_probability: float = 0.28) -> Network:
    """Create a reproducible connected random network with one sink."""
    if n_sensors < 2: raise ValueError("Use at least 2 sensor nodes.")
    if not 0 <= extra_link_probability <= 1: raise ValueError("Link probability must be in [0, 1].")
    if area_size_m <= 0: raise ValueError("Area size must be positive.")
    rng = np.random.default_rng(seed)
    n_total, sink = n_sensors + 1, n_sensors
    positions = rng.uniform(0.0, area_size_m, size=(n_total, 2))
    candidates = sorted((euclidean(positions[i], positions[j]), i, j)
                        for i in range(n_total) for j in range(i + 1, n_total))
    parent = list(range(n_total))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edges = set()
    for _, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj; edges.add((i, j))
    for _, i, j in candidates:
        if (i, j) not in edges and rng.random() < extra_link_probability: edges.add((i, j))
    arcs = []
    for i, j in sorted(edges):
        if i != sink: arcs.append((i, j))
        if j != sink: arcs.append((j, i))
    tx_energy = {}
    for i, j in arcs:
        d = euclidean(positions[i], positions[j])
        tx_energy[(i, j)] = 50e-9 + 100e-12 * d**2
    generation_rate = {i: float(rng.uniform(500.0, 1500.0)) for i in range(n_sensors)}
    battery = {i: float(rng.uniform(0.4, 0.8)) for i in range(n_sensors)}
    return Network(positions, sink, arcs, tx_energy, 50e-9, generation_rate, battery)


def validate_network(net: Network) -> list[int]:
    """Return sensors that have no directed path to the sink."""
    adjacency = {i: [] for i in range(len(net.positions))}
    for u, v in net.arcs: adjacency.setdefault(u, []).append(v)
    disconnected = []
    for source in net.battery:
        seen, stack = {source}, [source]
        while stack:
            u = stack.pop()
            for v in adjacency.get(u, []):
                if v not in seen: seen.add(v); stack.append(v)
        if net.sink not in seen: disconnected.append(source)
    return disconnected


def solve_lifetime_lp(net: Network):
    """Solve lifetime-maximizing LP with SciPy/HiGHS and return diagnostics."""
    disconnected = validate_network(net)
    if disconnected: raise ValueError(f"Sensors cannot reach sink {net.sink}: {disconnected}")
    arcs, sensors = net.arcs, sorted(net.battery)
    t_idx, n_vars = len(arcs), len(arcs) + 1
    A_ub = np.zeros((len(sensors), n_vars)); b_ub = np.array([net.battery[i] for i in sensors], dtype=float)
    sensor_row = {node: row for row, node in enumerate(sensors)}
    for k, (i, j) in enumerate(arcs):
        if i in sensor_row: A_ub[sensor_row[i], k] += net.tx_energy[(i, j)]
        if j in sensor_row: A_ub[sensor_row[j], k] += net.rx_energy
    A_eq = np.zeros((len(sensors), n_vars)); b_eq = np.zeros(len(sensors))
    for row, i in enumerate(sensors):
        for k, (u, v) in enumerate(arcs):
            if u == i: A_eq[row, k] += 1.0
            if v == i: A_eq[row, k] -= 1.0
        A_eq[row, t_idx] = -net.generation_rate[i]
    objective = np.zeros(n_vars); objective[t_idx] = -1.0
    result = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * n_vars, method="highs")
    if not result.success: raise RuntimeError(f"LP solver failed: {result.message}")
    flows = {arc: float(result.x[k]) for k, arc in enumerate(arcs)}
    lifetime = float(result.x[t_idx]); energy_used = {i: 0.0 for i in sensors}
    for (i, j), flow in flows.items():
        energy_used[i] += net.tx_energy[(i, j)] * flow
        if j in energy_used: energy_used[j] += net.rx_energy * flow
    slack = {i: float(net.battery[i] - energy_used[i]) for i in sensors}
    shadow = {i: max(0.0, -float(result.ineqlin.marginals[sensor_row[i]])) for i in sensors}
    # Solver diagnostics: primal residuals and complementary-slackness products.
    eq_residual_max = float(np.max(np.abs(result.eqlin.residual))) if len(result.eqlin.residual) else 0.0
    ub_violation_max = float(max(0.0, np.max(-result.ineqlin.residual))) if len(result.ineqlin.residual) else 0.0
    battery_cs = {i: abs(slack[i] * shadow[i]) for i in sensors}
    flow_cs = {}
    lower_marginals = getattr(result, "lower", None)
    if lower_marginals is not None:
        for k, arc in enumerate(arcs):
            flow_cs[arc] = abs(flows[arc] * float(lower_marginals.marginals[k]))
    return {"lifetime": lifetime, "flows": flows, "energy_used": energy_used,
            "battery_slack": slack, "shadow_prices": shadow, "solver_result": result,
            "max_flow_conservation_residual": eq_residual_max,
            "max_battery_violation": ub_violation_max,
            "battery_complementarity_products": battery_cs,
            "flow_complementarity_products": flow_cs}


def dijkstra_path_to_sink(net: Network, source: int) -> List[Node]:
    """Shortest path by sum of transmission energy per bit."""
    outgoing: Dict[int, List[Tuple[int, float]]] = {}
    for i, j in net.arcs: outgoing.setdefault(i, []).append((j, net.tx_energy[(i, j)]))
    dist = {node: math.inf for node in range(len(net.positions))}; previous = {}; dist[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]: continue
        if u == net.sink: break
        for v, weight in outgoing.get(u, []):
            nd = d + weight
            if nd < dist[v]: dist[v] = nd; previous[v] = u; heapq.heappush(queue, (nd, v))
    if not math.isfinite(dist[net.sink]): raise RuntimeError(f"Sensor {source} cannot reach the sink.")
    path, cur = [net.sink], net.sink
    while cur != source: cur = previous[cur]; path.append(cur)
    return list(reversed(path))


def shortest_path_baseline(net: Network):
    """Route each sensor's own traffic along its shortest-energy path."""
    rates = {arc: 0.0 for arc in net.arcs}
    for source, rate in net.generation_rate.items():
        path = dijkstra_path_to_sink(net, source)
        for u, v in zip(path[:-1], path[1:]): rates[(u, v)] += rate
    power = {i: 0.0 for i in net.battery}
    for (i, j), rate in rates.items():
        power[i] += net.tx_energy[(i, j)] * rate
        if j in power: power[j] += net.rx_energy * rate
    node_lifetimes = {i: (net.battery[i] / p if p > 0 else math.inf) for i, p in power.items()}
    return {"lifetime": min(node_lifetimes.values()), "rates": rates, "power": power, "node_lifetimes": node_lifetimes}


def run_experiment(n_sensors: int, seed: int):
    net = make_network(n_sensors=n_sensors, seed=seed); lp = solve_lifetime_lp(net); baseline = shortest_path_baseline(net)
    improvement = 100.0 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"]
    tight = [i for i in sorted(net.battery) if abs(lp["energy_used"][i] - net.battery[i]) <= max(1e-8, 1e-5 * net.battery[i])]
    print(f"\n=== {n_sensors} sensors | seed={seed} ===")
    print(f"Directed links: {len(net.arcs)}\nLP lifetime: {lp['lifetime']:.6f} s\nShortest-path lifetime: {baseline['lifetime']:.6f} s")
    print(f"Improvement: {improvement:.2f}%\nBattery constraints tight at LP optimum: {tight}\nSolver status: {lp['solver_result'].message}")
    print("Sensor | generation (bit/s) | battery (J) | LP energy used (J)")
    for i in sorted(net.battery): print(f"{i:>6} | {net.generation_rate[i]:>18.2f} | {net.battery[i]:>11.4f} | {lp['energy_used'][i]:>18.6f}")
    print("Positive LP flows (total bits over optimized lifetime):")
    for (u, v), flow in lp["flows"].items():
        if flow > 1e-7: print(f"  {u} -> {v}: {flow:.6f} bits")
    return net, lp, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[10, 20], help="number of sensors (sink is added separately)")
    parser.add_argument("--seed", type=int, default=7, help="random seed for reproducibility")
    args = parser.parse_args()
    for n in args.nodes: run_experiment(n, args.seed)

if __name__ == "__main__": main()
