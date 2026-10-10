"""Research dashboard for wireless sensor network lifetime optimization.

Run alongside wsn_lifetime_lp.py with: streamlit run wsn_dashboard.py
Dependencies: streamlit, numpy, pandas, matplotlib, scipy, plotly.
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from wsn_lifetime_lp import Network, euclidean, make_network, solve_lifetime_lp, shortest_path_baseline

st.set_page_config(page_title="WSN Lifetime Lab", page_icon="📡", layout="wide")

# ---------- Model helpers ----------
def connected_to_sink(net: Network) -> tuple[bool, list[int]]:
    adjacency = {i: set() for i in range(len(net.positions))}
    for u, v in net.arcs:
        adjacency[u].add(v)
    disconnected = []
    for source in net.battery:
        seen, stack = {source}, [source]
        while stack:
            u = stack.pop()
            for v in adjacency.get(u, ()):
                if v not in seen:
                    seen.add(v); stack.append(v)
        if net.sink not in seen:
            disconnected.append(source)
    return not disconnected, disconnected


def set_topology(net: Network, undirected_edges: set[tuple[int, int]]) -> Network:
    """Rebuild directed arcs and distance-based transmit costs after editing links."""
    arcs = []
    for a, b in sorted({tuple(sorted(e)) for e in undirected_edges if e[0] != e[1]}):
        if a != net.sink: arcs.append((a, b))
        if b != net.sink: arcs.append((b, a))
    net.arcs = arcs
    # Keep the same illustrative first-order model as the LP module.
    net.tx_energy = {(u, v): 50e-9 + 100e-12 * euclidean(net.positions[u], net.positions[v]) ** 2 for u, v in arcs}
    return net


def edge_set(net: Network) -> set[tuple[int, int]]:
    return {tuple(sorted((u, v))) for u, v in net.arcs}


def battery_table(net, lp, baseline):
    out = []
    marginals = getattr(getattr(lp["solver_result"], "ineqlin", None), "marginals", np.zeros(len(net.battery)))
    for i in sorted(net.battery):
        power_lp = lp["energy_used"][i] / lp["lifetime"] if lp["lifetime"] > 0 else 0.0
        out.append({
            "Sensor": i,
            "Generation (bit/s)": net.generation_rate[i],
            "Battery (J)": net.battery[i],
            "LP transmit + receive energy (J)": lp["energy_used"][i],
            "LP average power (W)": power_lp,
            "Shortest-path power (W)": baseline["power"][i],
            "LP battery used (%)": min(100.0, 100 * lp["energy_used"][i] / net.battery[i]),
            "Battery slack at LP optimum (J)": max(0.0, net.battery[i] - lp["energy_used"][i]),
            "Shadow price (s/J)": max(0.0, -float(marginals[i])) if i < len(marginals) else 0.0,
        })
    return pd.DataFrame(out)


def topology_figure(net, flows, energy_fraction=None, packet_progress=None, title="Network topology"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    pos = net.positions
    fig = go.Figure()
    positive = [x for x in flows.values() if x > 1e-10]
    max_flow = max(positive) if positive else 1.0
    # Draw each undirected physical edge once; traffic direction is overlaid separately.
    seen = set()
    for u, v in net.arcs:
        key = tuple(sorted((u, v)))
        if key in seen: continue
        seen.add(key)
        fig.add_trace(go.Scatter(x=[pos[u, 0], pos[v, 0]], y=[pos[u, 1], pos[v, 1]],
            mode="lines", line=dict(color="#9aa4b2", width=1.2), hoverinfo="text",
            text=f"Link {u} — {v}", showlegend=False))
    # Directed flow arrows; width indicates relative traffic.
    for (u, v), value in flows.items():
        if value <= 1e-10: continue
        fig.add_annotation(x=pos[v, 0], y=pos[v, 1], ax=pos[u, 0], ay=pos[u, 1],
            xref="x", yref="y", axref="x", ayref="y", showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=1 + 3 * value / max_flow,
            arrowcolor="#0d9488", opacity=0.85)
    fractions = energy_fraction or {i: 0.0 for i in net.battery}
    sensor_ids = sorted(net.battery)
    fig.add_trace(go.Scatter(x=pos[sensor_ids, 0], y=pos[sensor_ids, 1], mode="markers+text",
        text=[str(i) for i in sensor_ids], textposition="middle center",
        marker=dict(size=18, color=[fractions.get(i, 0) for i in sensor_ids], cmin=0, cmax=1,
                    colorscale="RdYlGn_r", colorbar=dict(title="Battery used"), line=dict(color="#202938", width=1)),
        customdata=[[i, net.battery[i], net.generation_rate[i], fractions.get(i, 0)] for i in sensor_ids],
        hovertemplate="Sensor %{customdata[0]}<br>Battery: %{customdata[1]:.3g} J<br>Generation: %{customdata[2]:.1f} bit/s<br>Battery used: %{customdata[3]:.1%}<extra></extra>", name="Sensors"))
    s = net.sink
    fig.add_trace(go.Scatter(x=[pos[s, 0]], y=[pos[s, 1]], mode="markers+text", text=["SINK"],
        textposition="top center", marker=dict(symbol="star", size=24, color="#6d28d9"), name="Sink"))
    fig.update_layout(title=title, height=570, margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="x (m)", yaxis_title="y (m)", template="plotly_white", showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def animated_topology_figure(net, flows, power, lifetime, title="Live network animation", frame_count=32):
    """Plotly-native Play/Pause animation: moving packets + battery depletion.

    Packets are illustrative moving markers on links with positive average flow.
    Battery depletion follows the modelled average power, not packet-level radio events.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if lifetime <= 0:
        return topology_figure(net, flows, title=title)

    # Initial battery fractions at t=0.
    fractions0 = {i: 0.0 for i in net.battery}
    fig = topology_figure(net, flows, fractions0, title=title)
    if fig is None:
        return None
    active = [(u, v, val) for (u, v), val in flows.items() if val > 1e-10]
    if active:
        max_flow = max(v for _, _, v in active)
        # One moving marker per active directed flow; marker size reflects relative flow.
        packet_x, packet_y, packet_size = [], [], []
        for u, v, val in active:
            packet_x.append(float(net.positions[u, 0])); packet_y.append(float(net.positions[u, 1]))
            packet_size.append(6 + 8 * val / max_flow)
    else:
        packet_x, packet_y, packet_size = [], [], []

    edge_count = len(edge_set(net))
    sensor_trace_index = edge_count
    packet_trace_index = len(fig.data)
    fig.add_trace(go.Scatter(x=packet_x, y=packet_y, mode="markers",
        marker=dict(size=packet_size, color="#2563eb", line=dict(color="white", width=1)),
        name="Moving data packets", hoverinfo="skip"))

    frames = []
    for k in range(frame_count + 1):
        phase = k / frame_count
        t = lifetime * phase
        battery_used = {i: min(1.0, max(0.0, power.get(i, 0.0) * t / net.battery[i])) for i in net.battery}
        # The packet phase cycles along each active edge while the clock advances.
        px, py = [], []
        for u, v, _ in active:
            local_phase = (phase * 3.0) % 1.0
            px.append(float(net.positions[u, 0] + local_phase * (net.positions[v, 0] - net.positions[u, 0])))
            py.append(float(net.positions[u, 1] + local_phase * (net.positions[v, 1] - net.positions[u, 1])))
        frames.append(go.Frame(name=f"f{k}", data=[
            go.Scatter(marker=dict(color=[battery_used[i] for i in sorted(net.battery)])),
            go.Scatter(x=px, y=py),
        ], traces=[sensor_trace_index, packet_trace_index], layout=go.Layout(title_text=f"{title} · t = {t:.2f} s / {lifetime:.2f} s")))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.13,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
            ])],
        sliders=[dict(active=0, x=0.1, y=0, len=0.85, currentvalue={"prefix": "Simulation frame: "},
            steps=[dict(method="animate", args=[[f"f{k}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=str(k)) for k in range(frame_count + 1)])],
        margin=dict(t=90, b=60),
    )
    return fig

def run_model(net):
    ok, disconnected = connected_to_sink(net)
    if not ok:
        raise ValueError(f"Disconnected sensors cannot reach the sink: {disconnected}. Add links before solving.")
    lp = solve_lifetime_lp(net)
    baseline = shortest_path_baseline(net)
    return lp, baseline

# ---------- Sidebar / session state ----------
with st.sidebar:
    st.title("📡 WSN Lifetime Lab")
    st.caption("Controls persist while exploring the five research tabs.")
    n_sensors = st.slider("Sensor count", 5, 50, 20, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)
    density = st.slider("Extra-link probability", 0.0, 0.8, 0.28, 0.04)
    battery_scale = st.slider("Battery capacity scale", 0.25, 2.0, 1.0, 0.05)
    strategy = st.radio("Traffic shown", ["LP optimized", "Shortest-path baseline"])
    if st.button("Reset network to generated topology", use_container_width=True):
        st.session_state.pop("custom_net", None)
        st.session_state.pop("sim_time", None)
        st.rerun()

base_key = (int(n_sensors), int(seed), float(density), float(battery_scale))
if st.session_state.get("base_key") != base_key:
    st.session_state["base_key"] = base_key
    net = make_network(n_sensors=n_sensors, seed=int(seed), extra_link_probability=float(density))
    net.battery = {i: b * battery_scale for i, b in net.battery.items()}
    st.session_state["custom_net"] = net
    st.session_state["sim_time"] = 0.0
net: Network = st.session_state["custom_net"]

try:
    lp, baseline = run_model(net)
    model_error = None
except Exception as exc:
    lp = baseline = None
    model_error = str(exc)

st.title("Wireless Sensor Network — Lifetime Optimization")
st.caption("Explore network topology, battery depletion, LP routing, dual sensitivity, and reproducible experiments.")
if model_error:
    st.error(f"Model validity / solver warning: {model_error}")
    st.info("Open **Network Simulator** and add links until every sensor can reach the sink.")

if lp and baseline:
    improvement = 100 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"] if baseline["lifetime"] else 0.0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Network lifetime (LP)", f"{lp['lifetime']:.2f} s")
    m2.metric("Lifetime improvement", f"{improvement:+.1f}%")
    m3.metric("Sensors / links", f"{len(net.battery)} / {len(edge_set(net))}")
    m4.metric("Energy available", f"{sum(net.battery.values()):.3f} J")
    m5.metric("Solver", "Optimal" if lp["solver_result"].success else "Failed")

# ---------- Five main tabs ----------
t_overview, t_network, t_energy, t_analysis, t_experiments = st.tabs([
    "01 · Overview", "02 · Network Simulator", "03 · Energy Simulation",
    "04 · Optimization & Analysis", "05 · Experiments & Reports"])

with t_overview:
    st.subheader("Simulation overview")
    if not lp:
        st.warning("A valid connected network is required for summary metrics.")
    else:
        left, right = st.columns([1.4, 1])
        with left:
            weights = lp["flows"] if strategy == "LP optimized" else baseline["rates"]
            selected_life = lp["lifetime"] if strategy == "LP optimized" else baseline["lifetime"]
            used = lp["energy_used"] if strategy == "LP optimized" else {i: baseline["power"][i] * selected_life for i in net.battery}
            fractions = {i: min(1.0, used[i] / net.battery[i]) for i in net.battery}
            fig = topology_figure(net, weights, fractions, title=f"Current topology · {strategy}")
            if fig is not None: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Install Plotly (`pip install plotly`) for interactive topology visualization.")
        with right:
            st.markdown("#### Lifetime comparison")
            comp = pd.DataFrame({"Strategy": ["Shortest path", "LP optimized"], "Lifetime (s)": [baseline["lifetime"], lp["lifetime"]]})
            st.bar_chart(comp.set_index("Strategy"))
            st.write(f"**LP solver:** {lp['solver_result'].message}")
            ok, disconnected = connected_to_sink(net)
            st.write(f"**Connectivity:** {'All sensors reach sink' if ok else 'Disconnected sensors: ' + str(disconnected)}")
            st.write(f"**Total LP energy used:** {sum(lp['energy_used'].values()):.5f} J")
        st.markdown("#### Network health")
        table = battery_table(net, lp, baseline)
        st.dataframe(table, use_container_width=True, hide_index=True)

with t_network:
    st.subheader("Interactive topology editor")
    st.markdown("Add/remove physical links using node IDs, or edit coordinates in the table. Link changes are checked for sink reachability before solving.")
    c1, c2, c3 = st.columns(3)
    nodes = list(range(len(net.positions)))
    with c1: node_a = st.selectbox("First node", nodes, key="edge_a")
    with c2: node_b = st.selectbox("Second node", [x for x in nodes if x != node_a], key="edge_b")
    with c3:
        action = st.selectbox("Link action", ["Add link", "Remove link"], key="edge_action")
    if st.button("Apply link edit", type="primary"):
        edges = edge_set(net); edge = tuple(sorted((node_a, node_b)))
        if action == "Add link": edges.add(edge)
        else: edges.discard(edge)
        st.session_state["custom_net"] = set_topology(net, edges)
        st.rerun()
    st.caption(f"Physical links: {len(edge_set(net))}. The sink is node {net.sink}; sink-outgoing links are not modeled.")
    st.markdown("#### Move a node")
    move_node = st.selectbox("Node to move", list(range(len(net.positions))), key="move_node_id")
    mx, my = st.columns(2)
    with mx:
        move_x = st.number_input("New x coordinate (m)", value=float(net.positions[move_node, 0]), key="move_node_x")
    with my:
        move_y = st.number_input("New y coordinate (m)", value=float(net.positions[move_node, 1]), key="move_node_y")
    if st.button("Apply selected node position"):
        if not (math.isfinite(float(move_x)) and math.isfinite(float(move_y))):
            st.error("Coordinates must be finite numbers.")
        else:
            net.positions[move_node] = [float(move_x), float(move_y)]
            set_topology(net, edge_set(net))
            st.session_state["custom_net"] = net
            st.rerun()
    coords = pd.DataFrame([{"Node": i, "x (m)": float(net.positions[i, 0]), "y (m)": float(net.positions[i, 1]), "Type": "Sink" if i == net.sink else "Sensor"} for i in range(len(net.positions))])
    edited = st.data_editor(coords, use_container_width=True, hide_index=True, disabled=["Node", "Type"], key="coordinates_editor")
    if st.button("Apply node positions"):
        try:
            for _, row in edited.iterrows():
                i = int(row["Node"]); x, y = float(row["x (m)"],), float(row["y (m)"])
                if not (math.isfinite(x) and math.isfinite(y)): raise ValueError("Coordinates must be finite numbers.")
                net.positions[i] = [x, y]
            set_topology(net, edge_set(net))
            st.session_state["custom_net"] = net
            st.rerun()
        except Exception as exc: st.error(f"Could not apply coordinates: {exc}")
    if lp and baseline:
        weights = lp["flows"] if strategy == "LP optimized" else baseline["rates"]
        frac = {i: min(1, lp["energy_used"][i] / net.battery[i]) for i in net.battery}
        fig = topology_figure(net, weights, frac, title="Edited network")
        if fig is not None: st.plotly_chart(fig, use_container_width=True)
    with st.expander("Topology validity details"):
        ok, disconnected = connected_to_sink(net)
        if ok: st.success("Every sensor has a directed path to the sink.")
        else: st.error(f"These sensors cannot reach the sink: {disconnected}. The LP is not run on invalid topology.")

with t_energy:
    st.subheader("Battery depletion and traffic animation")
    if not lp or not baseline:
        st.warning("Fix connectivity in the Network Simulator first.")
    else:
        selected_lp = strategy == "LP optimized"
        lifetime = lp["lifetime"] if selected_lp else baseline["lifetime"]
        power = ({i: lp["energy_used"][i] / lp["lifetime"] for i in net.battery} if selected_lp else baseline["power"])
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            if st.button("▶ Advance one step", use_container_width=True):
                st.session_state["sim_time"] = min(lifetime, st.session_state.get("sim_time", 0) + max(lifetime / 100, 0.01) * st.session_state.get("sim_speed", 5))
        with c2:
            if st.button("↺ Reset clock", use_container_width=True): st.session_state["sim_time"] = 0.0
        with c3:
            speed = st.select_slider("Step size", options=[1, 2, 5, 10, 20], value=5, format_func=lambda x: f"{x}×", key="sim_speed")
        # Step size is applied to the next click; this explicit clock keeps runs deterministic.
        current_time = st.slider("Simulated time (s)", 0.0, float(max(lifetime, 0.001)),
                                 min(float(st.session_state.get("sim_time", 0.0)), float(max(lifetime, 0.001))),
                                 step=float(max(lifetime / 100, 0.001)), key="sim_clock")
        st.session_state["sim_time"] = current_time
        st.progress(min(1.0, current_time / lifetime) if lifetime > 0 else 1.0, text=f"Elapsed {current_time:.2f} s of {lifetime:.2f} s")
        remaining = {i: max(0.0, net.battery[i] - power[i] * current_time) for i in net.battery}
        fraction_used = {i: 1 - remaining[i] / net.battery[i] for i in net.battery}
        first_out = min(net.battery, key=lambda i: net.battery[i] / power[i] if power[i] > 0 else math.inf)
        exhaustion = net.battery[first_out] / power[first_out] if power[first_out] > 0 else math.inf
        if current_time >= exhaustion - 1e-9:
            st.error(f"First sensor to exhaust its battery: sensor {first_out} at approximately {exhaustion:.2f} s.")
        else:
            st.info(f"Predicted first battery exhaustion: sensor {first_out} at approximately {exhaustion:.2f} s.")
        weights = lp["flows"] if selected_lp else baseline["rates"]
        st.markdown("#### Packet and battery animation")
        st.caption("Use the chart's ▶ Play / ⏸ Pause controls or its frame slider. Moving markers illustrate traffic on active links; battery colors follow the average-power model.")
        anim_fig = animated_topology_figure(net, weights, power, lifetime, title=f"{strategy} · average-rate simulation")
        if anim_fig is not None:
            st.plotly_chart(anim_fig, use_container_width=True, key="wsn_live_animation")
        else:
            fig = topology_figure(net, weights, fraction_used, title=f"Battery state at t = {current_time:.2f} s")
            if fig is not None: st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Current state at selected simulation time")
        fig_now = topology_figure(net, weights, fraction_used, title=f"Battery state at t = {current_time:.2f} s")
        if fig_now is not None: st.plotly_chart(fig_now, use_container_width=True, key="wsn_current_state")
        battery_df = pd.DataFrame([{"Sensor": i, "Remaining energy (J)": remaining[i], "Remaining (%)": 100 * remaining[i] / net.battery[i], "Power (W)": power[i]} for i in sorted(net.battery)])
        st.dataframe(battery_df, use_container_width=True, hide_index=True)
        st.caption("This is a deterministic average-rate depletion animation, not packet-level radio simulation. LP flow variables are total bits over the optimized lifetime and are converted to average power for visualization.")

with t_analysis:
    st.subheader("Optimization, sensitivity and bottlenecks")
    if not lp or not baseline:
        st.warning("A valid network is required for analysis.")
    else:
        df = battery_table(net, lp, baseline)
        a, b = st.columns(2)
        with a:
            st.markdown("#### Per-sensor energy burden")
            st.bar_chart(df.set_index("Sensor")[["LP average power (W)", "Shortest-path power (W)"]])
        with b:
            st.markdown("#### Shadow prices")
            st.bar_chart(df.set_index("Sensor")[["Shadow price (s/J)"]])
        slack_tol = max(1e-8, 1e-5 * max(net.battery.values()))
        df["Binding battery constraint"] = df["Battery slack at LP optimum (J)"] <= slack_tol
        df["Zero LP flow at sensor? (approx.)"] = [sum(v for (u, _), v in lp["flows"].items() if u == int(i)) < 1e-8 for i in df["Sensor"]]
        st.markdown("#### Complementary-slackness / bottleneck diagnostics")
        st.write("A battery constraint is approximately binding when its remaining slack is near zero. Shadow prices estimate the local lifetime value of extra battery energy under this LP formulation; they are not guaranteed gains for large battery changes.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write(f"**Binding battery constraints:** {df.loc[df['Binding battery constraint'], 'Sensor'].astype(int).tolist()}")
        st.write(f"**Largest LP energy burden:** sensor {int(df.loc[df['LP average power (W)'].idxmax(), 'Sensor'])}")
        st.write(f"**Solver status:** {lp['solver_result'].message}")
        st.write(f"**Equality residual max abs:** {float(np.max(np.abs(lp['solver_result'].eqlin.residual))):.3e}")
        st.write(f"**Inequality feasibility max violation:** {float(max(0, np.max(-lp['solver_result'].ineqlin.residual))):.3e}")
        comp_products = [abs(lp["battery_slack"][i] * lp["shadow_prices"][i]) for i in net.battery]
        lower_marginals = getattr(lp["solver_result"], "lower", None)
        flow_comp = 0.0
        if lower_marginals is not None:
            flow_comp = max((abs(lp["flows"][arc] * float(lower_marginals.marginals[k])) for k, arc in enumerate(net.arcs)), default=0.0)
        d1, d2 = st.columns(2)
        d1.metric("Max battery slack × shadow price", f"{max(comp_products, default=0.0):.3e}")
        d2.metric("Max flow × lower-bound marginal", f"{flow_comp:.3e}")
        st.caption("These are numerical complementary-slackness residual checks. Interpret them with solver tolerances and the LP's sign convention; they should be close to zero at an accurate optimum.")
        with st.expander("Interpretation notes"):
            st.markdown("""- LP routing distributes total flow to maximize lifetime subject to flow conservation and battery limits.
- Shortest-path baseline minimizes the sum of per-bit transmission costs along each sensor's route; it does not optimize network-wide lifetime.
- A high shadow price suggests lifetime is sensitive to that sensor's battery near the current optimum.
- Complementary slackness is approximate in floating-point solver output; use a stated tolerance.
- Link capacities, collisions, retransmissions, sleep states and packet scheduling are not included in this model.""")

with t_experiments:
    st.subheader("Repeatable experiments and exports")
    st.markdown("Run a seed sweep with the current sensor count, link density and battery scale. Every row records settings, solver status and both routing lifetimes.")
    c1, c2 = st.columns(2)
    with c1: n_runs = st.slider("Number of random seeds", 2, 50, 10)
    with c2: seed_start = st.number_input("First experiment seed", 0, 100000, int(seed), 1)
    if st.button("Run multi-seed experiment", type="primary"):
        records = []
        progress = st.progress(0.0, text="Running experiments...")
        for k in range(n_runs):
            try:
                enet = make_network(n_sensors=n_sensors, seed=int(seed_start) + k, extra_link_probability=float(density))
                enet.battery = {i: b * battery_scale for i, b in enet.battery.items()}
                elp, ebase = run_model(enet)
                gain = 100 * (elp["lifetime"] - ebase["lifetime"]) / ebase["lifetime"] if ebase["lifetime"] else 0.0
                records.append({"Seed": int(seed_start) + k, "Sensors": n_sensors, "Extra-link probability": density,
                    "Battery scale": battery_scale, "Directed links": len(enet.arcs), "LP lifetime (s)": elp["lifetime"],
                    "Shortest-path lifetime (s)": ebase["lifetime"], "Improvement (%)": gain,
                    "LP solver status": elp["solver_result"].message})
            except Exception as exc:
                records.append({"Seed": int(seed_start) + k, "Sensors": n_sensors, "Extra-link probability": density,
                    "Battery scale": battery_scale, "LP solver status": f"FAILED: {exc}"})
            progress.progress((k + 1) / n_runs, text=f"Completed {k + 1}/{n_runs}")
        st.session_state["experiment_results"] = pd.DataFrame(records)
    if "experiment_results" in st.session_state:
        exp = st.session_state["experiment_results"]
        st.dataframe(exp, use_container_width=True, hide_index=True)
        numeric = exp[pd.to_numeric(exp.get("LP lifetime (s)"), errors="coerce").notna()] if "LP lifetime (s)" in exp else pd.DataFrame()
        if not numeric.empty:
            st.markdown("#### Experiment summary")
            life = numeric["LP lifetime (s)"].astype(float)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Valid runs", len(numeric)); s2.metric("Mean LP lifetime", f"{life.mean():.2f} s")
            s3.metric("Median", f"{life.median():.2f} s"); s4.metric("Worst case", f"{life.min():.2f} s")
            st.line_chart(numeric.set_index("Seed")[["LP lifetime (s)", "Shortest-path lifetime (s)"]])
            st.bar_chart(numeric.set_index("Seed")[["Improvement (%)"]])
        csv = exp.to_csv(index=False).encode("utf-8")
        st.download_button("Download experiment results (CSV)", data=csv, file_name="wsn_experiments.csv", mime="text/csv")
    if lp and baseline:
        current = battery_table(net, lp, baseline)
        st.download_button("Download sensor diagnostics (CSV)", current.to_csv(index=False).encode("utf-8"),
            file_name="wsn_sensor_diagnostics.csv", mime="text/csv")
        # Publication-friendly figure exports.
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        ax.bar(["Shortest path", "LP optimized"], [baseline["lifetime"], lp["lifetime"]])
        ax.set_ylabel("Network lifetime (s)"); ax.set_title("WSN lifetime comparison"); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
        st.download_button("Download publication-quality lifetime figure (PNG)", buf.getvalue(), "wsn_lifetime_comparison.png", "image/png")
        report = ["WSN Lifetime Optimization — Experiment Summary", "", f"Sensors: {n_sensors}", f"Seed: {seed}",
            f"Extra-link probability: {density}", f"Battery scale: {battery_scale}", f"Directed links: {len(net.arcs)}",
            f"LP lifetime (s): {lp['lifetime']:.8g}", f"Shortest-path lifetime (s): {baseline['lifetime']:.8g}",
            f"Improvement (%): {improvement:.6g}", f"Solver status: {lp['solver_result'].message}",
            "Model: total-flow lifetime LP; first-order illustrative radio-energy model.",
            f"Max flow-conservation residual: {float(np.max(np.abs(lp['solver_result'].eqlin.residual))):.8g}",
            f"Max battery constraint violation: {float(max(0, np.max(-lp['solver_result'].ineqlin.residual))):.8g}",
            "Excluded: link capacities, packet collisions, retransmissions, and packet scheduling."]
        st.download_button("Download run summary (TXT)", "\n".join(report).encode("utf-8"), "wsn_run_summary.txt", "text/plain")

st.divider()
st.caption("Research note: results are model-based, not hardware measurements. Keep model assumptions and all experiment settings with exported results.")
