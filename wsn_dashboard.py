"""Interactive Streamlit dashboard for the WSN lifetime LP project.

Run from the folder containing wsn_lifetime_lp.py:
    streamlit run wsn_dashboard.py
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from wsn_lifetime_lp import make_network, solve_lifetime_lp, shortest_path_baseline

st.set_page_config(page_title="WSN Lifetime Optimization", layout="wide")
st.title("Wireless Sensor Network: Lifetime Optimization")
st.caption("Compare shortest-path routing with LP-based traffic distribution. Model parameters are illustrative.")

with st.sidebar:
    st.header("Network settings")
    n_sensors = st.slider("Number of sensor nodes", min_value=5, max_value=50, value=30, step=5)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)
    density = st.slider("Extra-link probability", min_value=0.0, max_value=0.8, value=0.28, step=0.04)
    mode = st.radio("Routing strategy to visualize", ["LP optimized", "Shortest-path baseline"])
    run = st.button("Run / update network", type="primary")

# Streamlit reruns on interaction; this button also provides a clear action cue.
@st.cache_data(show_spinner=False)
def calculate(n, random_seed, extra_probability):
    net = make_network(n_sensors=int(n), seed=int(random_seed), extra_link_probability=float(extra_probability))
    lp = solve_lifetime_lp(net)
    baseline = shortest_path_baseline(net)
    return net, lp, baseline

try:
    with st.spinner("Generating network and solving the LP..."):
        net, lp, baseline = calculate(n_sensors, seed, density)
except Exception as exc:
    st.error(f"Could not solve this network: {exc}")
    st.stop()

improvement = 100 * (lp["lifetime"] - baseline["lifetime"]) / baseline["lifetime"]
selected_lp = mode == "LP optimized"
selected_lifetime = lp["lifetime"] if selected_lp else baseline["lifetime"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sensors", n_sensors)
m2.metric("Directed links", len(net.arcs))
m3.metric("LP lifetime", f"{lp['lifetime']:.2f} s")
m4.metric("LP improvement", f"{improvement:.1f}%")
st.divider()

left, right = st.columns([1.5, 1])
with left:
    st.subheader(f"Network topology — {mode}")
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = net.positions
    # Edge widths are proportional to the selected strategy's traffic.
    weights = lp["flows"] if selected_lp else baseline["rates"]
    positive = [v for v in weights.values() if v > 1e-8]
    max_flow = max(positive) if positive else 1.0
    for (u, v) in net.arcs:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        val = weights.get((u, v), 0.0)
        if val > 1e-8:
            width = 0.6 + 3.0 * val / max_flow
            alpha = 0.85
        else:
            width = 0.45
            alpha = 0.18
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linewidth=width, alpha=alpha,
                                    color="#1f77b4" if val > 1e-8 else "#999999",
                                    shrinkA=7, shrinkB=7))
    sensors = [i for i in range(n_sensors)]
    energy_used = lp["energy_used"] if selected_lp else {
        i: baseline["power"][i] * baseline["lifetime"] for i in sensors
    }
    fraction_used = [min(1.0, energy_used[i] / net.battery[i]) for i in sensors]
    sc = ax.scatter(pos[sensors, 0], pos[sensors, 1], c=fraction_used, cmap="RdYlGn_r",
                    vmin=0, vmax=1, s=90, edgecolors="black", zorder=3)
    ax.scatter(pos[net.sink, 0], pos[net.sink, 1], marker="*", s=300,
               c="#6a3d9a", edgecolors="black", zorder=4, label="Sink")
    for i in sensors:
        ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=7, ha="center", va="center", zorder=5)
    fig.colorbar(sc, ax=ax, label="Battery energy used / available (fraction)")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Arrow thickness indicates traffic on that link. Pale arrows are available links carrying little or no traffic. Node color shows fraction of battery consumed by the selected strategy.")

with right:
    st.subheader("Lifetime comparison")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.bar(["Shortest path", "LP optimized"], [baseline["lifetime"], lp["lifetime"]])
    ax2.set_ylabel("Lifetime (seconds)")
    ax2.grid(axis="y", alpha=0.25)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.metric("Currently selected strategy lifetime", f"{selected_lifetime:.2f} s")
    st.write(f"**Solver status:** {lp['solver_result'].message}")
    st.write(f"**Reception energy:** {net.rx_energy:.3g} J/bit")

st.subheader("Sensor-level energy table")
rows = []
for i in sorted(net.battery):
    used_energy = lp["energy_used"][i] if selected_lp else baseline["power"][i] * baseline["lifetime"]
    rows.append({
        "Sensor": i,
        "Data generation (bit/s)": net.generation_rate[i],
        "Battery (J)": net.battery[i],
        "Energy used by selected strategy (J)": used_energy,
        "Battery used (%)": min(100.0, 100.0 * used_energy / net.battery[i]),
        "LP shadow price (s/J)": max(0.0, -float(lp["solver_result"].ineqlin.marginals[i])),
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

with st.expander("Model assumptions and interpretation"):
    st.markdown("""
    - One fixed sink collects data; the sink has no battery constraint.
    - Sensor-to-sensor links are bidirectional, and the generated topology is connected.
    - Radio transmission energy depends on link distance using the first-order model in `wsn_lifetime_lp.py`.
    - LP variables represent total bits sent over the optimized lifetime; they are not instantaneous bit rates.
    - Shortest-path baseline sends each sensor's own traffic along a minimum-transmission-energy path, then accounts for forwarding and reception energy.
    - Results are simulation outputs, not hardware measurements. Energy parameters should be justified and tested for robustness in the final report.
    """)
