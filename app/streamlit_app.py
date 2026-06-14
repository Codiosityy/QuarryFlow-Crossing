from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quarryflow.dashboard_data import (  # noqa: E402
    action_summary_frame,
    actions_frame,
    comparison_frame,
    decision_trace_frame,
    history_frame,
    improvement_frame,
    judge_summary,
    run_policy_suite,
    vehicle_frame,
)
from quarryflow.hybrid import load_hybrid_controller  # noqa: E402
from quarryflow.model import SurrogateModel  # noqa: E402
from quarryflow.reporting import build_assumptions_markdown, build_pitch_markdown  # noqa: E402
from quarryflow.scenarios import build_scenario  # noqa: E402

# ═══════════════════════════════════════════════════════════════
#  Plotly Theme
# ═══════════════════════════════════════════════════════════════

QUARRY_COLORS = [
    "#55f0a6", "#ffbf4d", "#4dd4ff", "#b68fff",
    "#ff6a6a", "#f0e655", "#ff8f6a", "#6af0e6",
]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#b8e6d0", size=12),
        title=dict(font=dict(size=15, color="#eafff4")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=24, r=24, t=52, b=24),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(85, 240, 166, 0.08)",
            linecolor="rgba(85, 240, 166, 0.15)",
            zerolinecolor="rgba(85, 240, 166, 0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(85, 240, 166, 0.08)",
            linecolor="rgba(85, 240, 166, 0.15)",
            zerolinecolor="rgba(85, 240, 166, 0.1)",
        ),
        colorway=QUARRY_COLORS,
    )
)

pio.templates["quarryflow"] = PLOTLY_TEMPLATE
pio.templates.default = "quarryflow"


st.set_page_config(
    page_title="QuarryFlow Crossing — Railway Bottleneck Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def load_css() -> None:
    css_path = ROOT / "app" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def num(value: float) -> str:
    return f"{value:.2f}"


def read_optional_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def metric_card(label: str, value: str, delta: str, subtext: str) -> str:
    return f"""
    <div class="metric-panel">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-delta">{delta}</div>
      <div class="metric-subtext">{subtext}</div>
    </div>
    """


def add_closure_bands(fig: go.Figure, closures: list[tuple[float, float]]) -> go.Figure:
    for start, end in closures:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 106, 106, 0.1)",
            line_width=0, layer="below",
            annotation_text="🚂", annotation_position="top left",
            annotation_font_size=10,
        )
    return fig


def build_snapshot_figure(vehicles: pd.DataFrame, scenario_name: str, seed: int, time_value: float) -> go.Figure:
    config = build_scenario(scenario_name, seed=seed)
    half_box = config.crossing_box_length / 2
    approach = config.approach_length

    vehicles = vehicles.copy()
    vehicles["x"] = vehicles.apply(
        lambda row: row["progress"] - approach - half_box
        if row["side"] == "left"
        else -(row["progress"] - approach - half_box),
        axis=1,
    )
    vehicles["y"] = vehicles.apply(
        lambda row: 1.1 + row["lateral_offset"]
        if row["side"] == "left"
        else -1.1 + row["lateral_offset"],
        axis=1,
    )

    fig = px.scatter(
        vehicles, x="x", y="y",
        color="vehicle_class", symbol="side",
        hover_data=["vehicle_id", "speed", "progress"],
        color_discrete_map={"car": "#55f0a6", "bike": "#4dd4ff", "auto": "#ffbf4d"},
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1.5, color="#060d0b"), opacity=0.9))
    fig.update_layout(
        title=f"⏱ t = {time_value:.1f}s",
        xaxis_title="Distance from crossing center (m)",
        yaxis_title="",
        height=480,
        xaxis=dict(range=[-approach - 12, approach + config.recovery_length + 12]),
        yaxis=dict(range=[-3.4, 3.4], showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    # Approach lanes
    fig.add_hrect(y0=0.3, y1=2.6, fillcolor="rgba(85, 240, 166, 0.04)", line_width=0)
    fig.add_hrect(y0=-2.6, y1=-0.3, fillcolor="rgba(85, 240, 166, 0.04)", line_width=0)
    # Crossing box
    fig.add_vrect(x0=-half_box, x1=half_box, fillcolor="rgba(255, 191, 77, 0.08)", line_width=0)
    fig.add_vline(x=0, line_color="rgba(255,191,77,0.5)", line_dash="dash")
    fig.add_annotation(x=0, y=3.1, text="🚧 Crossing Box", showarrow=False, font=dict(color="#ffbf4d", size=11))
    fig.add_annotation(x=-approach + 18, y=2.9, text="← Left", showarrow=False, font=dict(color="#55f0a6", size=10))
    fig.add_annotation(x=approach - 18, y=-2.9, text="Right →", showarrow=False, font=dict(color="#55f0a6", size=10))
    return fig


# ═══════════════════════════════════════════════════════════════
#  Simulation Runner
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def _run_cached_suite(scenario, seed, model_path, record_history, fast_mode):
    return run_policy_suite(
        scenario, seed=seed,
        model_path=model_path,
        record_history=record_history, fast_mode=fast_mode,
    )


def load_run(scenario, seed, model_path, *, record_history=True, fast_mode=False):
    model_path_arg = model_path if model_path and Path(model_path).exists() else None

    if fast_mode:
        with st.spinner("⚡ Loading preview..."):
            results = _run_cached_suite(
                scenario, seed=seed,
                model_path=model_path_arg,
                record_history=record_history, fast_mode=True,
            )
    else:
        progress_bar = st.progress(0, text="🔬 Running Hybrid MCTS Simulation...")
        results = run_policy_suite(
            scenario, seed=seed,
            model_path=model_path_arg,
            record_history=record_history, fast_mode=False,
            progress_callback=lambda p, m: progress_bar.progress(min(p, 1.0), text=f"🔬 {m}"),
        )
        progress_bar.empty()

    st.session_state["quarryflow_results"] = results
    st.session_state["quarryflow_summary"] = judge_summary(results, scenario)
    st.session_state["quarryflow_scenario"] = scenario
    st.session_state["quarryflow_seed"] = seed
    st.session_state["quarryflow_fast_mode"] = fast_mode
    st.session_state["quarryflow_model_label"] = "MCTS Agent"


# ═══════════════════════════════════════════════════════════════
#  Page Layout
# ═══════════════════════════════════════════════════════════════

load_css()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-shell">
  <div class="eyebrow">CONVOKE 8.0 · Data Science Challenge</div>
  <div class="hero-title">QuarryFlow Crossing</div>
  <div class="hero-copy">
    A behaviour-aware railway-crossing surge optimizer that reveals why traffic stays gridlocked
    after the barrier lifts — then demonstrates how adaptive staged release cuts delay,
    improves throughput, and reduces conflict without changing infrastructure.
  </div>
  <div class="tag-row">
    <span class="tag-chip">🧪 Microsimulation</span>
    <span class="tag-chip">🧠 ML Surrogate</span>
    <span class="tag-chip">🎰 LinUCB Bandit</span>
    <span class="tag-chip">🛡️ Safety Shield</span>
    <span class="tag-chip">📊 Counterfactual</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Config")
    scenario = st.selectbox(
        "Scenario",
        ["peak", "chaotic", "light", "peak_left_skew", "peak_right_skew", "chaotic_aggressive", "chaotic_long_gate"],
        index=0,
        help="Choose traffic scenario. Peak = best for judging. Chaotic = most dramatic contrast.",
    )
    seed = st.number_input("Random Seed", min_value=1, max_value=999, value=11, step=1)

    st.markdown("---")
    st.markdown("### 📁 ML Prior Models")
    model_path = st.text_input("Legacy Model", value=str(ROOT / "artifacts" / "models" / "surrogate.pkl"))

    st.markdown("---")
    run_button = st.button("🚀 Run Full Hybrid Simulation", type="primary", width='stretch')
    st.caption("Initial load uses fast heuristic preview. Full mode runs ML-Guided MCTS Rollout agent.")

# ── Load Data ─────────────────────────────────────────────────
if run_button:
    _run_cached_suite.clear()
    load_run(scenario, int(seed), model_path, record_history=True, fast_mode=False)
elif "quarryflow_results" not in st.session_state:
    load_run(scenario, int(seed), model_path, record_history=True, fast_mode=True)

results = st.session_state["quarryflow_results"]
summary = st.session_state["quarryflow_summary"]
scenario = st.session_state["quarryflow_scenario"]
seed = st.session_state["quarryflow_seed"]
model_label = st.session_state.get("quarryflow_model_label", "heuristic-only")

is_fast_mode = st.session_state.get("quarryflow_fast_mode", False)
config = build_scenario(scenario, seed=seed)
primary_policy = "MCTS Rollout"
comp = comparison_frame(results)
impr = improvement_frame(results)

# ── Mode indicator ────────────────────────────────────────────
mode_label = "preview" if is_fast_mode else "live"
mode_class = "preview" if is_fast_mode else "live"
mode_text = "Fast Preview" if is_fast_mode else "Full MCTS Mode"
st.markdown(f'<span class="status-badge {mode_class}">{mode_text}</span>', unsafe_allow_html=True)

# ── About Section ─────────────────────────────────────────────
st.markdown("""
<div class="about-section">
  <div class="about-heading">The Problem → The Approach → The Result</div>
  <div class="about-body">
    When a train passes and the barrier lifts, traffic <strong>should</strong> resume instantly — but it doesn't.
    Vehicles from both sides rush the crossing box simultaneously, creating lateral squeeze, disorder,
    and conflict freezes that keep the road jammed <strong>long after the train is gone</strong>.
    QuarryFlow models this bottleneck with a behaviour-aware microsimulator, evaluates staged-release
    strategies via counterfactual prediction, and picks the best action using a <strong>safety-shielded
    contextual bandit</strong>.
  </div>
  <div class="highlight-row">
    <div class="highlight-card">
      <div class="highlight-card-value">6</div>
      <div class="highlight-card-label">Driver Profiles</div>
    </div>
    <div class="highlight-card">
      <div class="highlight-card-value">3</div>
      <div class="highlight-card-label">Vehicle Classes</div>
    </div>
    <div class="highlight-card">
      <div class="highlight-card-value">7</div>
      <div class="highlight-card-label">Scenarios</div>
    </div>
    <div class="highlight-card">
      <div class="highlight-card-value">4</div>
      <div class="highlight-card-label">Policies Tested</div>
    </div>
  </div>
  <div class="feature-grid">
    <div class="feature-item"><span class="feature-icon">⚙️</span> Event-driven microsimulation</div>
    <div class="feature-item"><span class="feature-icon">🧠</span> Monte Carlo Tree Search</div>
    <div class="feature-item"><span class="feature-icon">🛡️</span> Transposition Table Caching</div>
    <div class="feature-item"><span class="feature-icon">📊</span> Multi-Objective Rollout Scoring</div>
    <div class="feature-item"><span class="feature-icon">🚗</span> Car · Bike · Auto-rickshaw</div>
    <div class="feature-item"><span class="feature-icon">🔄</span> Heuristic Candidate Pruning</div>
  </div>
  <div class="tech-tags">
    <span class="tech-tag">Python</span>
    <span class="tech-tag">Streamlit</span>
    <span class="tech-tag">Plotly</span>
    <span class="tech-tag">scikit-learn</span>
    <span class="tech-tag">NumPy</span>
    <span class="tech-tag">Pandas</span>
    <span class="tech-tag">Self-Generated Data</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline Diagram ──────────────────────────────────────────
st.markdown("""
<div class="pipeline-flow">
  <div class="pipeline-step"><div class="pipeline-icon green">🚂</div><div class="pipeline-label">Train Schedule</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon green">⚙️</div><div class="pipeline-label">Simulator</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon cyan">📐</div><div class="pipeline-label">State Features</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon amber">🧠</div><div class="pipeline-label">ML Surrogate</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon purple">🎰</div><div class="pipeline-label">LinUCB Bandit</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon green">🛡️</div><div class="pipeline-label">Safety Shield</div></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><div class="pipeline-icon amber">📊</div><div class="pipeline-label">Dashboard</div></div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(metric_card("⏱️ Delay Reduction", pct(summary["adaptive_delay_gain_pct"]), "vs free-flow reopening", ""), unsafe_allow_html=True)
with k2:
    st.markdown(metric_card("🚗 Throughput Gain", pct(summary["adaptive_throughput_gain_pct"]), "vehicles cleared", ""), unsafe_allow_html=True)
with k3:
    st.markdown(metric_card("📏 Congestion Cut", pct(summary["adaptive_congestion_gain_pct"]), "peak spillback reduced", ""), unsafe_allow_html=True)
with k4:
    st.markdown(metric_card("🎯 Scenario", scenario.replace("_", " ").title(), summary["best_policy"], summary["scenario_description"]), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════

tabs = st.tabs(["📊 Impact", "📈 Traffic Story", "🎬 Replay", "🔬 Sensitivity", "📚 Learning", "⚙️ Technical"])

# ── Tab 0: Impact ─────────────────────────────────────────────
with tabs[0]:
    st.markdown(f"""<div class="section-note"><strong>Key Finding:</strong> {summary['impact_headline']}</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        bar = px.bar(comp, x="policy", y=["average_waiting_time", "throughput", "max_congestion_length"],
                     barmode="group", title="Policy Comparison — Key Metrics")
        bar.update_layout(xaxis_title="", yaxis_title="Value", legend_title_text="")
        st.plotly_chart(bar, width='stretch')
    with c2:
        imp_long = impr.melt(
            id_vars=["policy"],
            value_vars=["waiting_time_improvement_pct", "throughput_improvement_pct", "congestion_improvement_pct"],
            var_name="metric", value_name="improvement_pct",
        )
        imp_long["metric"] = imp_long["metric"].str.replace("_improvement_pct", "").str.replace("_", " ").str.title()
        fig = px.bar(imp_long, x="metric", y="improvement_pct", color="policy", barmode="group",
                     title="Improvement vs Free Flow Baseline")
        fig.update_layout(xaxis_title="", yaxis_title="Improvement (%)", legend_title_text="")
        st.plotly_chart(fig, width='stretch')

# ── Tab 1: Traffic Story ──────────────────────────────────────
with tabs[1]:
    st.markdown("""<div class="story-card"><div class="story-title">Narrative Flow</div>
    <div class="story-copy">Red bands mark train closures. The critical moment is <strong>after each closure ends</strong> —
    the barrier is open, but congestion persists unless release behaviour is coordinated.</div></div>""", unsafe_allow_html=True)

    queue_frames = []
    for pol, res in results.items():
        h = history_frame(res)
        if h.empty: continue
        m = h.melt(id_vars=["time"], value_vars=["queue_left", "queue_right"], var_name="side", value_name="queue_m")
        m["policy"] = pol
        queue_frames.append(m)
    if queue_frames:
        qdf = pd.concat(queue_frames, ignore_index=True)
        qfig = px.line(qdf, x="time", y="queue_m", color="policy", line_dash="side",
                       title="Queue Evolution · Left vs Right by Policy")
        add_closure_bands(qfig, config.train_closures)
        st.plotly_chart(qfig, width='stretch')

    risk_frames = []
    for pol, res in results.items():
        h = history_frame(res)
        if h.empty: continue
        t = h[["time", "disorder_index", "occupancy_risk"]].copy()
        t["policy"] = pol
        risk_frames.append(t)
    if risk_frames:
        rdf = pd.concat(risk_frames, ignore_index=True)
        rlong = rdf.melt(id_vars=["time", "policy"], value_vars=["disorder_index", "occupancy_risk"],
                         var_name="metric", value_name="value")
        rfig = px.line(rlong, x="time", y="value", color="policy", facet_row="metric",
                       title="Disorder & Risk Trajectory")
        add_closure_bands(rfig, config.train_closures)
        rfig.update_yaxes(matches=None)
        st.plotly_chart(rfig, width='stretch')

# ── Tab 2: Replay ─────────────────────────────────────────────
with tabs[2]:
    pol_opts = list(results)
    default_idx = pol_opts.index(primary_policy) if primary_policy in pol_opts else 0
    chart_pol = st.selectbox("Replay policy", pol_opts, index=default_idx)
    rr = results[chart_pol]
    rh = history_frame(rr)
    ah = actions_frame(rr)
    th = decision_trace_frame(rr)

    if rh.empty:
        st.info("Click **Run Full ML Simulation** in the sidebar to load replay data.")
    else:
        rc, mc = st.columns([1.4, 0.9])
        with rc:
            si = st.slider("Frame", 0, len(rh) - 1, min(70, len(rh) - 1))
            vf = vehicle_frame(rr, si)
            ft = float(rh.iloc[si]["time"])
            if not vf.empty:
                st.plotly_chart(build_snapshot_figure(vf, scenario, seed, ft), width='stretch')
        with mc:
            snap = rr.snapshots[si]
            st.markdown(metric_card("Barrier", "🔴 Closed" if snap.barrier_closed else "🟢 Open",
                                    f"action: {snap.current_action}", ""), unsafe_allow_html=True)
            st.markdown(metric_card("Queue Pressure",
                                    f"{snap.queue_counts['left']} ← | → {snap.queue_counts['right']}",
                                    "left | right vehicles",
                                    f"{snap.queue_lengths['left']:.0f}m | {snap.queue_lengths['right']:.0f}m"), unsafe_allow_html=True)
            st.markdown(metric_card("Risk", num(snap.occupancy_risk),
                                    f"disorder: {num(snap.disorder_index)}",
                                    f"occupancy: {snap.crossing_occupancy} · conflicts: {snap.conflict_count}"), unsafe_allow_html=True)

        sc, lc = st.columns([0.8, 1.2])
        with sc:
            asum = action_summary_frame(rr)
            if not asum.empty:
                fig = px.bar(asum, x="share_pct", y="action", orientation="h", title="Action Mix",
                             color="share_pct", color_continuous_scale=["#55f0a6", "#ffbf4d"])
                st.plotly_chart(fig, width='stretch')
        with lc:
            st.subheader("Decision Log")
            if ah.empty:
                st.caption("No log for this policy.")
            else:
                st.dataframe(ah, width='stretch', height=320)

        if not th.empty:
            ts = th[th["time"] <= ft]
            if ts.empty: ts = th.iloc[[0]]
            tt = float(ts["time"].max())
            at = th[th["time"] == tt].copy()
            chosen = at[at["candidate_action"] == at["chosen_action"]]
            alt = at[at["candidate_action"] != at["chosen_action"]]
            rl, rr2 = st.columns([0.9, 1.1])
            with rl:
                st.subheader("Decision Rationale")
                if not chosen.empty:
                    cr = chosen.iloc[0]
                    st.markdown(metric_card("✅ Chosen", str(cr["chosen_action"]),
                                            f"score: {num(float(cr['score']))}",
                                            f"bandit: {num(float(cr['linucb_bonus']))} · σ: {num(float(cr['utility_std']))}"),
                                unsafe_allow_html=True)
                if not alt.empty:
                    ar = alt.sort_values("score", ascending=False).iloc[0]
                    st.markdown(metric_card("❌ Alternative", str(ar["candidate_action"]),
                                            f"score: {num(float(ar['score']))}",
                                            f"veto: {ar['veto_reason'] or 'none'}"), unsafe_allow_html=True)
            with rr2:
                st.dataframe(at, width='stretch', height=240, hide_index=True)

# ── Tab 3: Sensitivity ────────────────────────────────────────
with tabs[3]:
    st.markdown("""<div class="story-card"><div class="story-title">Parameter Sensitivity Analysis</div>
    <div class="story-copy">Understanding <strong>which parameters drive congestion</strong> is key to effective intervention.
    These pre-computed charts show how delay, throughput, and congestion respond to changes in
    arrival rates, driver aggression, and gate closure duration.</div></div>""", unsafe_allow_html=True)

    arr = read_optional_frame(ROOT / "artifacts" / "analysis" / "arrival_rate_sweep.csv")
    agg = read_optional_frame(ROOT / "artifacts" / "analysis" / "aggression_sweep.csv")
    clo = read_optional_frame(ROOT / "artifacts" / "analysis" / "closure_duration_sweep.csv")

    if arr.empty and agg.empty and clo.empty:
        st.warning("Run `python scripts/precompute_analysis.py` to generate sensitivity data.")
    else:
        if not arr.empty:
            st.subheader("📊 Arrival Rate vs Delay")
            st.caption("How traffic volume from each side impacts waiting time. Adaptive policy absorbs pressure the free-flow approach cannot.")
            al, ar2 = st.columns(2)
            for col, label in [(al, "Free Flow"), (ar2, "Adaptive")]:
                with col:
                    sub = arr[arr["policy"] == label]
                    if not sub.empty:
                        piv = sub.pivot_table(index="arrival_left", columns="arrival_right",
                                              values="average_waiting_time", aggfunc="mean")
                        hm = go.Figure(data=go.Heatmap(
                            z=piv.values, x=[str(c) for c in piv.columns], y=[str(r) for r in piv.index],
                            colorscale=[[0, "#0a1511"], [0.4, "#55f0a6"], [0.7, "#ffbf4d"], [1, "#ff6a6a"]],
                            colorbar=dict(title="Delay (s)", len=0.8),
                            text=piv.values.round(1), texttemplate="%{text}s", textfont=dict(size=11),
                        ))
                        hm.update_layout(title=f"{label}", xaxis_title="Right (veh/min)",
                                         yaxis_title="Left (veh/min)", height=380)
                        st.plotly_chart(hm, width='stretch')

        if not agg.empty:
            st.subheader("🧠 Driver Aggression Impact")
            cat_order = {"driver_mix": ["Calm", "Normal", "Aggressive", "Reckless"]}
            ag1, ag2 = st.columns(2)
            with ag1:
                fig = px.bar(agg, x="driver_mix", y="average_waiting_time", color="policy",
                             barmode="group", title="Wait Time by Driver Mix", category_orders=cat_order)
                fig.update_layout(xaxis_title="Population Profile", yaxis_title="Avg Wait (s)")
                st.plotly_chart(fig, width='stretch')
            with ag2:
                fig = px.bar(agg, x="driver_mix", y="disorder_peak", color="policy",
                             barmode="group", title="Peak Disorder by Driver Mix", category_orders=cat_order)
                fig.update_layout(xaxis_title="Population Profile", yaxis_title="Disorder Index")
                st.plotly_chart(fig, width='stretch')

        if not clo.empty:
            st.subheader("⏱️ Closure Duration Impact")
            cl1, cl2 = st.columns(2)
            with cl1:
                fig = px.line(clo, x="closure_duration_s", y="average_waiting_time", color="policy",
                              markers=True, title="Wait Time vs Closure Duration")
                fig.update_layout(xaxis_title="Duration (s)", yaxis_title="Avg Wait (s)")
                st.plotly_chart(fig, width='stretch')
            with cl2:
                fig = px.line(clo, x="closure_duration_s", y="throughput", color="policy",
                              markers=True, title="Throughput vs Closure Duration")
                fig.update_layout(xaxis_title="Duration (s)", yaxis_title="Throughput (veh/hr)")
                st.plotly_chart(fig, width='stretch')

# ── Tab 4: Learning ───────────────────────────────────────────
with tabs[4]:
    st.subheader("MCTS State Evaluation Cache")
    st.caption("MCTS caches evaluations using a Transposition Table to speed up subsequent rollouts.")
    if not is_fast_mode and "MCTS Rollout" in results:
        pol_obj = getattr(results["MCTS Rollout"], "policy_obj", None)
        if pol_obj and hasattr(pol_obj, "ttable"):
            st.metric("Cached States", len(pol_obj.ttable))
            df = pd.DataFrame(
                [{"State Hash": k, "Score": v} for k, v in list(pol_obj.ttable.items())[:100]]
            )
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("Cache metrics are not available after reloading from cache. Run Full Simulation again.")
    else:
        st.info("Run Full Hybrid Simulation to view cache metrics.")

# ── Tab 5: Technical ──────────────────────────────────────────
with tabs[5]:
    pitch_md = build_pitch_markdown(results, scenario, model_label=model_label)
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Technical Stack")
        st.dataframe(pd.DataFrame([
            {"Layer": "🔧 Simulation", "Details": "Event-driven microsimulation with mixed vehicles & behavior profiles"},
            {"Layer": "📐 Features", "Details": "Queue, disorder, occupancy risk, aggressive share, pressure imbalance"},
            {"Layer": "🧠 Model", "Details": f"Bootstrap surrogate ensemble ({model_label})"},
            {"Layer": "🎰 Learner", "Details": "LinUCB residual policy trained on simulator horizon reward"},
            {"Layer": "🛡️ Shield", "Details": "Vetoes actions with high occupancy risk or starvation tendency"},
        ]), width='stretch', hide_index=True)
        st.subheader("Assumptions")
        st.markdown(build_assumptions_markdown())
    with t2:
        st.subheader("Policy Comparison")
        st.dataframe(comp, width='stretch', hide_index=True)
        st.subheader("Improvement Table")
        st.dataframe(impr, width='stretch', hide_index=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📄 Download Report", data=pitch_md,
                           file_name=f"quarryflow_{scenario}_report.md", mime="text/markdown",
                           width='stretch')
    with d2:
        st.download_button("📋 Download Assumptions", data=build_assumptions_markdown(),
                           file_name="quarryflow_assumptions.md", mime="text/markdown",
                           width='stretch')
