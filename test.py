from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[0]
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


st.set_page_config(
    page_title="QuarryFlow Crossing",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ROOT / "app" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def pct(value: float) -> str:
    return f"{value:.1f}%"


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
            x0=start,
            x1=end,
            fillcolor="rgba(255, 106, 106, 0.12)",
            line_width=0,
            layer="below",
        )
    return fig


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbf8ea"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(85, 240, 166, 0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(85, 240, 166, 0.12)")
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
        vehicles,
        x="x",
        y="y",
        color="vehicle_class",
        symbol="side",
        hover_data=["vehicle_id", "speed", "progress"],
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "#08110f"}})
    fig.update_layout(
        title=f"Crossing Replay at t={time_value:.1f}s",
        xaxis_title="Distance from crossing center (m)",
        yaxis_title="Approach lanes / lateral spread",
        height=500,
        xaxis={"range": [-approach - 12, approach + config.recovery_length + 12]},
        yaxis={"range": [-3.4, 3.4]},
    )
    fig.add_hrect(y0=0.3, y1=2.6, fillcolor="rgba(85, 240, 166, 0.05)", line_width=0)
    fig.add_hrect(y0=-2.6, y1=-0.3, fillcolor="rgba(85, 240, 166, 0.05)", line_width=0)
    fig.add_vrect(x0=-half_box, x1=half_box, fillcolor="rgba(255, 191, 77, 0.12)", line_width=0)
    fig.add_vline(x=0, line_color="rgba(255,191,77,0.65)", line_dash="dash")
    fig.add_annotation(x=0, y=3.0, text="Rail Crossing Box", showarrow=False, font={"color": "#ffbf4d"})
    fig.add_annotation(x=-approach + 14, y=2.9, text="Left Approach", showarrow=False, font={"color": "#55f0a6"})
    fig.add_annotation(x=approach - 14, y=-2.9, text="Right Approach", showarrow=False, font={"color": "#55f0a6"})
    return style_figure(fig)


def load_run(scenario: str, seed: int, model_path: str, ensemble_path: str, controller_path: str) -> None:
    model_path_arg = model_path if model_path and Path(model_path).exists() else None
    ensemble_path_arg = ensemble_path if ensemble_path and Path(ensemble_path).exists() else None
    controller_path_arg = controller_path if controller_path and Path(controller_path).exists() else None
    results = run_policy_suite(
        scenario,
        seed=seed,
        model_path=model_path_arg,
        ensemble_path=ensemble_path_arg,
        controller_path=controller_path_arg,
        record_history=True,
    )
    st.session_state["quarryflow_results"] = results
    st.session_state["quarryflow_summary"] = judge_summary(results, scenario)
    st.session_state["quarryflow_scenario"] = scenario
    st.session_state["quarryflow_seed"] = seed
    st.session_state["quarryflow_model_path"] = model_path_arg
    st.session_state["quarryflow_ensemble_path"] = ensemble_path_arg
    st.session_state["quarryflow_controller_path"] = controller_path_arg
    if ensemble_path_arg:
        from quarryflow.model import BootstrapSurrogateEnsemble  # noqa: E402

        st.session_state["quarryflow_model_label"] = BootstrapSurrogateEnsemble.load(ensemble_path_arg).backend
    elif model_path_arg:
        st.session_state["quarryflow_model_label"] = SurrogateModel.load(model_path_arg).backend
    else:
        st.session_state["quarryflow_model_label"] = "heuristic-only adaptive control"
    if controller_path_arg:
        _, _, metadata = load_hybrid_controller(controller_path_arg)
        st.session_state["quarryflow_controller_metadata"] = metadata
    else:
        st.session_state["quarryflow_controller_metadata"] = {}


load_css()

st.markdown(
    """
    <div class="hero-shell">
      <div class="eyebrow">KnowledgeQuarry ML Track</div>
      <div class="hero-title">QuarryFlow Crossing</div>
      <div class="hero-copy">
        A behavior-aware railway-crossing surge optimizer that shows why traffic stays jammed after the train is gone,
        then demonstrates how adaptive staged release cuts delay and improves flow without changing infrastructure.
      </div>
      <div class="tag-row">
        <span class="tag-chip">Mixed Traffic Simulation</span>
        <span class="tag-chip">Adaptive Control</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Run Setup")
    scenario = st.selectbox(
        "Scenario preset",
        ["light", "peak", "chaotic", "peak_left_skew", "peak_right_skew", "chaotic_aggressive", "chaotic_long_gate"],
        index=1,
    )
    seed = st.number_input("Random seed", min_value=1, max_value=999, value=11, step=1)
    ensemble_path = st.text_input(
        "Hybrid ensemble path",
        value=str(ROOT / "artifacts" / "models" / "surrogate_ensemble.pkl"),
    )
    controller_path = st.text_input(
        "Hybrid controller path",
        value=str(ROOT / "artifacts" / "models" / "hybrid_controller.json"),
    )
    model_path = st.text_input(
        "Legacy model path",
        value=str(ROOT / "artifacts" / "models" / "surrogate.pkl"),
    )
    run_button = st.button("Run Simulation", type="primary", width="stretch")

if run_button or "quarryflow_results" not in st.session_state:
    load_run(scenario, int(seed), model_path, ensemble_path, controller_path)

results = st.session_state["quarryflow_results"]
summary = st.session_state["quarryflow_summary"]
scenario = st.session_state["quarryflow_scenario"]
seed = st.session_state["quarryflow_seed"]
model_label = st.session_state["quarryflow_model_label"]
controller_metadata = st.session_state.get("quarryflow_controller_metadata", {})
config = build_scenario(scenario, seed=seed)
learning_curve = read_optional_frame(ROOT / "artifacts" / "eval" / "learning_curve.csv")
holdout_summary = read_optional_frame(ROOT / "artifacts" / "eval" / "holdout_summary.csv")
validation_summary = read_optional_frame(ROOT / "artifacts" / "eval" / "validation_summary.csv")
primary_policy = "Hybrid Adaptive" if "Hybrid Adaptive" in results else "Legacy Adaptive"

comparison = comparison_frame(results)
improvements = improvement_frame(results)
adaptive_result = results[primary_policy]
adaptive_history = history_frame(adaptive_result)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(
        metric_card(
            "Delay Reduction",
            pct(summary["adaptive_delay_gain_pct"]),
            "vs free-flow reopening",
            "",
        ),
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    st.markdown(
        metric_card(
            "Throughput Gain",
            pct(summary["adaptive_throughput_gain_pct"]),
            "more vehicles cleared",
            "",
        ),
        unsafe_allow_html=True,
    )
with metric_cols[2]:
    st.markdown(
        metric_card(
            "Peak Congestion Cut",
            pct(summary["adaptive_congestion_gain_pct"]),
            "less spillback",
            "",
        ),
        unsafe_allow_html=True,
    )
with metric_cols[3]:
    st.markdown(
        metric_card(
            "Live Scenario",
            scenario.title(),
            summary["best_policy"],
            summary["scenario_description"],
        ),
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Impact Summary", "Traffic Story", "Crossing Replay", "Learning Under Constraints", "Technical Details"])

with tabs[0]:
    st.markdown(
        f"""
        <div class="section-note">
          <strong>Impact Summary:</strong> {summary['impact_headline']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.15, 1.0])
    with top_left:
        bar = px.bar(
            comparison,
            x="policy",
            y=["average_waiting_time", "throughput", "max_congestion_length"],
            barmode="group",
            title="Policy Comparison",
        )
        style_figure(bar)
        st.plotly_chart(bar, width="stretch")

    with top_right:
        improvement_long = improvements.melt(
            id_vars=["policy"],
            value_vars=[
                "waiting_time_improvement_pct",
                "throughput_improvement_pct",
                "congestion_improvement_pct",
            ],
            var_name="metric",
            value_name="improvement_pct",
        )
        fig = px.bar(
            improvement_long,
            x="metric",
            y="improvement_pct",
            color="policy",
            barmode="group",
            title="Improvement vs Free Flow",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Percent")
        style_figure(fig)
        st.plotly_chart(fig, width="stretch")

with tabs[1]:
    st.markdown(
        """
        <div class="story-card">
          <div class="story-title">Narrative Flow</div>
          <div class="story-copy">
            Red bands show train closures. The key moment is after each red zone ends:
            the barrier is open, but congestion persists unless release behavior is coordinated.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    queue_frames = []
    for policy_name, result in results.items():
        history = history_frame(result)
        if history.empty:
            continue
        melted = history.melt(
            id_vars=["time"],
            value_vars=["queue_left", "queue_right"],
            var_name="queue_side",
            value_name="queue_length",
        )
        melted["policy"] = policy_name
        queue_frames.append(melted)
    queue_frame = pd.concat(queue_frames, ignore_index=True)
    queue_fig = px.line(
        queue_frame,
        x="time",
        y="queue_length",
        color="policy",
        line_dash="queue_side",
        title="Queue Evolution by Policy and Side",
    )
    add_closure_bands(queue_fig, config.train_closures)
    style_figure(queue_fig)
    st.plotly_chart(queue_fig, width="stretch")

    risk_frames = []
    for policy_name, result in results.items():
        history = history_frame(result)
        if history.empty:
            continue
        trimmed = history[["time", "disorder_index", "occupancy_risk"]].copy()
        trimmed["policy"] = policy_name
        risk_frames.append(trimmed)
    risk_frame = pd.concat(risk_frames, ignore_index=True)
    risk_long = risk_frame.melt(
        id_vars=["time", "policy"],
        value_vars=["disorder_index", "occupancy_risk"],
        var_name="metric",
        value_name="value",
    )
    risk_fig = px.line(
        risk_long,
        x="time",
        y="value",
        color="policy",
        facet_row="metric",
        title="Disorder and Risk Trajectory",
    )
    add_closure_bands(risk_fig, config.train_closures)
    style_figure(risk_fig)
    risk_fig.update_yaxes(matches=None)
    st.plotly_chart(risk_fig, width="stretch")

with tabs[2]:
    policy_options = list(results)
    default_index = policy_options.index(primary_policy) if primary_policy in policy_options else 0
    chart_policy = st.selectbox("Replay policy", policy_options, index=default_index)
    replay_result = results[chart_policy]
    replay_history = history_frame(replay_result)
    action_history = actions_frame(replay_result)
    trace_history = decision_trace_frame(replay_result)

    if replay_history.empty:
        st.warning("No replay history available for this run.")
    else:
        replay_col, meta_col = st.columns([1.4, 0.9])
        with replay_col:
            step_index = st.slider(
                "Simulation frame",
                min_value=0,
                max_value=len(replay_history) - 1,
                value=min(70, len(replay_history) - 1),
            )
            vehicles = vehicle_frame(replay_result, step_index)
            frame_time = float(replay_history.iloc[step_index]["time"])
            if not vehicles.empty:
                st.plotly_chart(
                    build_snapshot_figure(vehicles, scenario, seed, frame_time),
                    width="stretch",
                )

        with meta_col:
            snapshot = replay_result.snapshots[step_index]
            st.markdown(
                metric_card(
                    "Barrier State",
                    "Closed" if snapshot.barrier_closed else "Open",
                    f"action: {snapshot.current_action}",
                    "",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card(
                    "Queue Pressure",
                    f"{snapshot.queue_counts['left']} | {snapshot.queue_counts['right']}",
                    "left | right vehicles",
                    f"Queue length: {snapshot.queue_lengths['left']:.1f}m | {snapshot.queue_lengths['right']:.1f}m",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card(
                    "Risk State",
                    num(snapshot.occupancy_risk),
                    f"disorder: {num(snapshot.disorder_index)}",
                    f"Crossing occupancy: {snapshot.crossing_occupancy}, conflicts: {snapshot.conflict_count}",
                ),
                unsafe_allow_html=True,
            )

        summary_col, log_col = st.columns([0.8, 1.2])
        with summary_col:
            action_summary = action_summary_frame(replay_result)
            if not action_summary.empty:
                fig = px.bar(
                    action_summary,
                    x="share_pct",
                    y="action",
                    orientation="h",
                    title="Action Mix",
                    color="share_pct",
                    color_continuous_scale=["#55f0a6", "#ffbf4d"],
                )
                style_figure(fig)
                st.plotly_chart(fig, width="stretch")

        with log_col:
            st.subheader("Decision Log")
            if action_history.empty:
                st.caption("No adaptive action log for this policy.")
            else:
                st.dataframe(action_history, width="stretch", height=320)

        if trace_history.empty:
            st.caption("No decision-trace rationale available for this policy.")
        else:
            trace_slice = trace_history[trace_history["time"] <= frame_time]
            if trace_slice.empty:
                trace_slice = trace_history.iloc[[0]]
            trace_time = float(trace_slice["time"].max())
            active_trace = trace_history[trace_history["time"] == trace_time].copy()
            chosen = active_trace[active_trace["candidate_action"] == active_trace["chosen_action"]]
            alt = active_trace[active_trace["candidate_action"] != active_trace["chosen_action"]]
            rationale_left, rationale_right = st.columns([0.9, 1.1])
            with rationale_left:
                st.subheader("Decision Rationale")
                if not chosen.empty:
                    chosen_row = chosen.iloc[0]
                    st.markdown(
                        metric_card(
                            "Chosen Action",
                            str(chosen_row["chosen_action"]),
                            f"score: {num(float(chosen_row['score']))}",
                            f"bandit bonus: {num(float(chosen_row['linucb_bonus']))}, uncertainty: {num(float(chosen_row['utility_std']))}",
                        ),
                        unsafe_allow_html=True,
                    )
                if not alt.empty:
                    alt_row = alt.sort_values("score", ascending=False).iloc[0]
                    st.markdown(
                        metric_card(
                            "Top Alternative",
                            str(alt_row["candidate_action"]),
                            f"score: {num(float(alt_row['score']))}",
                            f"veto: {alt_row['veto_reason'] or 'none'}",
                        ),
                        unsafe_allow_html=True,
                    )
            with rationale_right:
                st.dataframe(active_trace, width="stretch", height=240, hide_index=True)

with tabs[3]:
    st.subheader("Learning Curve")
    if learning_curve.empty:
        st.caption(
            "Run `python scripts/train_hybrid_controller.py --profile fast --scenarios light peak chaotic --stage-passes 1 --n-models 2` to generate learning artifacts."
        )
    else:
        curve_fig = px.line(
            learning_curve,
            x="iteration",
            y=["validation_hybrid_reward", "validation_legacy_reward", "best_validation_reward"],
            markers=True,
            title="Curriculum Validation Reward",
        )
        style_figure(curve_fig)
        st.plotly_chart(curve_fig, width="stretch")
        st.dataframe(learning_curve, width="stretch", hide_index=True)

    holdout_left, holdout_right = st.columns([1.0, 1.0])
    with holdout_left:
        st.subheader("Validation Summary")
        if validation_summary.empty:
            st.caption("No validation summary found.")
        else:
            st.dataframe(validation_summary, width="stretch", hide_index=True)
    with holdout_right:
        st.subheader("Holdout Summary")
        if holdout_summary.empty:
            st.caption("No holdout summary found.")
        else:
            st.dataframe(holdout_summary, width="stretch", hide_index=True)

    if controller_metadata:
        st.subheader("Hybrid Gate")
        st.json(controller_metadata.get("gate_metrics", {}))
        st.caption(
            "Hybrid default enabled."
            if controller_metadata.get("hybrid_default_ok")
            else "Hybrid retained as visible technical-depth mode; legacy adaptive remains the safer default."
        )

with tabs[4]:
    pitch_markdown = build_pitch_markdown(results, scenario, model_label=model_label)
    depth_left, depth_right = st.columns([1.0, 1.0])

    with depth_left:
        st.subheader("Technical Stack")
        stack = pd.DataFrame(
            [
                {"Layer": "Simulation", "Details": "Event-driven railway-crossing microsimulation with mixed vehicles and behavior profiles"},
                {"Layer": "State Features", "Details": "Queue lengths, disorder index, occupancy risk, aggressive share, and pressure imbalance"},
                {"Layer": "Counterfactual Model", "Details": f"Bootstrap surrogate ensemble using {model_label}"},
                {"Layer": "Residual Learner", "Details": "LinUCB residual policy trained on simulator-derived horizon reward"},
                {"Layer": "Safety Shield", "Details": "Vetoes actions predicted to create high occupancy risk or unfair starvation"},
            ]
        )
        st.dataframe(stack, width="stretch", hide_index=True)

        st.subheader("Assumptions & Data Source")
        st.markdown(build_assumptions_markdown())

    with depth_right:
        st.subheader("Comparison Table")
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.subheader("Improvement Table")
        st.dataframe(improvements, width="stretch", hide_index=True)

    report_col, assumptions_col = st.columns([1.0, 1.0])
    with report_col:
        st.subheader("Download Report")
        st.download_button(
            "Download markdown report",
            data=pitch_markdown,
            file_name=f"quarryflow_{scenario}_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with assumptions_col:
        st.subheader("Download Assumptions")
        st.download_button(
            "Download assumptions sheet",
            data=build_assumptions_markdown(),
            file_name="quarryflow_assumptions.md",
            mime="text/markdown",
            width="stretch",
        )
    st.code(
        """flowchart LR
A["Train Closure Schedule"] --> B["Railway-Crossing Simulator"]
B --> C["State Snapshot Features"]
C --> D["Surrogate Ensemble"]
D --> E["LinUCB Residual + Safety Shield"]
E --> B
B --> F["Metrics: Delay, Throughput, Congestion, Risk"]
F --> G["User Dashboard"]""",
        language="mermaid",
    )
