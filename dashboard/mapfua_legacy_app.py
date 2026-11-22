from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smart_warehouse.control import write_order_surge_command

COMPARISON_STATE_PATH = ROOT_DIR / "runtime" / "comparison_state.json"
CONTROL_PATH = ROOT_DIR / "runtime" / "control.json"
MODEL_METRICS_PATH = ROOT_DIR / "models" / "model_metrics.json"
EXPERIMENT_SUMMARY_PATH = ROOT_DIR / "runtime" / "experiment_summary.csv"
EXPERIMENT_IMPROVEMENTS_PATH = ROOT_DIR / "runtime" / "experiment_improvements.json"
FOOTWEAR_MANIFEST_PATH = ROOT_DIR / "data" / "footwear_manifest.json"

st.set_page_config(page_title="Adaptive Warehouse Command Center", layout="wide")
st.title("Adaptive Multi-Robot Warehouse Command Center")
st.caption(
    "Same Footwear task stream · Reactive baseline vs threshold-based Adaptive Hybrid · "
    "plain-language live explanation"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def zone_matrix(values: dict) -> list[list[float]]:
    return [
        [float(values.get("A", 0)), float(values.get("B", 0))],
        [float(values.get("C", 0)), float(values.get("D", 0))],
    ]


def heatmap_figure(values: dict, title: str, colorbar_title: str) -> go.Figure:
    matrix = zone_matrix(values)
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["West", "East"],
            y=["North", "South"],
            text=[
                [f"Zone A<br>{matrix[0][0]:.1f}", f"Zone B<br>{matrix[0][1]:.1f}"],
                [f"Zone C<br>{matrix[1][0]:.1f}", f"Zone D<br>{matrix[1][1]:.1f}"],
            ],
            texttemplate="%{text}",
            colorscale="YlOrRd",
            colorbar={"title": colorbar_title},
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(title=title, height=330, margin={"l": 25, "r": 25, "t": 55, "b": 25})
    return figure


def comparison_metrics_frame(state: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for label, key in (("Reactive baseline", "reactive"), ("Adaptive Hybrid", "hybrid")):
        metrics = state.get(key, {}).get("metrics", {})
        rows.append(
            {
                "system": label,
                "completed": metrics.get("completed_tasks", 0),
                "waiting": metrics.get("waiting_tasks", 0),
                "throughput/h": metrics.get("throughput_per_hour", 0),
                "average wait": metrics.get("average_wait_time", 0),
                "P95 wait": metrics.get("p95_wait_time", 0),
                "task distance": metrics.get("task_distance", 0),
                "reposition distance": metrics.get("reposition_distance", 0),
                "total distance": metrics.get("total_distance", 0),
            }
        )
    return pd.DataFrame(rows)


def policy_signal_frame(state: dict) -> pd.DataFrame:
    decision = state.get("hybrid", {}).get("controller", {}).get("decision") or {}
    signals = decision.get("signals") or {}
    rows = [
        ("Suitability score", decision.get("score", 0)),
        ("Advance-notice enter threshold", signals.get("enter_threshold", 0)),
        ("Regular forecast enter threshold", signals.get("regular_enter_threshold", 0)),
        ("Required enter threshold now", signals.get("required_enter_threshold", 0)),
        ("Exit threshold", signals.get("exit_threshold", 0)),
        ("Forecast hotspot", signals.get("top_zone", "-")),
        ("Hotspot share", signals.get("top_zone_share", 0)),
        ("Top-two gap", signals.get("forecast_gap", 0)),
        ("Idle robots", signals.get("idle_robots", 0)),
        ("Waiting tasks", signals.get("waiting_tasks", 0)),
        ("Queue per robot", signals.get("queue_per_robot", 0)),
        ("Advance notice active", signals.get("forecast_notice_active", False)),
    ]
    return pd.DataFrame(rows, columns=["signal", "value"])


@st.fragment(run_every="1s")
def live_comparison() -> None:
    state = load_json(COMPARISON_STATE_PATH)
    if not state:
        st.info(
            "Start the comparison simulator after preparing the Footwear data: "
            "`python run_comparison.py`"
        )
        return

    scenario = state.get("scenario", {})
    situation = state.get("situation", {})
    hybrid_controller = state.get("hybrid", {}).get("controller", {})
    active_policy = hybrid_controller.get("active_policy", "REACTIVE")

    source_col, button_col = st.columns([0.72, 0.28])
    with source_col:
        st.success(
            f"Fair comparison: one shared task stream · {scenario.get('description', 'unknown source')} · "
            f"time basis: {scenario.get('time_basis', 'unknown')}"
        )
    with button_col:
        if st.button("Forecast a Zone C order surge", type="primary", use_container_width=True):
            command_id = write_order_surge_command(CONTROL_PATH, "C", 20)
            st.toast(f"Order surge forecast queued ({command_id[:8]})")

    level = str(situation.get("level", "NORMAL"))
    if level == "OVERLOADED":
        st.error(f"**{level}: {situation.get('headline', '')}** — {situation.get('summary', '')}")
    elif level == "BUSY":
        st.warning(f"**{level}: {situation.get('headline', '')}** — {situation.get('summary', '')}")
    else:
        st.info(f"**{level}: {situation.get('headline', '')}** — {situation.get('summary', '')}")

    policy_left, policy_right = st.columns([0.34, 0.66])
    with policy_left:
        st.metric("Adaptive Hybrid active policy", active_policy)
        st.caption(
            "Predictive means idle robots may be repositioned before demand arrives. "
            "Reactive means robots only respond to released tasks."
        )
    with policy_right:
        st.markdown(f"**Why this mode?** {situation.get('policy_reason', '-')}")
        st.markdown(f"**Current action:** {situation.get('current_action', '-')}")

    improvements = state.get("improvements", {})
    cards = st.columns(5)
    cards[0].metric("Average wait reduction", f"{improvements.get('wait_time_reduction_pct', 0):+.1f}%")
    cards[1].metric("P95 wait reduction", f"{improvements.get('p95_wait_reduction_pct', 0):+.1f}%")
    cards[2].metric("Throughput increase", f"{improvements.get('throughput_increase_pct', 0):+.1f}%")
    cards[3].metric("Task travel reduction", f"{improvements.get('task_distance_reduction_pct', 0):+.1f}%")
    cards[4].metric("Simulation tick", state.get("tick", 0))

    metrics_df = comparison_metrics_frame(state)
    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Same tasks, two control strategies")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        chart_columns = ["completed", "waiting", "throughput/h"]
        chart_data = metrics_df.melt(
            id_vars="system",
            value_vars=chart_columns,
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(
            px.bar(chart_data, x="metric", y="value", color="system", barmode="group"),
            use_container_width=True,
        )
    with right:
        st.subheader("Threshold decision signals")
        st.dataframe(policy_signal_frame(state), use_container_width=True, hide_index=True)
        st.caption(
            "Hard safety gates force Reactive when backlog is high, idle capacity is low, "
            "or the forecast is too weak or too evenly spread."
        )

    st.subheader("What the map colors mean")
    st.markdown(
        "**Red square:** waiting task · **Blue robot:** idle · **Green robot/line:** moving to a task · "
        "**Purple robot/line:** predictive repositioning · **Orange robot:** processing · "
        "**Dark block:** shelf/obstacle. A darker zone background means higher current or forecast pressure."
    )

    heat_left, heat_right = st.columns(2)
    with heat_left:
        reactive_heat = state.get("reactive", {}).get("demand_heatmap", {})
        st.plotly_chart(
            heatmap_figure(reactive_heat, "Reactive: current queue pressure", "Pressure"),
            use_container_width=True,
        )
    with heat_right:
        hybrid_heat = state.get("hybrid", {}).get("demand_heatmap", {})
        st.plotly_chart(
            heatmap_figure(hybrid_heat, "Adaptive Hybrid: next-window demand", "Forecast"),
            use_container_width=True,
        )

    st.subheader("Recent system decisions")
    event_df = pd.DataFrame(state.get("event_log", []))
    if event_df.empty:
        st.write("No events yet.")
    else:
        visible = [column for column in ["tick", "minute", "type", "message", "zone_id", "count"] if column in event_df]
        st.dataframe(event_df[visible], use_container_width=True, hide_index=True)


live_comparison()

st.divider()
experiment_left, experiment_right = st.columns([1.25, 0.75])
with experiment_left:
    st.subheader("Multi-seed experiment average")
    summary = safe_read_csv(EXPERIMENT_SUMMARY_PATH)
    if summary.empty:
        st.info("Run `python -m scripts.run_experiment` to calculate the averaged result.")
    else:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        value_columns = [
            column
            for column in ["throughput_per_hour", "average_wait_time", "p95_wait_time", "total_distance"]
            if column in summary.columns
        ]
        if value_columns:
            long_summary = summary.melt(
                id_vars="mode", value_vars=value_columns, var_name="metric", value_name="value"
            )
            st.plotly_chart(
                px.bar(long_summary, x="metric", y="value", color="mode", barmode="group"),
                use_container_width=True,
            )
with experiment_right:
    st.subheader("Experiment improvements")
    experiment_improvements = load_json(EXPERIMENT_IMPROVEMENTS_PATH)
    if experiment_improvements:
        st.json(experiment_improvements)
    else:
        st.write("No averaged experiment result yet.")

st.divider()
model_left, source_right = st.columns(2)
with model_left:
    st.subheader("Prediction model comparison")
    metadata = load_json(MODEL_METRICS_PATH)
    results = pd.DataFrame(metadata.get("results", []))
    if results.empty:
        st.info("Prepare the Footwear data and run `python -m scripts.train_models`.")
    else:
        st.write(f"Selected model: **{metadata.get('best_model', '-')}**")
        st.dataframe(results, use_container_width=True, hide_index=True)
        result_long = results.melt(
            id_vars="model", value_vars=["mae", "rmse"], var_name="metric", value_name="value"
        )
        st.plotly_chart(
            px.bar(result_long, x="model", y="value", color="metric", barmode="group"),
            use_container_width=True,
        )
with source_right:
    st.subheader("Footwear data provenance and assumptions")
    manifest = load_json(FOOTWEAR_MANIFEST_PATH)
    if manifest:
        st.json(manifest)
    else:
        st.info(
            "Download and extract the Mendeley dataset, then run "
            "`python -m scripts.prepare_footwear_data --dataset-dir <folder>`."
        )
