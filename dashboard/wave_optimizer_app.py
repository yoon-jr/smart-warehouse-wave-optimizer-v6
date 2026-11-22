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

from smart_warehouse.wave_optimization import (
    RouteOptimizationConfig,
    canonical_wave_id,
    expand_route_path,
    route_for_wave,
)
from smart_warehouse.warehouse import WarehouseMap

TASKS_PATH = ROOT_DIR / "data" / "footwear_replay_tasks.csv"
DATASET_PATH = ROOT_DIR / "data" / "wave_optimization_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "runtime" / "wave_optimizer"
REPORT_PATH = OUTPUT_DIR / "wave_optimization_report.json"
MODEL_LEADERBOARD_PATH = OUTPUT_DIR / "wave_model_leaderboard.csv"
POLICY_LEADERBOARD_PATH = OUTPUT_DIR / "wave_adaptive_policy_leaderboard.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "wave_test_predictions.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "wave_feature_importance.csv"
EXPLANATIONS_PATH = OUTPUT_DIR / "wave_explanations.csv"

st.set_page_config(page_title="WareFlow AI Digital Twin", layout="wide")
st.title("WareFlow AI · WMS Wave Optimization Digital Twin")
st.caption(
    "Actual Footwear WMS pick waves · chronological holdout · model-selected optimization · "
    "same pickup stops, shorter route"
)


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def route_figure(
    warehouse: WarehouseMap,
    original_path: list[tuple[int, int]],
    optimized_path: list[tuple[int, int]],
    stops: tuple[tuple[int, int], ...],
    depot: tuple[int, int],
) -> go.Figure:
    figure = go.Figure()
    obstacle_x = [point[0] for point in sorted(warehouse.obstacles)]
    obstacle_y = [point[1] for point in sorted(warehouse.obstacles)]
    figure.add_trace(
        go.Scatter(
            x=obstacle_x,
            y=obstacle_y,
            mode="markers",
            name="Shelves",
            marker={"symbol": "square", "size": 13, "color": "#374151"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in original_path],
            y=[point[1] for point in original_path],
            mode="lines",
            name="Original WMS order",
            line={"width": 4, "color": "#ef4444"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in optimized_path],
            y=[point[1] for point in optimized_path],
            mode="lines",
            name="NN + 2-opt",
            line={"width": 4, "color": "#10b981"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in stops],
            y=[point[1] for point in stops],
            mode="markers+text",
            name="Pickup stops",
            text=[str(index + 1) for index in range(len(stops))],
            textposition="top center",
            marker={"size": 10, "color": "#2563eb"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[depot[0]],
            y=[depot[1]],
            mode="markers+text",
            name="Depot",
            text=["DEPOT"],
            textposition="bottom center",
            marker={"size": 16, "symbol": "star", "color": "#f59e0b"},
        )
    )
    figure.update_layout(
        height=650,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        xaxis={"range": [-1, warehouse.width], "dtick": 1, "title": "Grid X"},
        yaxis={
            "range": [warehouse.height, -1],
            "dtick": 1,
            "title": "Grid Y",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": 1.07},
        title="Before / After route on the same warehouse map",
    )
    return figure


def animated_route_figure(
    warehouse: WarehouseMap,
    original_path: list[tuple[int, int]],
    optimized_path: list[tuple[int, int]],
    stops: tuple[tuple[int, int], ...],
    depot: tuple[int, int],
) -> go.Figure:
    """Animate original and optimized routes at one grid step per frame.

    This is a route-replay digital twin rather than a collision-aware multi-robot
    simulator.  Both markers advance one grid cell per frame, so an optimized
    route that reaches the depot earlier visibly represents fewer travel steps.
    """
    original = original_path or [depot]
    optimized = optimized_path or [depot]
    frame_count = max(len(original), len(optimized))

    figure = go.Figure()
    obstacle_x = [point[0] for point in sorted(warehouse.obstacles)]
    obstacle_y = [point[1] for point in sorted(warehouse.obstacles)]
    figure.add_trace(
        go.Scatter(
            x=obstacle_x,
            y=obstacle_y,
            mode="markers",
            name="Shelves",
            marker={"symbol": "square", "size": 13, "color": "#374151"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in original],
            y=[point[1] for point in original],
            mode="lines",
            name="Original route",
            line={"width": 3, "color": "rgba(239,68,68,0.38)"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in optimized],
            y=[point[1] for point in optimized],
            mode="lines",
            name="Optimized route",
            line={"width": 3, "color": "rgba(16,185,129,0.45)"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point[0] for point in stops],
            y=[point[1] for point in stops],
            mode="markers",
            name="Pickup stops",
            marker={"size": 9, "color": "#2563eb"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[depot[0]],
            y=[depot[1]],
            mode="markers+text",
            name="Depot",
            text=["DEPOT"],
            textposition="bottom center",
            marker={"size": 16, "symbol": "star", "color": "#f59e0b"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[original[0][0]],
            y=[original[0][1]],
            mode="markers",
            name="Original picker",
            marker={"size": 22, "symbol": "circle", "color": "#ef4444", "line": {"width": 2, "color": "white"}},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[optimized[0][0]],
            y=[optimized[0][1]],
            mode="markers",
            name="AI optimized picker",
            marker={"size": 22, "symbol": "diamond", "color": "#10b981", "line": {"width": 2, "color": "white"}},
        )
    )

    frames: list[go.Frame] = []
    for step in range(frame_count):
        original_position = original[min(step, len(original) - 1)]
        optimized_position = optimized[min(step, len(optimized) - 1)]
        frames.append(
            go.Frame(
                name=str(step),
                data=[
                    go.Scatter(x=[original_position[0]], y=[original_position[1]]),
                    go.Scatter(x=[optimized_position[0]], y=[optimized_position[1]]),
                ],
                traces=[5, 6],
            )
        )
    figure.frames = frames

    slider_steps = [
        {
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
            "label": str(step),
            "method": "animate",
        }
        for step in range(0, frame_count, max(1, frame_count // 12))
    ]
    if slider_steps[-1]["label"] != str(frame_count - 1):
        slider_steps.append(
            {
                "args": [[str(frame_count - 1)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                "label": str(frame_count - 1),
                "method": "animate",
            }
        )

    figure.update_layout(
        height=680,
        margin={"l": 20, "r": 20, "t": 65, "b": 20},
        xaxis={"range": [-1, warehouse.width], "dtick": 1, "title": "Grid X"},
        yaxis={
            "range": [warehouse.height, -1],
            "dtick": 1,
            "title": "Grid Y",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": 1.08},
        title="Animated route race — one grid step per frame",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.13,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 55, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Grid step: "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )
    return figure


report = read_json(REPORT_PATH)
leaderboard = read_csv(MODEL_LEADERBOARD_PATH)
policy = read_csv(POLICY_LEADERBOARD_PATH)
predictions = read_csv(PREDICTIONS_PATH)
importance = read_csv(IMPORTANCE_PATH)
explanations = read_csv(EXPLANATIONS_PATH)
tasks = read_csv(TASKS_PATH)

if not report or leaderboard.empty or predictions.empty or tasks.empty:
    st.warning(
        "Pipeline outputs are missing. Run: `python -m scripts.run_wave_optimizer_pipeline --coverage 0.20`"
    )
    st.stop()

selected_model = str(report.get("selected_model", "unknown"))
adaptive_report = report.get("adaptive_policy", {})
test_eval = report.get("test_evaluation", {})

cards = st.columns(6)
cards[0].metric("Selected model", selected_model)
cards[1].metric("Test MAE", f"{float(test_eval.get('mae', 0)):.2f} pp")
cards[2].metric(
    "ML captured savings",
    f"{float(adaptive_report.get('ml_captured_savings_pct', 0)):.1f}%",
)
cards[3].metric(
    "End-to-end distance reduction",
    f"{float(adaptive_report.get('ml_end_to_end_distance_reduction_pct', 0)):.1f}%",
)
cards[4].metric(
    "Optimization work avoided",
    f"{float(adaptive_report.get('optimization_workload_reduction_pct', 0)):.1f}%",
)
cards[5].metric("Held-out test waves", int(report.get("split", {}).get("test_waves", 0)))

st.divider()
left, right = st.columns([1.05, 0.95])
with left:
    st.subheader("Model selection on chronological validation set")
    visible = [
        column
        for column in [
            "model",
            "mae",
            "rmse",
            "r2",
            "spearman",
            "top20_recall_pct",
            "captured_savings_pct_at_20",
            "inference_ms_per_wave",
        ]
        if column in leaderboard.columns
    ]
    st.dataframe(leaderboard[visible], use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            leaderboard.sort_values("mae"),
            x="model",
            y="mae",
            title="Validation MAE — lower is better",
        ),
        use_container_width=True,
    )
with right:
    st.subheader("Adaptive strategy on unseen test waves")
    st.dataframe(policy, use_container_width=True, hide_index=True)
    policy_chart = policy[
        ~policy["method"].isin(["original_no_optimization"])
    ].copy()
    st.plotly_chart(
        px.bar(
            policy_chart,
            x="method",
            y="captured_savings_pct",
            color="coverage_pct",
            title="Savings captured under the optimization budget",
        ),
        use_container_width=True,
    )

st.divider()
st.subheader("Digital twin: select a held-out WMS wave")
ranked = predictions.sort_values("predicted_saving_pct", ascending=False, kind="stable").copy()
ranked["_wave_key"] = ranked["wave_id"].map(canonical_wave_id)
ranked = ranked[ranked["_wave_key"].ne("")].drop_duplicates("_wave_key", keep="first")
if ranked.empty:
    st.error("No valid wave identifiers were found in the prediction output.")
    st.stop()
wave_ids = ranked["_wave_key"].tolist()
selected_wave = st.selectbox("Wave", wave_ids, index=0, key="wave_selector_v61")
wave_row = ranked[ranked["_wave_key"] == selected_wave].iloc[0]

warehouse = WarehouseMap.create_default()
route_config = RouteOptimizationConfig()
route = route_for_wave(tasks, selected_wave, warehouse=warehouse, config=route_config)
original_path = expand_route_path(
    route.original_stops,
    warehouse=warehouse,
    depot=route_config.depot,
)
optimized_path = expand_route_path(
    route.optimized_stops,
    warehouse=warehouse,
    depot=route_config.depot,
)

route_cards = st.columns(6)
route_cards[0].metric("Tasks", int(wave_row["task_count"]))
route_cards[1].metric("Pickup stops", int(wave_row["unique_grid_stop_count"]))
route_cards[2].metric("Original distance", int(route.original_distance))
route_cards[3].metric("Optimized distance", int(route.optimized_distance))
route_cards[4].metric("Actual saving", f"{route.saving_pct:.1f}%")
route_cards[5].metric("ML prediction", f"{float(wave_row['predicted_saving_pct']):.1f}%")

animation_tab, path_tab = st.tabs(["▶ Animated route race", "🗺 Before / after paths"])
with animation_tab:
    st.caption(
        "Red circle = original WMS order · green diamond = NN + 2-opt. "
        "Both move one grid cell per frame; the shorter route returns to the depot first."
    )
    st.plotly_chart(
        animated_route_figure(
            warehouse,
            original_path,
            optimized_path,
            route.original_stops,
            route_config.depot,
        ),
        use_container_width=True,
    )
with path_tab:
    st.plotly_chart(
        route_figure(
            warehouse,
            original_path,
            optimized_path,
            route.original_stops,
            route_config.depot,
        ),
        use_container_width=True,
    )

explanation_keys = explanations["wave_id"].map(canonical_wave_id) if "wave_id" in explanations else pd.Series(dtype=str)
explanation = explanations[explanation_keys == selected_wave]
if not explanation.empty:
    item = explanation.iloc[0]
    st.info(
        "Why optimize this wave?  "
        f"**{item['reason_1']}** · **{item['reason_2']}** · **{item['reason_3']}**"
    )
else:
    st.info(
        f"Predicted saving {float(wave_row['predicted_saving_pct']):.1f}% · "
        f"{int(wave_row['zone_transition_count'])} cross-zone transitions · "
        f"original route distance {int(wave_row['original_route_distance'])}"
    )

st.divider()
analysis_left, analysis_right = st.columns(2)
with analysis_left:
    st.subheader("Actual savings distribution")
    st.plotly_chart(
        px.histogram(
            predictions,
            x="saving_pct",
            color="selected_by_ml",
            nbins=30,
            title="Held-out test waves: actual distance saving",
        ),
        use_container_width=True,
    )
with analysis_right:
    st.subheader("What drives the selected model")
    if importance.empty:
        st.write("Native feature importance is not available for the selected estimator.")
    else:
        st.plotly_chart(
            px.bar(
                importance.head(12).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top model features",
            ),
            use_container_width=True,
        )

st.caption(
    "Distance is measured as shortest-path grid steps on the 20×20 simulator map. "
    "The same unique pickup-stop set is used before and after optimization. "
    "Estimated time is simulation-based because the public source does not provide reliable timestamps."
)
