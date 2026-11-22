from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ALIASES = {
    "wave_id": ["waveNumber", "wave_number", "wave_id", "wave"],
    "unique_stops": ["unique_grid_stop_count", "unique_location_count", "pickup_stops", "unique_stops"],
    "saving_pct": ["saving_pct", "actual_saving_pct", "distance_saving_pct", "route_distance_reduction_pct"],
}


def find_column(frame: pd.DataFrame, aliases: list[str], label: str) -> str:
    lookup = {str(c).strip().lower(): str(c) for c in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(f"Missing column for {label}. Tried {aliases}. Available: {list(frame.columns)}")


def save(fig: plt.Figure, base: Path, dpi: int, transparent: bool) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", transparent=transparent)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def data_scale(pick_lines: int, waves: int, base: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    ax.axis("off")
    centers = [0.17, 0.50, 0.83]
    titles = [f"{pick_lines:,}", f"{waves:,}", "1 Wave"]
    subtitles = ["Processed WMS Pick Lines", "Wave-Level Samples", "Unit of ML Analysis"]
    for x, title, subtitle in zip(centers, titles, subtitles):
        ax.add_patch(FancyBboxPatch((x-0.13, 0.25), 0.26, 0.50,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    fill=False, linewidth=2))
        ax.text(x, 0.56, title, ha="center", va="center", fontsize=26, weight="bold")
        ax.text(x, 0.39, subtitle, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(0.36,0.50), xytext=(0.31,0.50),
                arrowprops={"arrowstyle":"->","linewidth":2})
    ax.annotate("", xy=(0.69,0.50), xytext=(0.64,0.50),
                arrowprops={"arrowstyle":"->","linewidth":2})
    ax.text(0.5, 0.08,
            "Raw operational records were aggregated into one spatial sample per picking wave.",
            ha="center", fontsize=11)
    fig.tight_layout()
    save(fig, base, dpi, transparent)


def pipeline(base: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.7))
    ax.axis("off")
    steps = [
        ("Picking_Wave.csv\nStorage_Location.csv", "Real WMS sources"),
        ("Clean & Standardize", "Columns, locations, quantities"),
        ("Join Location Data", "Attach warehouse coordinates"),
        ("Map to 20 × 20 Grid", "Common spatial representation"),
        ("Construct Wave Routes", "Preserve recorded order"),
        ("Generate Features & Labels", "One row per wave"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    width, height = 0.14, 0.44
    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        ax.add_patch(FancyBboxPatch((x-width/2, 0.33), width, height,
                                    boxstyle="round,pad=0.015,rounding_size=0.02",
                                    fill=False, linewidth=1.8))
        ax.text(x, 0.60, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x, 0.43, subtitle, ha="center", va="center", fontsize=8.5)
        if i < len(steps)-1:
            ax.annotate("", xy=(xs[i+1]-width/2-0.008,0.55), xytext=(x+width/2+0.008,0.55),
                        arrowprops={"arrowstyle":"->","linewidth":1.7})
    ax.text(0.5, 0.14,
            "Observed WMS records → spatial wave representation → machine-learning dataset",
            ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, base, dpi, transparent)


def unique_stops_chart(values: pd.Series, base: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.hist(values, bins=35, edgecolor="white", linewidth=0.6)
    median = float(values.median())
    ax.axvline(median, linestyle="--", linewidth=1.8, label=f"Median = {median:.1f}")
    ax.set_title("Distribution of Unique Picking Stops per Wave", fontsize=15, pad=12)
    ax.set_xlabel("Number of Unique Picking Stops")
    ax.set_ylabel("Number of Waves")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.98, 0.96, f"N = {len(values):,} waves",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    save(fig, base, dpi, transparent)


def saving_chart(values: pd.Series, base: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.hist(values, bins=30, edgecolor="white", linewidth=0.6)
    mean, median = float(values.mean()), float(values.median())
    ax.axvline(mean, linewidth=1.8, label=f"Mean = {mean:.1f}%")
    ax.axvline(median, linestyle="--", linewidth=1.8, label=f"Median = {median:.1f}%")
    ax.set_title("Distribution of Achieved Route-Distance Reduction", fontsize=15, pad=12)
    ax.set_xlabel("Achieved Route-Distance Reduction (%)")
    ax.set_ylabel("Number of Waves")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.98, 0.96, f"N = {len(values):,} waves",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    save(fig, base, dpi, transparent)


def raw_to_wave(base: Path, dpi: int, transparent: bool) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axis("off")
    raw_rows = [
        "Wave 33182 | SKU A | Qty 1 | Location L-17",
        "Wave 33182 | SKU B | Qty 2 | Location L-17",
        "Wave 33182 | SKU C | Qty 1 | Location M-08",
        "Wave 33182 | SKU D | Qty 3 | Location Q-05",
    ]
    ax.text(0.08, 0.88, "Raw WMS Pick Lines", fontsize=14, weight="bold")
    for y, text in zip([0.72,0.57,0.42,0.27], raw_rows):
        ax.add_patch(FancyBboxPatch((0.08,y-0.045),0.37,0.085,
                                    boxstyle="round,pad=0.01,rounding_size=0.01",
                                    fill=False, linewidth=1.4))
        ax.text(0.095, y, text, fontsize=9.5, va="center")
    ax.annotate("Aggregate by Wave", xy=(0.58,0.50), xytext=(0.49,0.50),
                ha="center", va="center", fontsize=11, weight="bold",
                arrowprops={"arrowstyle":"->","linewidth":2})
    ax.add_patch(FancyBboxPatch((0.63,0.23),0.30,0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.025",
                                fill=False, linewidth=2))
    ax.text(0.78, 0.70, "One Wave-Level ML Sample", ha="center", fontsize=14, weight="bold")
    lines = [
        "task_count = 24",
        "unique_stops = 18",
        "original_route_distance = 198",
        "zone_entropy = …",
        "bounding_box_area = …",
        "target: saving_pct = 60.6%",
    ]
    for i, line in enumerate(lines):
        ax.text(0.665, 0.60-i*0.075, line, fontsize=10.5)
    ax.text(0.5,0.08,
            "Multiple tasks at the same location are retained, but counted as one physical route stop.",
            ha="center", fontsize=10.5)
    fig.tight_layout()
    save(fig, base, dpi, transparent)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wave-dataset", type=Path, default=Path("data/wave_optimization_dataset.csv"))
    p.add_argument("--tasks", type=Path, default=Path("data/footwear_replay_tasks.csv"))
    p.add_argument("--pick-lines", type=int, default=191583)
    p.add_argument("--output-dir", type=Path, default=Path("runtime/wave_optimizer/ppt_visuals"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--transparent", action="store_true")
    args = p.parse_args()

    if not args.wave_dataset.exists():
        raise FileNotFoundError(f"Missing {args.wave_dataset}. Run the optimizer pipeline first.")

    waves = pd.read_csv(args.wave_dataset, low_memory=False)
    wave_col = find_column(waves, ALIASES["wave_id"], "wave id")
    stops_col = find_column(waves, ALIASES["unique_stops"], "unique stops")
    saving_col = find_column(waves, ALIASES["saving_pct"], "saving percentage")

    unique_stops = pd.to_numeric(waves[stops_col], errors="coerce").dropna()
    saving_pct = pd.to_numeric(waves[saving_col], errors="coerce").dropna()
    wave_count = int(waves[wave_col].nunique())

    if args.tasks.exists():
        pick_lines = len(pd.read_csv(args.tasks, low_memory=False))
    else:
        pick_lines = args.pick_lines

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_scale(pick_lines, wave_count, args.output_dir/"01_data_scale", args.dpi, args.transparent)
    pipeline(args.output_dir/"02_preprocessing_pipeline", args.dpi, args.transparent)
    unique_stops_chart(unique_stops, args.output_dir/"03_unique_stops_distribution", args.dpi, args.transparent)
    saving_chart(saving_pct, args.output_dir/"04_route_saving_distribution", args.dpi, args.transparent)
    raw_to_wave(args.output_dir/"05_raw_to_wave_sample", args.dpi, args.transparent)

    pd.DataFrame({
        "metric": ["processed_pick_lines","wave_level_samples","unique_stops_mean",
                   "unique_stops_median","saving_pct_mean","saving_pct_median","saving_pct_p95"],
        "value": [pick_lines,wave_count,unique_stops.mean(),unique_stops.median(),
                  saving_pct.mean(),saving_pct.median(),saving_pct.quantile(0.95)],
    }).to_csv(args.output_dir/"visual_summary.csv", index=False)

    print("Generated:")
    for path in sorted(args.output_dir.glob("*")):
        print(" ", path)


if __name__ == "__main__":
    main()
