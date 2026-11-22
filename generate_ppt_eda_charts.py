from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "unique_grid_stop_count",
    "saving_pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PPT-ready EDA charts for WareFlow AI."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/wave_optimization_dataset.csv"),
        help="Wave-level optimizer dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/wave_optimizer/ppt_charts"),
        help="Directory for generated PNG and CSV files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. 300 dpi is suitable for PowerPoint.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nRun the wave optimizer pipeline first."
        )


def integer_histogram_bins(values: pd.Series, maximum_bins: int = 40) -> np.ndarray | int:
    minimum = int(values.min())
    maximum = int(values.max())
    integer_range = maximum - minimum + 1

    if integer_range <= maximum_bins:
        return np.arange(minimum - 0.5, maximum + 1.5, 1.0)

    return maximum_bins


def save_unique_stops_chart(
    values: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    ax.hist(
        values,
        bins=integer_histogram_bins(values),
        edgecolor="white",
        linewidth=0.6,
    )

    median_value = float(values.median())
    ax.axvline(
        median_value,
        linestyle="--",
        linewidth=1.8,
        label=f"Median = {median_value:.1f}",
    )

    ax.set_title(
        "Distribution of Unique Picking Stops per Wave",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Number of Unique Picking Stops", fontsize=11)
    ax.set_ylabel("Number of Waves", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.97,
        f"N = {len(values):,} waves",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_saving_distribution_chart(
    values: pd.Series,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    ax.hist(
        values,
        bins=30,
        edgecolor="white",
        linewidth=0.6,
    )

    mean_value = float(values.mean())
    median_value = float(values.median())

    ax.axvline(
        mean_value,
        linestyle="-",
        linewidth=1.8,
        label=f"Mean = {mean_value:.1f}%",
    )
    ax.axvline(
        median_value,
        linestyle="--",
        linewidth=1.8,
        label=f"Median = {median_value:.1f}%",
    )

    ax.set_title(
        "Distribution of Achieved Route-Distance Reduction",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Achieved Route-Distance Reduction (%)", fontsize=11)
    ax.set_ylabel("Number of Waves", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.97,
        f"N = {len(values):,} waves",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    unique_stops: pd.Series,
    saving_pct: pd.Series,
    output_path: Path,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "metric": "unique_picking_stops",
                "count": int(unique_stops.count()),
                "mean": float(unique_stops.mean()),
                "median": float(unique_stops.median()),
                "p95": float(unique_stops.quantile(0.95)),
                "minimum": float(unique_stops.min()),
                "maximum": float(unique_stops.max()),
            },
            {
                "metric": "achieved_route_distance_reduction_pct",
                "count": int(saving_pct.count()),
                "mean": float(saving_pct.mean()),
                "median": float(saving_pct.median()),
                "p95": float(saving_pct.quantile(0.95)),
                "minimum": float(saving_pct.min()),
                "maximum": float(saving_pct.max()),
            },
        ]
    )
    summary.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\n"
            "Expected file: data/wave_optimization_dataset.csv"
        )

    frame = pd.read_csv(args.dataset, low_memory=False)
    validate_columns(frame)

    unique_stops = pd.to_numeric(
        frame["unique_grid_stop_count"],
        errors="coerce",
    ).dropna()

    saving_pct = pd.to_numeric(
        frame["saving_pct"],
        errors="coerce",
    ).dropna()

    if unique_stops.empty:
        raise ValueError("unique_grid_stop_count has no valid numeric values.")
    if saving_pct.empty:
        raise ValueError("saving_pct has no valid numeric values.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stops_path = args.output_dir / "unique_picking_stops_distribution.png"
    saving_path = args.output_dir / "route_distance_reduction_distribution.png"
    summary_path = args.output_dir / "eda_chart_summary.csv"

    save_unique_stops_chart(unique_stops, stops_path, args.dpi)
    save_saving_distribution_chart(saving_pct, saving_path, args.dpi)
    save_summary(unique_stops, saving_pct, summary_path)

    print("Generated:")
    print(f"  {stops_path}")
    print(f"  {saving_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
