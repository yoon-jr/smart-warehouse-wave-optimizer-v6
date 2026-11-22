from __future__ import annotations

import pandas as pd

ZONE_IDS = ("A", "B", "C", "D")
FEATURE_COLUMNS = ["zone", "hour", "slot_in_hour", "lag_1", "rolling_3", "rolling_6"]
TARGET_COLUMN = "target_next_count"


def build_training_table(tasks: pd.DataFrame, total_minutes: int, bin_minutes: int = 5) -> pd.DataFrame:
    if total_minutes <= bin_minutes:
        raise ValueError("total_minutes must be larger than bin_minutes.")

    bins = pd.RangeIndex(0, total_minutes, bin_minutes, name="bin_start")
    base = pd.MultiIndex.from_product([bins, ZONE_IDS], names=["bin_start", "zone"]).to_frame(index=False)

    if tasks.empty:
        counts = pd.DataFrame(columns=["bin_start", "zone", "count"])
    else:
        task_copy = tasks.copy()
        task_copy["bin_start"] = (task_copy["absolute_minute"] // bin_minutes) * bin_minutes
        counts = task_copy.groupby(["bin_start", "zone"]).size().rename("count").reset_index()

    table = base.merge(counts, on=["bin_start", "zone"], how="left")
    table["count"] = table["count"].fillna(0).astype(int)
    table = table.sort_values(["zone", "bin_start"]).reset_index(drop=True)
    table["hour"] = (table["bin_start"] % (24 * 60) // 60).astype(int)
    table["slot_in_hour"] = ((table["bin_start"] % 60) // bin_minutes).astype(int)

    grouped = table.groupby("zone", group_keys=False)["count"]
    table["lag_1"] = grouped.shift(1).fillna(0)
    table["rolling_3"] = grouped.transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    table["rolling_6"] = grouped.transform(lambda series: series.shift(1).rolling(6, min_periods=1).mean()).fillna(0)
    table[TARGET_COLUMN] = grouped.shift(-1)
    table = table.dropna(subset=[TARGET_COLUMN]).copy()
    table[TARGET_COLUMN] = table[TARGET_COLUMN].astype(float)
    return table


def live_feature_rows(
    current_minute: float,
    recent_counts: dict[str, list[int]],
    bin_minutes: int = 5,
) -> pd.DataFrame:
    minute_of_day = int(current_minute) % (24 * 60)
    rows: list[dict] = []
    for zone in ZONE_IDS:
        history = recent_counts.get(zone, [])
        lag_1 = history[-1] if history else 0
        rolling_3 = sum(history[-3:]) / max(1, len(history[-3:]))
        rolling_6 = sum(history[-6:]) / max(1, len(history[-6:]))
        rows.append(
            {
                "zone": zone,
                "hour": minute_of_day // 60,
                "slot_in_hour": (minute_of_day % 60) // bin_minutes,
                "lag_1": lag_1,
                "rolling_3": rolling_3,
                "rolling_6": rolling_6,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
