from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from smart_warehouse.config import SimulationConfig
from smart_warehouse.features import build_training_table
from smart_warehouse.warehouse import WarehouseMap

DATASET_NAME = "Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company"
DATASET_DOI = "10.17632/pf2w725pw3.1"
DATASET_URL = "https://data.mendeley.com/datasets/pf2w725pw3/1"
DATASET_LICENSE = "CC BY 4.0"


class FootwearDataError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedFootwearData:
    tasks: pd.DataFrame
    layout_map: pd.DataFrame
    training_table: pd.DataFrame
    manifest: dict[str, Any]


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _find_file(dataset_dir: Path, filename: str, *, required: bool = True) -> Path | None:
    matches = [path for path in dataset_dir.rglob("*") if path.is_file() and path.name.lower() == filename.lower()]
    if matches:
        return sorted(matches)[0]
    if required:
        raise FileNotFoundError(
            f"Required file '{filename}' was not found under {dataset_dir}. "
            "Download and extract the Mendeley dataset before preprocessing."
        )
    return None


def read_csv_flexible(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    attempts: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        for separator in (None, ",", ";", "\t"):
            try:
                kwargs: dict[str, Any] = {
                    "encoding": encoding,
                    "nrows": nrows,
                    "low_memory": False,
                }
                if separator is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = separator
                frame = pd.read_csv(path, **kwargs)
                if len(frame.columns) > 1 or frame.empty:
                    frame.columns = [str(column).strip() for column in frame.columns]
                    return frame
                attempts.append(f"{encoding}/{separator!r}: only one column")
            except Exception as exc:  # pragma: no cover - diagnostic aggregation
                attempts.append(f"{encoding}/{separator!r}: {type(exc).__name__}")
    raise FootwearDataError(f"Could not parse {path}. Attempts: {', '.join(attempts[-8:])}")


def find_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = False,
    label: str = "column",
) -> str | None:
    normalized = {normalize_name(column): str(column) for column in frame.columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise FootwearDataError(
            f"Could not detect {label}. Tried aliases {list(aliases)}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


WAVE_ALIASES = (
    "picking_wave",
    "pickingwave",
    "wave",
    "wave_id",
    "waveid",
    "wave_number",
    "wavenumber",
    "number_wave",
    "picklist",
    "picklist_id",
)
LOCATION_ALIASES = (
    "storage_location",
    "storagelocation",
    "original_location",
    "originallocation",
    "location",
    "locations",
    "position",
    "address",
)
QUANTITY_ALIASES = (
    "quantity_to_pick",
    "quantitytopick",
    "quantity_to_pick_units",
    "quantitytopickunits",
    "qty_to_pick",
    "qtypick",
    "quantity",
    "qty",
    "pairs_to_pick",
    "pairstopick",
)
SEQUENCE_ALIASES = (
    "pick_sequence",
    "picksequence",
    "sequence",
    "sequence_number",
    "sequencenumber",
    "order_sequence",
    "position_in_wave",
)
DURATION_ALIASES = (
    "wave_duration",
    "waveduration",
    "duration",
    "duration_minutes",
    "durationminutes",
    "picking_time",
    "pickingtime",
    "elapsed_time",
)
TIMESTAMP_ALIASES = (
    "timestamp",
    "datetime",
    "date_time",
    "created_at",
    "createdat",
    "wave_timestamp",
    "wavetimestamp",
    "start_time",
    "starttime",
    "order_timestamp",
    "ordertimestamp",
    "order_date",
    "orderdate",
)
DATE_ALIASES = ("date", "wave_date", "wavedate", "order_date", "orderdate", "created_date")
TIME_ALIASES = ("time", "wave_time", "wavetime", "order_time", "ordertime", "created_time")
ORDER_ALIASES = ("order_number", "ordernumber", "order_id", "orderid", "order", "order_to_collect")
REFERENCE_ALIASES = ("reference", "product_reference", "productreference", "sku", "item")
OPERATOR_ALIASES = ("operator", "operator_id", "operatorid", "picker", "employee")
POINT_ALIASES = ("points", "point", "coordinates", "coordinate", "cartesian", "xyz")
X_ALIASES = ("x", "point_x", "pointx", "coordinate_x", "coordinatex")
Y_ALIASES = ("y", "point_y", "pointy", "coordinate_y", "coordinatey")
Z_ALIASES = ("z", "point_z", "pointz", "coordinate_z", "coordinatez")


def _clean_location(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )


def parse_point(value: Any) -> tuple[float, float, float] | None:
    """Parse common Cartesian point strings without confusing delimiters with decimals.

    The public file is documented with values such as ``(X, Y, Z)``. Commas
    are therefore treated as coordinate separators. European decimal commas
    remain supported when coordinates are separated by semicolons.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().strip("()[]{}")
    if not text:
        return None
    if ";" in text:
        tokens = [token.strip().replace(",", ".") for token in text.split(";")]
    else:
        tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    try:
        parsed = [float(token) for token in tokens if token][:3]
    except ValueError:
        return None
    if len(parsed) < 2:
        return None
    if len(parsed) == 2:
        parsed.append(0.0)
    return parsed[0], parsed[1], parsed[2]


def _parse_location_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    location_col = find_column(frame, LOCATION_ALIASES, required=True, label="storage location identifier")
    point_col = find_column(frame, POINT_ALIASES)
    x_col = find_column(frame, X_ALIASES)
    y_col = find_column(frame, Y_ALIASES)
    z_col = find_column(frame, Z_ALIASES)

    if point_col:
        parsed = frame[point_col].map(parse_point)
        coordinate_source = point_col
    elif x_col and y_col:
        x = pd.to_numeric(frame[x_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        y = pd.to_numeric(frame[y_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if z_col:
            z = pd.to_numeric(frame[z_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        else:
            z = pd.Series(0.0, index=frame.index)
        parsed = pd.Series(list(zip(x, y, z)), index=frame.index).map(
            lambda point: point if pd.notna(point[0]) and pd.notna(point[1]) else None
        )
        coordinate_source = ",".join(column for column in (x_col, y_col, z_col) if column)
    else:
        raise FootwearDataError(
            "Storage_Location.csv must contain a Points/Coordinates column or separate X and Y columns. "
            f"Available columns: {list(frame.columns)}"
        )

    locations = pd.DataFrame(
        {
            "source_location": _clean_location(frame[location_col]),
            "_point": parsed,
        }
    )
    locations = locations[locations["source_location"].ne("") & locations["_point"].notna()].copy()
    locations[["source_x", "source_y", "source_z"]] = pd.DataFrame(
        locations["_point"].tolist(), index=locations.index
    )
    locations = (
        locations.drop(columns="_point")
        .drop_duplicates(subset=["source_location"], keep="first")
        .reset_index(drop=True)
    )
    if locations.empty:
        raise FootwearDataError("No valid storage coordinates were parsed from Storage_Location.csv.")
    return locations, {
        "location_column": location_col,
        "coordinate_source": coordinate_source,
        "valid_storage_locations": int(len(locations)),
    }


def _detect_datetime_series(frame: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    direct = find_column(frame, TIMESTAMP_ALIASES)
    if direct:
        parsed = pd.to_datetime(frame[direct], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.50:
            return parsed, {"timestamp_column": direct}

    date_col = find_column(frame, DATE_ALIASES)
    time_col = find_column(frame, TIME_ALIASES)
    if date_col:
        values = frame[date_col].astype("string").fillna("")
        if time_col:
            values = values + " " + frame[time_col].astype("string").fillna("")
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=False)
        if parsed.notna().mean() >= 0.50:
            metadata: dict[str, Any] = {"date_column": date_col}
            if time_col:
                metadata["time_column"] = time_col
            return parsed, metadata
    return None, {}


def _duration_to_minutes(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text.replace(",", "."))
        if numeric < 0:
            return None
        # Large plain values in operational exports are commonly seconds.
        return numeric / 60.0 if numeric > 300 else numeric
    except ValueError:
        pass
    try:
        duration = pd.to_timedelta(text)
        minutes = duration.total_seconds() / 60.0
        return minutes if minutes >= 0 else None
    except (ValueError, TypeError):
        return None


def _attach_customer_timestamps(
    waves: pd.DataFrame,
    customers: pd.DataFrame | None,
    wave_id_col: str,
) -> tuple[pd.Series | None, dict[str, Any]]:
    direct, metadata = _detect_datetime_series(waves)
    if direct is not None:
        return direct, {"source": "Picking_Wave.csv", **metadata}
    if customers is None or customers.empty:
        return None, {}

    customer_time, customer_meta = _detect_datetime_series(customers)
    if customer_time is None:
        return None, {}

    customer_wave_col = find_column(customers, WAVE_ALIASES)
    if customer_wave_col:
        lookup = pd.DataFrame(
            {
                "_join": customers[customer_wave_col].astype("string").str.strip(),
                "_timestamp": customer_time,
            }
        ).dropna(subset=["_timestamp"])
        lookup = lookup.groupby("_join", as_index=False)["_timestamp"].min()
        joined = waves[wave_id_col].astype("string").str.strip().to_frame("_join").merge(
            lookup, on="_join", how="left"
        )
        if joined["_timestamp"].notna().mean() >= 0.50:
            return joined["_timestamp"], {
                "source": "Customer_Order.csv",
                "join": "wave_id",
                "customer_wave_column": customer_wave_col,
                **customer_meta,
            }

    wave_order_col = find_column(waves, ORDER_ALIASES)
    customer_order_col = find_column(customers, ORDER_ALIASES)
    if wave_order_col and customer_order_col:
        lookup = pd.DataFrame(
            {
                "_join": customers[customer_order_col].astype("string").str.strip(),
                "_timestamp": customer_time,
            }
        ).dropna(subset=["_timestamp"])
        lookup = lookup.groupby("_join", as_index=False)["_timestamp"].min()
        joined = waves[wave_order_col].astype("string").str.strip().to_frame("_join").merge(
            lookup, on="_join", how="left"
        )
        if joined["_timestamp"].notna().mean() >= 0.50:
            return joined["_timestamp"], {
                "source": "Customer_Order.csv",
                "join": "order_id",
                "wave_order_column": wave_order_col,
                "customer_order_column": customer_order_col,
                **customer_meta,
            }
    return None, {}


def _wave_release_minutes(
    frame: pd.DataFrame,
    *,
    wave_id_col: str,
    timestamps: pd.Series | None,
    duration_col: str | None,
    minutes_per_wave: float,
    start_minute: float,
) -> tuple[pd.Series, str, dict[str, Any]]:
    wave_values = frame[wave_id_col].astype("string").fillna("UNKNOWN").str.strip()
    first_order = pd.Series(pd.factorize(wave_values, sort=False)[0], index=frame.index)

    if timestamps is not None and timestamps.notna().any():
        timestamp_frame = pd.DataFrame({"wave": wave_values, "timestamp": timestamps})
        wave_time = timestamp_frame.groupby("wave", sort=False)["timestamp"].min()
        valid = wave_time.dropna()
        if not valid.empty:
            baseline = valid.min()
            elapsed = (wave_time - baseline).dt.total_seconds() / 60.0
            # Missing wave times retain sequence order after the latest known wave.
            next_minute = float(elapsed.dropna().max()) + minutes_per_wave
            for wave in wave_time.index[wave_time.isna()]:
                elapsed.loc[wave] = next_minute
                next_minute += minutes_per_wave
            releases = wave_values.map(elapsed).astype(float) + start_minute
            return releases, "recorded_timestamp", {
                "timestamp_coverage": round(float(timestamps.notna().mean()), 4),
                "recorded_start": baseline.isoformat(),
            }

    wave_order_table = pd.DataFrame({"wave": wave_values, "order": first_order})
    wave_order_table = wave_order_table.drop_duplicates("wave").sort_values("order")
    duration_used = False
    duration_metadata: dict[str, Any] = {}
    if duration_col:
        duration_values = frame[duration_col].map(_duration_to_minutes)
        duration_table = pd.DataFrame({"wave": wave_values, "duration": duration_values})
        wave_duration = duration_table.groupby("wave", sort=False)["duration"].first()
        coverage = float(wave_duration.notna().mean())
        if coverage >= 0.50:
            duration_used = True
            median_duration = float(wave_duration.dropna().median())
            if not math.isfinite(median_duration) or median_duration <= 0:
                median_duration = minutes_per_wave
            wave_duration = wave_duration.fillna(median_duration).clip(lower=1.0, upper=240.0)
            elapsed_by_wave: dict[str, float] = {}
            elapsed = start_minute
            for wave in wave_order_table["wave"]:
                elapsed_by_wave[str(wave)] = elapsed
                elapsed += float(wave_duration.loc[wave])
            duration_metadata = {
                "duration_column": duration_col,
                "duration_coverage": round(coverage, 4),
                "median_duration_minutes": round(median_duration, 3),
            }
            return wave_values.map(elapsed_by_wave).astype(float), "wave_duration_sequence", duration_metadata

    release_lookup = {
        str(wave): start_minute + index * minutes_per_wave
        for index, wave in enumerate(wave_order_table["wave"])
    }
    return wave_values.map(release_lookup).astype(float), "fixed_wave_sequence", {
        "minutes_per_wave": minutes_per_wave,
        "duration_column_detected_but_not_used": duration_col if duration_col and not duration_used else None,
    }


def _nearest_walkable(
    warehouse: WarehouseMap,
    approximate: tuple[int, int],
    *,
    preferred_zone: str | None = None,
) -> tuple[int, int]:
    candidates = [
        (x, y)
        for y in range(warehouse.height)
        for x in range(warehouse.width)
        if warehouse.walkable((x, y))
        and (preferred_zone is None or warehouse.zone_for((x, y)) == preferred_zone)
    ]
    if not candidates:
        raise FootwearDataError("Warehouse grid has no walkable cells for coordinate mapping.")
    ax, ay = approximate
    return min(candidates, key=lambda point: ((point[0] - ax) ** 2 + (point[1] - ay) ** 2, point))


def _map_locations_to_grid(
    locations: pd.DataFrame,
    warehouse: WarehouseMap,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_min, x_max = float(locations["source_x"].min()), float(locations["source_x"].max())
    y_min, y_max = float(locations["source_y"].min()), float(locations["source_y"].max())
    if math.isclose(x_min, x_max) or math.isclose(y_min, y_max):
        raise FootwearDataError("Storage coordinates do not span a usable two-dimensional area.")

    def scale(value: float, low: float, high: float, size: int) -> int:
        normalized = (float(value) - low) / (high - low)
        return int(round(1 + normalized * max(1, size - 3)))

    mapped_rows: list[dict[str, Any]] = []
    for row in locations.itertuples(index=False):
        approximate = (
            scale(float(row.source_x), x_min, x_max, warehouse.width),
            scale(float(row.source_y), y_min, y_max, warehouse.height),
        )
        zone_id = warehouse.zone_for(
            (
                min(max(approximate[0], 0), warehouse.width - 1),
                min(max(approximate[1], 0), warehouse.height - 1),
            )
        )
        grid_x, grid_y = _nearest_walkable(warehouse, approximate, preferred_zone=zone_id)
        mapped_rows.append(
            {
                "source_location": row.source_location,
                "source_x": float(row.source_x),
                "source_y": float(row.source_y),
                "source_z": float(row.source_z),
                "grid_x": grid_x,
                "grid_y": grid_y,
                "zone": warehouse.zone_for((grid_x, grid_y)),
            }
        )
    mapped = pd.DataFrame(mapped_rows)
    return mapped, {
        "coordinate_bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "grid_size": [warehouse.width, warehouse.height],
        "zone_method": "relative Cartesian coordinates mapped to four grid quadrants",
        "walkable_mapping": "nearest walkable cell within the same quadrant",
        "unique_grid_pickups": int(mapped[["grid_x", "grid_y"]].drop_duplicates().shape[0]),
    }


def prepare_footwear_dataset(
    dataset_dir: Path,
    *,
    config: SimulationConfig | None = None,
    minutes_per_wave: float = 15.0,
    minimum_join_rate: float = 0.80,
    max_rows: int | None = None,
) -> PreparedFootwearData:
    if minutes_per_wave <= 0:
        raise ValueError("minutes_per_wave must be positive")
    if not 0 < minimum_join_rate <= 1:
        raise ValueError("minimum_join_rate must be in (0, 1]")
    config = config or SimulationConfig()
    warehouse = WarehouseMap.create_default(config.grid_width, config.grid_height)

    wave_path = _find_file(dataset_dir, "Picking_Wave.csv")
    storage_path = _find_file(dataset_dir, "Storage_Location.csv")
    customer_path = _find_file(dataset_dir, "Customer_Order.csv", required=False)

    waves = read_csv_flexible(wave_path, nrows=max_rows)
    storage = read_csv_flexible(storage_path)
    customers = read_csv_flexible(customer_path) if customer_path else None

    wave_id_col = find_column(waves, WAVE_ALIASES, required=True, label="picking wave identifier")
    wave_location_col = find_column(waves, LOCATION_ALIASES, required=True, label="picking location")
    quantity_col = find_column(waves, QUANTITY_ALIASES)
    sequence_col = find_column(waves, SEQUENCE_ALIASES)
    duration_col = find_column(waves, DURATION_ALIASES)
    reference_col = find_column(waves, REFERENCE_ALIASES)
    operator_col = find_column(waves, OPERATOR_ALIASES)

    locations, location_metadata = _parse_location_frame(storage)
    layout_map, grid_metadata = _map_locations_to_grid(locations, warehouse)

    standardized = waves.copy()
    standardized["_source_location"] = _clean_location(standardized[wave_location_col])
    standardized["_wave_id"] = standardized[wave_id_col].astype("string").fillna("UNKNOWN").str.strip()
    standardized["_original_row"] = np.arange(len(standardized), dtype=int)
    standardized = standardized.merge(layout_map, left_on="_source_location", right_on="source_location", how="left")

    join_rate = float(standardized["grid_x"].notna().mean()) if len(standardized) else 0.0
    if join_rate < minimum_join_rate:
        unmatched = standardized.loc[standardized["grid_x"].isna(), "_source_location"].drop_duplicates().head(10)
        raise FootwearDataError(
            f"Only {join_rate:.1%} of Picking_Wave locations matched Storage_Location coordinates; "
            f"minimum required is {minimum_join_rate:.1%}. Example unmatched locations: {unmatched.tolist()}"
        )
    standardized = standardized.dropna(subset=["grid_x", "grid_y", "zone"]).copy()

    timestamps, timestamp_metadata = _attach_customer_timestamps(standardized, customers, wave_id_col)
    release_minutes, time_basis, time_metadata = _wave_release_minutes(
        standardized,
        wave_id_col=wave_id_col,
        timestamps=timestamps,
        duration_col=duration_col,
        minutes_per_wave=minutes_per_wave,
        start_minute=config.start_minute,
    )

    if sequence_col:
        sequence_numeric = pd.to_numeric(standardized[sequence_col], errors="coerce")
        fallback_sequence = standardized.groupby("_wave_id", sort=False).cumcount() + 1
        standardized["_sequence"] = sequence_numeric.fillna(fallback_sequence).astype(int)
    else:
        standardized["_sequence"] = standardized.groupby("_wave_id", sort=False).cumcount() + 1

    if quantity_col:
        quantities = pd.to_numeric(
            standardized[quantity_col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(1.0)
    else:
        quantities = pd.Series(1.0, index=standardized.index)
    quantities = quantities.clip(lower=1.0)
    priority_cutoff = float(quantities.quantile(0.75)) if len(quantities) else 1.0

    # A picking wave contains an ordered list of picks, not hundreds of tasks
    # that physically start at exactly the same instant. Spread each pick line
    # across its wave window using the recorded sequence. This preserves the
    # wave order and prevents an artificial one-tick burst caused by the MVP
    # conversion from human picking waves to independent robot tasks.
    if duration_col:
        parsed_duration = standardized[duration_col].map(_duration_to_minutes)
        wave_duration = pd.DataFrame(
            {"wave": standardized["_wave_id"], "duration": parsed_duration}
        ).groupby("wave", sort=False)["duration"].first()
        valid_duration = wave_duration.dropna()
        median_wave_duration = (
            float(valid_duration.median()) if not valid_duration.empty else float(minutes_per_wave)
        )
        if not math.isfinite(median_wave_duration) or median_wave_duration <= 0:
            median_wave_duration = float(minutes_per_wave)
        wave_duration = wave_duration.fillna(median_wave_duration).clip(lower=1.0, upper=240.0)
    else:
        median_wave_duration = float(minutes_per_wave)
        wave_duration = pd.Series(
            {wave: median_wave_duration for wave in standardized["_wave_id"].drop_duplicates()}
        )

    maximum_sequence = standardized.groupby("_wave_id", sort=False)["_sequence"].transform("max").clip(lower=1)
    line_fraction = (standardized["_sequence"] - 1).clip(lower=0) / maximum_sequence
    line_offset = pd.Series(
        [
            float(fraction) * float(wave_duration.loc[wave])
            for fraction, wave in zip(line_fraction, standardized["_wave_id"])
        ],
        index=standardized.index,
    )
    created_minute = release_minutes.astype(float) + line_offset
    absolute_minute = (created_minute - float(created_minute.min())).clip(lower=0.0)
    release_tick = np.floor(absolute_minute / config.simulation_minutes_per_tick).astype(int) + 1
    service_ticks = quantities.map(
        lambda quantity: int(
            min(
                config.work_duration_ticks * 3,
                max(4, round(config.work_duration_ticks * (0.65 + 0.20 * math.log2(1.0 + float(quantity))))),
            )
        )
    )

    task_rows: list[dict[str, Any]] = []
    for index, row in standardized.reset_index(drop=True).iterrows():
        quantity = float(quantities.iloc[index])
        task_rows.append(
            {
                "task_id": f"FW-{index + 1:07d}",
                "absolute_minute": round(float(absolute_minute.iloc[index]), 4),
                "created_minute": round(float(created_minute.iloc[index]), 4),
                "minute_of_day": round(float(created_minute.iloc[index] % (24 * 60)), 4),
                "day": int(absolute_minute.iloc[index] // (24 * 60)),
                "release_tick": int(release_tick.iloc[index]),
                "zone": str(row["zone"]),
                "x": int(row["grid_x"]),
                "y": int(row["grid_y"]),
                "priority": 2 if quantity >= priority_cutoff and priority_cutoff > 1 else 1,
                "quantity": round(quantity, 3),
                "service_ticks": int(service_ticks.iloc[index]),
                "wave_id": str(row["_wave_id"]),
                "source_location": str(row["_source_location"]),
                "source_sequence": int(row["_sequence"]),
                "product_reference": str(row[reference_col]) if reference_col else None,
                "operator": str(row[operator_col]) if operator_col else None,
                "time_basis": time_basis,
                "data_source": "footwear_wms",
            }
        )

    tasks = pd.DataFrame(task_rows).sort_values(
        ["release_tick", "wave_id", "source_sequence", "task_id"], kind="stable"
    ).reset_index(drop=True)
    if tasks.empty:
        raise FootwearDataError("No valid picking tasks remained after preprocessing.")

    total_minutes = max(
        60,
        int(math.ceil(float(tasks["absolute_minute"].max()))) + 10,
    )
    training = build_training_table(tasks, total_minutes=total_minutes)

    zone_counts = tasks["zone"].value_counts().reindex(list("ABCD"), fill_value=0)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "dataset": DATASET_NAME,
        "doi": DATASET_DOI,
        "source_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "picking_wave": str(wave_path),
            "storage_location": str(storage_path),
            "customer_order": str(customer_path) if customer_path else None,
        },
        "rows": {
            "picking_wave_input": int(len(waves)),
            "prepared_tasks": int(len(tasks)),
            "training_rows": int(len(training)),
            "storage_locations": int(len(layout_map)),
        },
        "join_rate": round(join_rate, 6),
        "time_basis": time_basis,
        "time_basis_note": (
            "Recorded timestamps were used."
            if time_basis == "recorded_timestamp"
            else "The public article is internally inconsistent about timestamps; no reliable timestamp was detected in the files. "
            "Chronological wave order was preserved and converted to simulation minutes using the documented fallback."
        ),
        "zone_task_counts": {zone: int(zone_counts.loc[zone]) for zone in "ABCD"},
        "detected_columns": {
            "wave_id": wave_id_col,
            "wave_location": wave_location_col,
            "quantity": quantity_col,
            "sequence": sequence_col,
            "duration": duration_col,
            "reference": reference_col,
            "operator": operator_col,
        },
        "timestamp_detection": timestamp_metadata,
        "time_conversion": time_metadata,
        "location_detection": location_metadata,
        "grid_mapping": grid_metadata,
        "simulation_minutes_per_tick": config.simulation_minutes_per_tick,
        "pick_line_release_method": "wave start plus sequence-proportional offset within wave duration",
        "median_wave_duration_minutes_for_release": round(float(median_wave_duration), 3),
        "modeling_assumptions": [
            "Each pick line in Picking_Wave.csv becomes one independent robot task.",
            "The original Cartesian pickup coordinate is scaled to the nearest walkable simulator cell while preserving its quadrant.",
            "Continuous human-picker trajectories are not claimed; the dataset provides pick sequence and locations, not exact trajectories.",
            "Pick lines are released across the wave duration according to their recorded sequence instead of being emitted as an artificial one-tick burst.",
            "Depot return and item handling are abstracted into task service time in this MVP.",
        ],
    }
    return PreparedFootwearData(tasks=tasks, layout_map=layout_map, training_table=training, manifest=manifest)


def save_prepared_footwear_data(
    prepared: PreparedFootwearData,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output_dir / "footwear_replay_tasks.csv",
        "historical": output_dir / "historical_tasks.csv",
        "training": output_dir / "training_table.csv",
        "layout": output_dir / "footwear_location_map.csv",
        "manifest": output_dir / "footwear_manifest.json",
    }
    prepared.tasks.to_csv(paths["tasks"], index=False)
    prepared.tasks.to_csv(paths["historical"], index=False)
    prepared.training_table.to_csv(paths["training"], index=False)
    prepared.layout_map.to_csv(paths["layout"], index=False)
    paths["manifest"].write_text(
        json.dumps(prepared.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths
