from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from smart_warehouse.wave_optimization import MODEL_FEATURE_COLUMNS

try:  # XGBoost is optional at import time, but included in requirements.txt.
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - environment-specific optional dependency
    XGBRegressor = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    mae: float
    rmse: float
    r2: float
    spearman: float
    top20_recall_pct: float
    captured_savings_pct_at_20: float
    distance_reduction_pct_at_20: float
    inference_ms_per_wave: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "model": self.model_name,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "spearman": self.spearman,
            "top20_recall_pct": self.top20_recall_pct,
            "captured_savings_pct_at_20": self.captured_savings_pct_at_20,
            "distance_reduction_pct_at_20": self.distance_reduction_pct_at_20,
            "inference_ms_per_wave": self.inference_ms_per_wave,
        }


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    if frame.empty:
        raise ValueError("wave dataset is empty")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    ordered = frame.sort_values("wave_order_index", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n < 12:
        raise ValueError("at least 12 waves are required for train/validation/test evaluation")
    train_end = max(1, int(math.floor(n * train_fraction)))
    validation_end = max(train_end + 1, int(math.floor(n * (train_fraction + validation_fraction))))
    validation_end = min(validation_end, n - 1)
    return ChronologicalSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )


def _tree_pipeline(model: RegressorMixin) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def candidate_models(random_seed: int = 42) -> dict[str, Pipeline]:
    workers = max(1, min(4, os.cpu_count() or 1))
    models: dict[str, Pipeline] = {
        "dummy_mean": _tree_pipeline(DummyRegressor(strategy="mean")),
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": _tree_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=workers,
                random_state=random_seed,
            )
        ),
        "hist_gradient_boosting": _tree_pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=random_seed,
            )
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = _tree_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                n_jobs=workers,
                random_state=random_seed,
            )
        )
    return models


def _selection_count(n: int, coverage: float) -> int:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    return max(1, min(n, int(math.ceil(n * coverage))))


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    # Stable mergesort makes tied predictions deterministic.
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return order[:count]


def policy_metrics_from_indices(
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
    *,
    method: str,
) -> dict[str, float | int | str]:
    selected = frame.iloc[np.asarray(selected_indices, dtype=int)]
    total_savings = float(frame["distance_saved"].clip(lower=0).sum())
    captured = float(selected["distance_saved"].clip(lower=0).sum())
    original_total = float(frame["original_route_distance"].sum())
    return {
        "method": method,
        "selected_waves": int(len(selected)),
        "coverage_pct": 100.0 * len(selected) / max(len(frame), 1),
        "captured_savings_pct": 100.0 * captured / total_savings if total_savings > 0 else 0.0,
        "end_to_end_distance_reduction_pct": 100.0 * captured / original_total if original_total > 0 else 0.0,
        "selected_mean_actual_saving_pct": float(selected["saving_pct"].mean()) if len(selected) else 0.0,
        "optimization_workload_reduction_pct": 100.0 * (1.0 - len(selected) / max(len(frame), 1)),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_name: str,
    prediction_seconds: float,
    coverage: float = 0.20,
) -> ModelEvaluation:
    actual = frame["saving_pct"].to_numpy(dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    count = _selection_count(len(frame), coverage)
    predicted_top = _top_indices(predictions, count)
    actual_top = set(_top_indices(actual, count).tolist())
    overlap = len(actual_top.intersection(predicted_top.tolist()))
    policy = policy_metrics_from_indices(frame, predicted_top, method=model_name)
    if np.ptp(actual) == 0 or np.ptp(predictions) == 0:
        spearman = 0.0
    else:
        spearman = float(pd.Series(actual).corr(pd.Series(predictions), method="spearman"))
        if not math.isfinite(spearman):
            spearman = 0.0
    return ModelEvaluation(
        model_name=model_name,
        mae=float(mean_absolute_error(actual, predictions)),
        rmse=float(math.sqrt(mean_squared_error(actual, predictions))),
        r2=float(r2_score(actual, predictions)),
        spearman=spearman,
        top20_recall_pct=100.0 * overlap / count,
        captured_savings_pct_at_20=float(policy["captured_savings_pct"]),
        distance_reduction_pct_at_20=float(policy["end_to_end_distance_reduction_pct"]),
        inference_ms_per_wave=1000.0 * prediction_seconds / max(len(frame), 1),
    )


def fit_and_select_model(
    split: ChronologicalSplit,
    *,
    coverage: float = 0.20,
    random_seed: int = 42,
) -> tuple[str, Pipeline, pd.DataFrame]:
    features = MODEL_FEATURE_COLUMNS
    x_train = split.train[features]
    y_train = split.train["saving_pct"].astype(float)
    x_validation = split.validation[features]
    leaderboard: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in candidate_models(random_seed).items():
        model.fit(x_train, y_train)
        started = time.perf_counter()
        predictions = model.predict(x_validation)
        elapsed = time.perf_counter() - started
        evaluation = evaluate_predictions(
            split.validation,
            predictions,
            model_name=name,
            prediction_seconds=elapsed,
            coverage=coverage,
        )
        leaderboard.append(evaluation.to_dict())
        fitted[name] = model

    table = pd.DataFrame(leaderboard).sort_values(
        ["mae", "captured_savings_pct_at_20", "inference_ms_per_wave"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_name = str(table.iloc[0]["model"])
    return selected_name, fitted[selected_name], table


def _fit_unsupervised_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_seed: int,
) -> dict[str, np.ndarray]:
    features = MODEL_FEATURE_COLUMNS
    x_train = train[features]
    x_test = test[features]

    isolation = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=random_seed,
                    n_jobs=max(1, min(4, os.cpu_count() or 1)),
                ),
            ),
        ]
    )
    isolation.fit(x_train)
    isolation_score = -isolation.decision_function(x_test)

    lof = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LocalOutlierFactor(
                    n_neighbors=max(5, min(35, len(train) - 1)),
                    novelty=True,
                    contamination="auto",
                ),
            ),
        ]
    )
    lof.fit(x_train)
    lof_score = -lof.decision_function(x_test)
    return {"isolation_forest": isolation_score, "local_outlier_factor": lof_score}


def build_adaptive_policy_leaderboard(
    train_for_unsupervised: pd.DataFrame,
    test: pd.DataFrame,
    selected_predictions: np.ndarray,
    *,
    selected_model_name: str,
    coverage: float = 0.20,
    random_seed: int = 42,
    random_repeats: int = 200,
) -> pd.DataFrame:
    n = len(test)
    count = _selection_count(n, coverage)
    rows: list[dict[str, float | int | str]] = []
    rows.append(policy_metrics_from_indices(test, np.array([], dtype=int), method="original_no_optimization"))
    rows.append(policy_metrics_from_indices(test, np.arange(n), method="optimize_all"))

    rng = np.random.default_rng(random_seed)
    random_rows = [
        policy_metrics_from_indices(
            test,
            np.sort(rng.choice(n, size=count, replace=False)),
            method="random_expected",
        )
        for _ in range(random_repeats)
    ]
    random_numeric = pd.DataFrame(random_rows).mean(numeric_only=True).to_dict()
    rows.append({"method": "random_expected", **random_numeric})

    heuristic_indices = _top_indices(test["original_route_distance"].to_numpy(dtype=float), count)
    rows.append(policy_metrics_from_indices(test, heuristic_indices, method="longest_route_heuristic"))

    for method, scores in _fit_unsupervised_scores(
        train_for_unsupervised,
        test,
        random_seed=random_seed,
    ).items():
        rows.append(policy_metrics_from_indices(test, _top_indices(scores, count), method=method))

    selected_indices = _top_indices(selected_predictions, count)
    rows.append(
        policy_metrics_from_indices(
            test,
            selected_indices,
            method=f"ml_adaptive_{selected_model_name}",
        )
    )
    return pd.DataFrame(rows)


def native_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps.get("model")
    values: np.ndarray | None = None
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif estimator is not None and hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_).astype(float))
    if values is None or len(values) != len(MODEL_FEATURE_COLUMNS):
        return pd.DataFrame(columns=["feature", "importance"])
    total = float(values.sum())
    if total > 0:
        values = values / total
    return (
        pd.DataFrame({"feature": MODEL_FEATURE_COLUMNS, "importance": values})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _percentile_rank(reference: pd.Series, value: float) -> float:
    if reference.empty:
        return 0.0
    return 100.0 * float((reference.astype(float) <= value).mean())


def build_wave_explanations(
    train_reference: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    feature_labels = {
        "original_route_distance": "recorded route distance",
        "unique_grid_stop_count": "unique pickup stops",
        "max_pairwise_distance": "warehouse spatial span",
        "zone_transition_count": "cross-zone transitions",
        "bounding_box_area": "pickup footprint area",
    }
    rows: list[dict[str, Any]] = []
    selected = test_predictions.sort_values(
        "predicted_saving_pct", ascending=False, kind="stable"
    ).head(top_n)
    for row in selected.itertuples(index=False):
        reasons: list[tuple[float, str]] = []
        for feature, label in feature_labels.items():
            value = float(getattr(row, feature))
            percentile = _percentile_rank(train_reference[feature], value)
            reasons.append((percentile, f"{label}: top {max(0.1, 100.0 - percentile):.1f}%"))
        reasons.sort(reverse=True)
        rows.append(
            {
                "wave_id": str(row.wave_id),
                "predicted_saving_pct": float(row.predicted_saving_pct),
                "actual_saving_pct": float(row.saving_pct),
                "reason_1": reasons[0][1],
                "reason_2": reasons[1][1],
                "reason_3": reasons[2][1],
            }
        )
    return pd.DataFrame(rows)


def train_wave_benefit_pipeline(
    dataset: pd.DataFrame,
    *,
    output_dir: Path,
    model_dir: Path,
    coverage: float = 0.20,
    random_seed: int = 42,
) -> dict[str, Any]:
    missing = set(MODEL_FEATURE_COLUMNS) - set(dataset.columns)
    required_targets = {"saving_pct", "distance_saved", "original_route_distance", "wave_order_index", "wave_id"}
    missing |= required_targets - set(dataset.columns)
    if missing:
        raise ValueError(f"wave optimization dataset is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    split = chronological_split(dataset)
    selected_name, selected_validation_model, validation_leaderboard = fit_and_select_model(
        split,
        coverage=coverage,
        random_seed=random_seed,
    )

    # Refit only the selected model on train + validation, then evaluate test once.
    train_validation = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = candidate_models(random_seed)[selected_name]
    final_model.fit(
        train_validation[MODEL_FEATURE_COLUMNS],
        train_validation["saving_pct"].astype(float),
    )
    started = time.perf_counter()
    test_predictions_array = final_model.predict(split.test[MODEL_FEATURE_COLUMNS])
    prediction_elapsed = time.perf_counter() - started
    test_evaluation = evaluate_predictions(
        split.test,
        test_predictions_array,
        model_name=selected_name,
        prediction_seconds=prediction_elapsed,
        coverage=coverage,
    )

    predictions = split.test.copy()
    predictions["predicted_saving_pct"] = np.maximum(0.0, test_predictions_array)
    predictions["prediction_error_pct_point"] = (
        predictions["predicted_saving_pct"] - predictions["saving_pct"]
    )
    count = _selection_count(len(predictions), coverage)
    selected_indices = _top_indices(predictions["predicted_saving_pct"].to_numpy(), count)
    predictions["selected_by_ml"] = 0
    predictions.iloc[selected_indices, predictions.columns.get_loc("selected_by_ml")] = 1

    adaptive = build_adaptive_policy_leaderboard(
        train_validation,
        predictions,
        predictions["predicted_saving_pct"].to_numpy(),
        selected_model_name=selected_name,
        coverage=coverage,
        random_seed=random_seed,
    )
    importance = native_feature_importance(final_model)
    explanations = build_wave_explanations(train_validation, predictions)

    artifact = {
        "schema_version": 1,
        "model_name": selected_name,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "target": "saving_pct",
        "coverage": coverage,
        "model": final_model,
    }
    model_path = model_dir / "wave_benefit_model.joblib"
    joblib.dump(artifact, model_path)

    validation_leaderboard.to_csv(output_dir / "wave_model_leaderboard.csv", index=False)
    adaptive.to_csv(output_dir / "wave_adaptive_policy_leaderboard.csv", index=False)
    predictions.to_csv(output_dir / "wave_test_predictions.csv", index=False)
    importance.to_csv(output_dir / "wave_feature_importance.csv", index=False)
    explanations.to_csv(output_dir / "wave_explanations.csv", index=False)

    optimize_all = adaptive[adaptive["method"] == "optimize_all"].iloc[0]
    ml_row = adaptive[adaptive["method"] == f"ml_adaptive_{selected_name}"].iloc[0]
    report = {
        "selected_model": selected_name,
        "selection_rule": "lowest validation MAE; captured savings and latency are deterministic tie-breakers",
        "coverage": coverage,
        "split": {
            "train_waves": len(split.train),
            "validation_waves": len(split.validation),
            "test_waves": len(split.test),
            "method": "chronological 70/15/15 holdout",
        },
        "validation_best": validation_leaderboard.iloc[0].to_dict(),
        "test_evaluation": test_evaluation.to_dict(),
        "adaptive_policy": {
            "ml_selected_waves": int(ml_row["selected_waves"]),
            "ml_captured_savings_pct": float(ml_row["captured_savings_pct"]),
            "ml_end_to_end_distance_reduction_pct": float(ml_row["end_to_end_distance_reduction_pct"]),
            "optimize_all_end_to_end_distance_reduction_pct": float(optimize_all["end_to_end_distance_reduction_pct"]),
            "optimization_workload_reduction_pct": float(ml_row["optimization_workload_reduction_pct"]),
        },
        "artifacts": {
            "model": str(model_path),
            "validation_leaderboard": str(output_dir / "wave_model_leaderboard.csv"),
            "adaptive_policy_leaderboard": str(output_dir / "wave_adaptive_policy_leaderboard.csv"),
            "test_predictions": str(output_dir / "wave_test_predictions.csv"),
            "feature_importance": str(output_dir / "wave_feature_importance.csv"),
            "explanations": str(output_dir / "wave_explanations.csv"),
        },
    }
    (output_dir / "wave_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_wave_benefit_model(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"invalid wave benefit model artifact: {path}")
    return artifact
