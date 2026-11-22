from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from smart_warehouse.config import DATA_DIR, MODEL_DIR
from smart_warehouse.features import FEATURE_COLUMNS, TARGET_COLUMN
from smart_warehouse.ml_models import NaiveLagRegressor


def preprocess(scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "zone"]
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("zone", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["zone"]),
            ("numeric", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "naive_lag": NaiveLagRegressor(),
        "poisson_regression": Pipeline(
            [("preprocess", preprocess(scale_numeric=True)), ("model", PoissonRegressor(alpha=0.1, max_iter=500))]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocess(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=2,
                        criterion="poisson",
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = Pipeline(
            [
                ("preprocess", preprocess(scale_numeric=False)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=150,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="count:poisson",
                        eval_metric="mae",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    except ImportError:
        print("xgboost is not installed; skipping XGBoost.")

    return models


def main() -> None:
    data_path = DATA_DIR / "training_table.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            "Footwear training data not found. Run: python -m scripts.prepare_footwear_data "
            "--dataset-dir <extracted-footwear-folder>"
        )

    data = pd.read_csv(data_path).sort_values(["bin_start", "zone"]).reset_index(drop=True)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN, "bin_start"]) - set(data.columns)
    if missing:
        raise ValueError(f"training_table.csv is missing columns: {sorted(missing)}")
    unique_bins = sorted(data["bin_start"].unique())
    if len(unique_bins) < 4:
        raise ValueError("At least four distinct time bins are required to train and evaluate models.")
    split_index = min(len(unique_bins) - 1, max(1, int(len(unique_bins) * 0.7)))
    split_bin = unique_bins[split_index]
    train = data[data["bin_start"] < split_bin].copy()
    test = data[data["bin_start"] >= split_bin].copy()
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    results: list[dict] = []
    fitted: dict[str, object] = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        prediction = np.clip(np.asarray(model.predict(X_test), dtype=float), 0, None)
        result = {
            "model": name,
            "mae": round(float(mean_absolute_error(y_test, prediction)), 5),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 5),
        }
        results.append(result)
        fitted[name] = model
        print(result)

    results.sort(key=lambda row: (row["mae"], row["rmse"]))
    best_name = results[0]["model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "results": results,
        "data_source": "Footwear Warehouse Order Picking Dataset",
        "split_method": "chronological 70/30 by time bin",
    }
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
