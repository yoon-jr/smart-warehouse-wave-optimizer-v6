from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smart_warehouse.wave_optimization import MODEL_FEATURE_COLUMNS


FEATURE_DESCRIPTIONS = {
    "task_count": "Wave 내 피킹 라인 수",
    "total_quantity": "Wave 내 총 피킹 수량",
    "unique_sku_count": "고유 SKU 수",
    "unique_location_count": "고유 원본 저장 위치 수",
    "unique_grid_stop_count": "시뮬레이터에서 방문해야 하는 고유 피킹 지점 수",
    "operator_count": "기록된 고유 작업자 수",
    "repeated_location_lines": "동일 위치에 속한 반복 피킹 라인 수",
    "active_zone_count": "작업이 존재하는 Zone 수",
    "dominant_zone_ratio": "가장 작업이 집중된 Zone의 비율",
    "zone_entropy": "Zone별 작업 분포의 균등도",
    "zone_a_ratio": "Zone A 작업 비율",
    "zone_b_ratio": "Zone B 작업 비율",
    "zone_c_ratio": "Zone C 작업 비율",
    "zone_d_ratio": "Zone D 작업 비율",
    "mean_x": "피킹 지점 X좌표 평균",
    "mean_y": "피킹 지점 Y좌표 평균",
    "std_x": "피킹 지점 X좌표 표준편차",
    "std_y": "피킹 지점 Y좌표 표준편차",
    "bounding_box_area": "피킹 지점을 포함하는 최소 사각형 면적",
    "mean_pairwise_distance": "피킹 지점 쌍 간 평균 최단거리",
    "max_pairwise_distance": "피킹 지점 쌍 간 최대 최단거리",
    "original_route_distance": "기록 순서 기준 depot 왕복 최단경로 거리",
    "distance_per_stop": "고유 지점당 원본 경로 거리",
    "mean_route_jump": "기록 순서에서 연속 방문 지점 간 평균 거리",
    "max_route_jump": "기록 순서에서 가장 긴 단일 이동 거리",
    "zone_transition_count": "기록 순서의 Zone 전환 횟수",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PPT-ready tables and an auto-filled Korean metrics brief.")
    parser.add_argument("--tasks", type=Path, default=Path("data/footwear_replay_tasks.csv"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/wave_optimization_dataset.csv"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/footwear_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/wave_optimizer"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    if not args.tasks.exists() or not args.dataset.exists():
        raise FileNotFoundError("Run preprocessing and the wave optimizer pipeline before exporting PPT data.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = pd.read_csv(args.tasks, low_memory=False)
    waves = pd.read_csv(args.dataset, low_memory=False)
    manifest = load_json(args.manifest)
    report = load_json(args.output_dir / "wave_optimization_report.json")
    model_board = pd.read_csv(args.output_dir / "wave_model_leaderboard.csv")
    policy_board = pd.read_csv(args.output_dir / "wave_adaptive_policy_leaderboard.csv")

    data_quality = pd.DataFrame(
        [
            ("prepared_pick_lines", len(tasks), "전처리 후 피킹 라인 수"),
            ("unique_waves", tasks["wave_id"].astype(str).nunique(), "고유 Wave 수"),
            (
                "unique_source_locations",
                tasks["source_location"].astype(str).nunique() if "source_location" in tasks else 0,
                "고유 원본 위치 수",
            ),
            ("duplicate_task_ids", int(tasks["task_id"].duplicated().sum()), "중복 task_id 수"),
            ("missing_wave_id", int(tasks["wave_id"].isna().sum()), "Wave ID 결측"),
            ("missing_coordinates", int(tasks[["x", "y"]].isna().any(axis=1).sum()), "좌표 결측"),
            ("location_join_rate", manifest.get("join_rate", "unknown"), "원본 위치-좌표 결합률"),
            ("time_basis", manifest.get("time_basis", "unknown"), "시간축 생성 방식"),
        ],
        columns=["metric", "value", "description"],
    )
    data_quality.to_csv(args.output_dir / "presentation_data_quality.csv", index=False)

    zone_summary = (
        tasks.assign(zone=tasks["zone"].astype(str))
        .groupby("zone", as_index=False)
        .agg(pick_lines=("task_id", "count"), total_quantity=("quantity", "sum"), waves=("wave_id", "nunique"))
    )
    zone_summary["pick_line_share_pct"] = 100.0 * zone_summary["pick_lines"] / zone_summary["pick_lines"].sum()
    zone_summary.to_csv(args.output_dir / "presentation_zone_summary.csv", index=False)

    q80 = float(waves["task_count"].quantile(0.80))
    q95 = float(waves["task_count"].quantile(0.95))
    load_distribution = waves[
        [
            "wave_id",
            "wave_order_index",
            "task_count",
            "unique_grid_stop_count",
            "original_route_distance",
            "optimized_route_distance",
            "saving_pct",
        ]
    ].copy()
    load_distribution["load_segment"] = "normal"
    load_distribution.loc[load_distribution["task_count"] >= q80, "load_segment"] = "high_load_top20"
    load_distribution.loc[load_distribution["task_count"] >= q95, "load_segment"] = "extreme_top5"
    load_distribution.to_csv(args.output_dir / "presentation_wave_distribution.csv", index=False)

    feature_dictionary = pd.DataFrame(
        [
            {
                "feature": feature,
                "description_ko": FEATURE_DESCRIPTIONS.get(feature, ""),
                "available_before_optimization": True,
                "used_as_model_input": True,
            }
            for feature in MODEL_FEATURE_COLUMNS
        ]
    )
    feature_dictionary.to_csv(args.output_dir / "presentation_feature_dictionary.csv", index=False)

    route_summary = pd.DataFrame(
        [
            {
                "method": "recorded_order",
                "mean_distance": float(waves["original_route_distance"].mean()),
                "median_distance": float(waves["original_route_distance"].median()),
                "p95_distance": float(waves["original_route_distance"].quantile(0.95)),
            },
            {
                "method": "nearest_neighbor",
                "mean_distance": float(waves["nearest_neighbor_distance"].mean()),
                "median_distance": float(waves["nearest_neighbor_distance"].median()),
                "p95_distance": float(waves["nearest_neighbor_distance"].quantile(0.95)),
            },
            {
                "method": "best_of_recorded_nn_2opt",
                "mean_distance": float(waves["optimized_route_distance"].mean()),
                "median_distance": float(waves["optimized_route_distance"].median()),
                "p95_distance": float(waves["optimized_route_distance"].quantile(0.95)),
            },
        ]
    )
    route_summary.to_csv(args.output_dir / "presentation_route_summary.csv", index=False)

    selected_model = str(report.get("selected_model", "unknown"))
    validation_best = report.get("validation_best", {})
    test_eval = report.get("test_evaluation", {})
    adaptive = report.get("adaptive_policy", {})
    ml_policy_name = f"ml_adaptive_{selected_model}"
    ml_policy = policy_board[policy_board["method"] == ml_policy_name]
    ml_row = ml_policy.iloc[0].to_dict() if not ml_policy.empty else {}

    brief = f"""# 발표용 자동 집계 수치

## 데이터
- 전처리 피킹 라인: **{len(tasks):,}개**
- Wave: **{tasks['wave_id'].astype(str).nunique():,}개**
- 위치 결합률: **{manifest.get('join_rate', 'unknown')}**
- 시간축: **{manifest.get('time_basis', 'unknown')}**
- 주의: 실제 timestamp가 없으면 시간 관련 표현은 `capacity-calibrated timeline`으로 표기

## 경로 최적화 라벨
- 평균 실제 거리 절감률: **{waves['saving_pct'].mean():.2f}%**
- 중앙값 절감률: **{waves['saving_pct'].median():.2f}%**
- 경로가 실제로 개선된 Wave 비율: **{100 * (waves['distance_saved'] > 0).mean():.2f}%**
- 동일한 고유 피킹 지점 집합을 사용하고 순서만 변경했으므로 비교가 공정함

## 모델 선택
- 선택 모델: **{selected_model}**
- 선택 기준: validation MAE 최저, 동률 시 captured savings와 추론시간 사용
- Validation MAE: **{float(validation_best.get('mae', 0)):.3f} percentage points**
- Test MAE: **{float(test_eval.get('mae', 0)):.3f} percentage points**
- Test Spearman: **{float(test_eval.get('spearman', 0)):.3f}**

## 운영 성능
- 최적화 대상 비율: **{100 * float(report.get('coverage', 0)):.1f}%**
- 전체 가능한 절감량 중 포착: **{float(ml_row.get('captured_savings_pct', adaptive.get('ml_captured_savings_pct', 0))):.2f}%**
- 전체 test 이동거리 절감: **{float(ml_row.get('end_to_end_distance_reduction_pct', adaptive.get('ml_end_to_end_distance_reduction_pct', 0))):.2f}%**
- 최적화 연산을 생략한 Wave 비율: **{float(ml_row.get('optimization_workload_reduction_pct', adaptive.get('optimization_workload_reduction_pct', 0))):.2f}%**

## 발표에서 피해야 할 표현
- 실제 운영시간을 정확히 재현했다
- 최적 경로의 전역 최적해를 보장한다
- Test set에서 가장 좋은 모델을 선택했다

## 권장 표현
- 실제 WMS 작업 순서·위치·수량을 보존한 held-out chronological evaluation
- NN + bounded 2-opt로 생성한 feasible improvement label
- 모든 Wave가 아니라 예상 절감효과가 큰 Wave에 계산을 집중하는 ML-adaptive optimization
"""
    (args.output_dir / "PRESENTATION_NUMBERS_KO.md").write_text(brief, encoding="utf-8")

    print(brief)
    print("Generated PPT-ready files in", args.output_dir)


if __name__ == "__main__":
    main()
