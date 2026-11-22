# v6 변경사항 — RL에서 ML-Adaptive Wave Optimization으로 피벗

## 추가된 핵심 모듈

- `smart_warehouse/wave_optimization.py`
  - simulator grid all-pairs shortest path
  - 동일 고유 지점 집합 기준 Original route 계산
  - deterministic Nearest Neighbor
  - bounded first-improvement 2-opt
  - Wave-level feature engineering
  - 실제 절감률 자동 라벨 생성

- `smart_warehouse/wave_benefit_models.py`
  - chronological 70/15/15 split
  - Dummy, Ridge, Random Forest, HistGradientBoosting, XGBoost 비교
  - validation MAE 기반 모델 선택
  - Isolation Forest, LOF, random, longest-route baseline
  - Captured Savings와 end-to-end distance reduction 평가

## 추가된 CLI

- `scripts/build_wave_optimization_dataset.py`
- `scripts/train_wave_benefit_models.py`
- `scripts/run_wave_optimizer_pipeline.py`
- `scripts/export_wave_optimizer_presentation_data.py`

## 추가된 시각화

- `dashboard/wave_optimizer_app.py`
  - KPI cards
  - model leaderboard
  - adaptive-policy leaderboard
  - held-out Wave 선택
  - Original vs Optimized route overlay
  - feature importance
  - explanation reason codes

## 추가된 검증

- Route optimizer가 Original보다 나빠지지 않는지 검증
- Leakage-safe feature list 검증
- Chronological split 검증
- Adaptive metrics 범위 검증
- End-to-end model artifact 생성 검증

전체 테스트: 33 passed.
