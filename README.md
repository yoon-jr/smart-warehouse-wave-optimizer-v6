# WareFlow AI — WMS Wave Optimization Digital Twin (v6)

실제 Footwear Warehouse WMS 피킹 데이터를 이용해 다음을 수행하는 해커톤용 end-to-end 시스템입니다.

1. 기록된 Wave별 피킹 순서의 이동거리를 계산합니다.
2. Nearest Neighbor와 bounded 2-opt로 같은 피킹 지점의 방문 순서만 개선합니다.
3. 실제 절감률을 자동 라벨로 만들어 Dummy, Ridge, Random Forest, HistGradientBoosting, XGBoost를 비교합니다.
4. Validation MAE가 가장 낮은 모델을 선택합니다.
5. Isolation Forest와 LOF는 라벨 없는 비지도 baseline으로 비교합니다.
6. 절감효과가 큰 일부 Wave만 최적화하는 ML-Adaptive 전략을 held-out test set에서 평가합니다.
7. Streamlit 디지털 트윈에서 원본/개선 경로와 모델 선택 근거를 시각화합니다.

## 핵심 연구 질문

> 모든 Wave에 무거운 경로 최적화를 수행하지 않고, ML이 실제 절감효과가 큰 Wave를 미리 선택해도 대부분의 절감량을 포착할 수 있는가?

## 데이터와 거리 정의

- 데이터: Mendeley Footwear Warehouse Order Picking Dataset
- 실제로 보존되는 항목: Wave ID, 작업 순서, 위치, 수량, SKU, 작업자
- 좌표: 원본 Cartesian 위치를 기존 전처리에서 20×20 시뮬레이터의 walkable grid로 매핑
- 거리: shelf와 corridor를 고려한 simulator grid shortest-path step
- 공정성: Original과 Optimized 모두 **같은 고유 물리 피킹 지점 집합**을 사용하며 순서만 변경
- 시간: timestamp가 없으면 실제 운영시간이 아니라 capacity-calibrated timeline으로 표기

## Windows PowerShell 빠른 실행

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

원본 CSV 전처리:

```powershell
python -m scripts.prepare_footwear_data `
  --dataset-dir ".\data\raw\footwear" `
  --minutes-per-wave 60
```

전체 파이프라인:

```powershell
python -m scripts.run_wave_optimizer_pipeline `
  --tasks ".\data\footwear_replay_tasks.csv" `
  --coverage 0.20
```

PPT용 수치와 표 export:

```powershell
python -m scripts.export_wave_optimizer_presentation_data
```

대시보드:

```powershell
streamlit run .\dashboard\wave_optimizer_app.py
```

테스트:

```powershell
python -m pytest -q .\tests
```

검증된 빌드에서는 33개 테스트가 통과합니다.

## 생성 파일

### 데이터

- `data/wave_optimization_dataset.csv`
- `data/wave_optimization_dataset.summary.json`

### 모델

- `models/wave_benefit_model.joblib`

### 실험 결과

- `runtime/wave_optimizer/wave_model_leaderboard.csv`
- `runtime/wave_optimizer/wave_adaptive_policy_leaderboard.csv`
- `runtime/wave_optimizer/wave_test_predictions.csv`
- `runtime/wave_optimizer/wave_feature_importance.csv`
- `runtime/wave_optimizer/wave_explanations.csv`
- `runtime/wave_optimizer/wave_optimization_report.json`

### PPT 전달 자료

- `runtime/wave_optimizer/PRESENTATION_NUMBERS_KO.md`
- `runtime/wave_optimizer/presentation_data_quality.csv`
- `runtime/wave_optimizer/presentation_zone_summary.csv`
- `runtime/wave_optimizer/presentation_wave_distribution.csv`
- `runtime/wave_optimizer/presentation_feature_dictionary.csv`
- `runtime/wave_optimizer/presentation_route_summary.csv`

## 모델 선택 원칙

- Train 70%, Validation 15%, Test 15% chronological split
- Test는 모델 선택에 사용하지 않음
- 1차: Validation MAE 최소
- 동률: Captured Savings@20%가 큰 모델
- 추가 동률: 추론시간이 짧은 모델

## 비교 대상

### 지도학습

- Dummy Mean
- Ridge
- Random Forest
- HistGradientBoosting
- XGBoost

### 비지도 및 운영 baseline

- Random Expected
- Longest Original Route heuristic
- Isolation Forest
- Local Outlier Factor
- Optimize All

## 중요한 제한

- bounded 2-opt는 전역 최적해를 보장하지 않습니다.
- Grid shortest-path distance는 실제 창고의 미터 단위 거리가 아닙니다.
- Estimated picking time은 시뮬레이션 기반입니다.
- ML 성능과 절감률은 반드시 실제 생성 결과를 사용해 발표해야 합니다.

자세한 실행 순서는 `QUICKSTART_WAVE_OPTIMIZER_KO.md`, 데이터 분석 발표 자료는 `DATA_ANALYSIS_PPT_HANDOFF_KO.md`를 참고하세요.
