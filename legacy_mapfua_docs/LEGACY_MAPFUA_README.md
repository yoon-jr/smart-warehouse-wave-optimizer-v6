# Lifelong MAPFUA Smart Warehouse v5

실제 Footwear WMS 작업 스트림을 이용해 **작업이 없는 유휴 로봇이 언제 정지하고, 병목을 비우고, 수요 구역으로 선배치되어야 하는지** 연구하는 해커톤 프로젝트입니다.

## 핵심 연구 질문

> 동적으로 작업이 계속 들어오는 창고에서 유휴 로봇의 `STAY`, `CLEAR`, `PREPOSITION` 행동을 협력형 MARL로 학습하면 고정 규칙보다 대기시간·이동량·병목의 trade-off를 더 잘 조정할 수 있는가?

이 프로젝트는 MARL이 모든 한 칸 이동을 직접 결정하도록 하지 않습니다.

- **MARL/heuristic**: 유휴 로봇의 고수준 행동 선택
- **deterministic allocator**: 발생한 작업을 로봇에 배정
- **reservation-table space-time planner**: 충돌 인식형 경로 생성
- **synchronous safety shield**: 실행 직전 vertex/edge 충돌 차단
- **forecast model**: 선택적 관측 및 ablation 입력

자세한 연구 배경은 [`TEAM_BRIEFING_KO.md`](TEAM_BRIEFING_KO.md), 논문 대응은 [`PAPER_FUTURE_WORK_MATRIX_KO.md`](PAPER_FUTURE_WORK_MATRIX_KO.md)를 참고하세요.

---

## 1. 최신 연구와의 연결

AAAI 2026의 MAPFUA 논문은 목표가 없는 로봇을 그대로 둘지, assigned robot의 길을 비우기 위해 이동시킬지를 별도 문제로 정의했습니다. 저자들은 **Lifelong MAPFUA**, 계획-실행 교차, 장기 최적화, 동적 sparse/dense 처리 등을 후속 연구로 제시했습니다.

v5는 이를 다음과 같이 구체화합니다.

```text
WMS-derived lifelong task replay
+ idle robot STAY/CLEAR/PREPOSITION decisions
+ parameter-sharing VDN
+ collision-aware classical planner
+ forecast/no-forecast ablation
```

세계 최초, 최적 MAPFUA solver, Amazon보다 우수하다는 주장은 하지 않습니다.

---

## 2. 주요 기능

### MAPFUA baseline

- `idle_stay`
- `nearest_parking`
- `density_clear`
- `forecast_rule`

### MARL

- parameter-sharing VDN
- joint experience replay
- Double-DQN target
- action mask
- low/medium/high workload-density curriculum
- chronological validation checkpoint
- forecast observation on/off

### 안전 계획

- vertex conflict 예약
- edge-swap conflict 예약
- wait action
- rolling planning horizon
- goal hold reservation
- synchronous collision shield

### 평가

- matched WMS task window per seed
- train/validation/test 시간순 분리
- average/P95 wait, throughput, distance
- idle movement, NUA, blocking proxy
- conflict/shield/deadlock
- decision latency와 wall time

---

## 3. 설치

권장 환경: Python 3.11 또는 3.12

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-marl.txt
```

기존 v4 기능만 실행한다면 `requirements.txt`로도 충분하지만, VDN 학습에는 PyTorch가 포함된 `requirements-marl.txt`가 필요합니다.

---

## 4. 데이터 준비

원본 데이터:

- **Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company**
- DOI: `10.17632/pf2w725pw3.1`
- License: CC BY 4.0

필수 파일:

```text
Picking_Wave.csv
Storage_Location.csv
```

예시 위치:

```text
data/raw/footwear/Picking_Wave.csv
data/raw/footwear/Storage_Location.csv
```

전처리:

```powershell
python -m scripts.prepare_footwear_data `
  --dataset-dir ".\data\raw\footwear" `
  --minutes-per-wave 60
```

생성물:

```text
data/footwear_replay_tasks.csv
data/historical_tasks.csv
data/training_table.csv
data/footwear_location_map.csv
data/footwear_manifest.json
```

실제 timestamp가 없으면 wave 순서를 보존한 보정 시간축을 사용합니다. `footwear_manifest.json`의 `time_basis`와 가정을 발표 전에 확인하세요.

---

## 5. 테스트

```powershell
python -m pytest -q
```

패키징 시점 기준 27개 테스트가 통과했습니다. 자세한 내용은 [`VERIFICATION_V5.md`](VERIFICATION_V5.md)를 참고하세요.

---

## 6. 규칙 baseline 평가

```powershell
python -m scripts.run_mapfua_experiment `
  --ticks 1200 `
  --split test `
  --seeds 42 43 44 45 46 `
  --policies idle_stay nearest_parking density_clear forecast_rule
```

출력:

```text
runtime/mapfua_experiment_detail.csv
runtime/mapfua_experiment_summary.csv
runtime/mapfua_experiment_comparisons.json
```

각 seed는 하나의 chronological workload window를 선택하며, 같은 seed의 모든 정책은 동일한 작업을 받습니다.

---

## 7. VDN 학습

### Forecast 없는 모델

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --model-out ".\models\mapfua_vdn_no_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_no_forecast_metrics.json"
```

### Forecast 포함 모델

수요 예측 모델을 먼저 학습합니다.

```powershell
python -m scripts.train_models
```

그다음:

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --include-forecast `
  --model-out ".\models\mapfua_vdn_with_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_with_forecast_metrics.json"
```

기본 curriculum:

```text
초기 25% episode: low-load WMS windows
다음 35%: medium-load windows
나머지 40%: high-load windows
```

`--no-density-curriculum`으로 비활성화할 수 있습니다.

---

## 8. 최종 ablation

```powershell
python -m scripts.run_mapfua_ablation `
  --ticks 1200 `
  --split test `
  --seeds 100 101 102 103 104 105 106 107 108 109 `
  --no-forecast-model ".\models\mapfua_vdn_no_forecast.pt" `
  --forecast-model ".\models\mapfua_vdn_with_forecast.pt"
```

정책:

```text
idle_stay
nearest_parking
density_clear
forecast_rule
vdn_no_forecast
vdn_with_forecast
```

결과:

```text
runtime/mapfua_ablation_detail.csv
runtime/mapfua_ablation_summary.csv
runtime/mapfua_ablation_results.json
```

---

## 9. Pygame 데모

```powershell
python run_mapfua_demo.py --policy density_clear --split test --seed 42
```

학습 모델:

```powershell
python run_mapfua_demo.py `
  --policy vdn_marl `
  --vdn-model ".\models\mapfua_vdn_no_forecast.pt" `
  --split test `
  --seed 42
```

왼쪽은 `Idle-Stay`, 오른쪽은 선택 정책이며 같은 WMS task window를 재생합니다.

```text
SPACE pause
R reset
ESC quit
```

기존 v4의 Reactive vs Adaptive Hybrid 화면도 남아 있습니다.

```powershell
python run_comparison.py
```

---

## 10. 프로젝트 구조

```text
smart_warehouse/
  reservations.py      collision-aware space-time planner
  idle_policy.py       MAPFUA actions and rule baselines
  simulation.py        lifelong simulator and metrics
  mapfua_env.py        MARL environment and chronological split
  marl_vdn.py          parameter-sharing VDN learner

scripts/
  prepare_footwear_data.py
  train_models.py
  train_mapfua_vdn.py
  run_mapfua_experiment.py
  run_mapfua_ablation.py

run_mapfua_demo.py      side-by-side visual demo
```

---

## 11. 중요한 한계

- prioritized space-time planner는 최적/완전 MAPFUA solver가 아님
- discrete grid와 discrete time 사용
- VDN checkpoint는 학습 시의 고정 robot count를 요구
- blocking/usefulness 지표 일부는 proxy 또는 heuristic estimate
- 실제 timestamp가 없는 데이터에서는 calibrated replay time 사용
- 실제 로봇 및 연속 동역학 검증 없음
- 최종 학습 모델은 패키지에 포함하지 않음

코드 pipeline 검증과 성능 우위 검증은 구분해야 합니다. 충분히 학습한 모델을 unseen test window에서 평가하기 전에는 MARL이 baseline보다 우수하다고 주장하지 않습니다.
