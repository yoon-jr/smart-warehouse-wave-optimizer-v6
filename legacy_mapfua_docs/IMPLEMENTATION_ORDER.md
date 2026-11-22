# 구현 및 실험 순서 — v5 Lifelong MAPFUA

## 1. 환경과 데이터

```powershell
python -m pip install -r requirements-marl.txt
python -m scripts.prepare_footwear_data --dataset-dir ".\data\raw\footwear" --minutes-per-wave 60
python -m pytest -q
```

목표:

- 전처리 성공
- join rate 확인
- manifest의 time basis 확인
- 테스트 전부 통과

## 2. 규칙 baseline

```powershell
python -m scripts.run_mapfua_experiment `
  --ticks 1200 `
  --split test `
  --seeds 42 43 44 45 46 `
  --policies idle_stay nearest_parking density_clear forecast_rule
```

먼저 MARL 없이 유휴 로봇 제어가 실제 차이를 만드는지 확인합니다. baseline 차이가 전혀 없다면 workload window 또는 병목 정의를 먼저 점검해야 합니다.

## 3. 작은 VDN smoke training

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 10 `
  --episode-ticks 300 `
  --model-out ".\models\smoke_vdn.pt" `
  --metadata-out ".\models\smoke_vdn_metrics.json"
```

목표는 성능이 아니라 학습 loop와 checkpoint 저장 확인입니다.

## 4. No-Forecast VDN 본 학습

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --model-out ".\models\mapfua_vdn_no_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_no_forecast_metrics.json"
```

검사:

- reward 추세
- validation completed/wait/distance
- epsilon 감소
- low→medium→high curriculum 진행
- best checkpoint episode

## 5. Forecast 모델과 Forecast VDN

```powershell
python -m scripts.train_models
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --include-forecast `
  --model-out ".\models\mapfua_vdn_with_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_with_forecast_metrics.json"
```

## 6. Unseen test ablation

학습에 쓰지 않은 seed를 사용합니다.

```powershell
python -m scripts.run_mapfua_ablation `
  --ticks 1200 `
  --split test `
  --seeds 100 101 102 103 104 105 106 107 108 109
```

## 7. 결과 해석 순서

1. completed/throughput이 악화되지 않았는가
2. average/P95 wait이 개선됐는가
3. 개선을 위해 idle distance와 NUA를 지나치게 사용하지 않았는가
4. blocking proxy가 감소했는가
5. unnecessary idle move가 증가하지 않았는가
6. conflict/deadlock이 안전하게 처리됐는가
7. decision latency가 허용 범위인가
8. forecast 포함 모델이 forecast 없는 모델보다 실제로 유리한가

## 8. 시각화

```powershell
python run_mapfua_demo.py --policy density_clear --split test --seed 42
python run_mapfua_demo.py --policy vdn_marl --vdn-model ".\models\mapfua_vdn_no_forecast.pt" --split test --seed 42
```

## 9. 추가 실험 순서

1. reward coefficient ablation
2. episode length 300/600/1200
3. robot count별 별도 학습 4/8/12
4. 고밀도 workload window 집중 평가
5. 장애·지연 stress test
