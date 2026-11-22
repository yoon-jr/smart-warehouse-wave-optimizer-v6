# 기술 설계 — ML-Adaptive Wave Optimization

## 1. 시스템 구성

```text
Footwear WMS CSV
  ↓
기존 전처리 및 위치 매핑
  ↓
Wave-level route optimization dataset
  ↓
Supervised model leaderboard
  ↓
Selected benefit predictor
  ↓
Top-K Wave selection
  ↓
NN + bounded 2-opt
  ↓
Digital Twin Before / After
```

## 2. Route Optimization

### 입력

Wave의 recorded first-visit unique grid stops.

### 거리

Depot에서 출발해 모든 지점을 방문하고 Depot으로 돌아오는 shortest-path grid distance.

### Nearest Neighbor

현재 위치에서 가장 가까운 미방문 지점을 선택합니다. 동률은 y, x, 원래 index 순으로 deterministic 처리합니다.

### bounded 2-opt

NN 경로의 두 edge를 교체해 더 짧아질 때 구간을 뒤집습니다.

- first-improvement
- 최대 pass 제한
- candidate span 제한
- 계산시간 상한을 둔 해커톤용 구현

### 비악화 보장

```text
optimized = min(recorded, nearest-neighbor, two-opt)
```

전역 최적해는 아니지만 기존 경로보다 길어지는 것을 방지합니다.

## 3. Label

```text
saving_pct = 100 × (original - optimized) / original
```

Wave에 최적화를 실제 실행해 얻은 label입니다.

## 4. Leakage Control

모델 입력은 최적화 전에 계산 가능한 값만 포함합니다.

금지:

- optimized_route_distance
- distance_saved
- saving_pct
- NN distance

NN distance도 target 계산과 거의 같은 최적화 연산을 요구하므로 feature에서 제외했습니다.

## 5. Model Selection

- chronological Train / Validation / Test
- Validation MAE 기준 선택
- selected model만 Train+Validation 재학습
- Test one-shot evaluation

## 6. Adaptive Policy Evaluation

모든 정책에 동일한 coverage를 부여합니다.

- Random Expected
- Longest Route
- Isolation Forest
- LOF
- Selected supervised model
- Optimize All

핵심 지표:

```text
Captured Savings
= 선택된 Wave의 실제 절감거리 / 전체 Wave의 가능한 절감거리

End-to-End Reduction
= 선택된 Wave의 실제 절감거리 / 전체 Original 거리
```

## 7. Explainability

SHAP 대신 빠르고 검증 가능한 reason code를 사용합니다.

- training reference percentile
- original distance
- stop count
- spatial span
- zone transition
- footprint area

이는 feature contribution의 엄밀한 인과 설명은 아니며, 운영자가 선택된 Wave의 특성을 이해하기 위한 transparent decision trace입니다.

## 8. Complexity

- Grid all-pairs shortest path: 작은 20×20 map에서 한 번만 계산
- NN: Wave당 O(n²)
- bounded 2-opt: O(p × n × span)
- ML inference: Wave당 수 ms 이하 기대

## 9. 제한

- 실제 창고 aisle topology가 20×20 simulator로 단순화됨
- 동시 다중 picker congestion은 route-label 단계에 직접 포함하지 않음
- 2-opt는 heuristic
- 실제 시간 및 에너지 검증 필요
