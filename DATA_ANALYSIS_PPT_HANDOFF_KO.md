# 데이터 분석·전처리 PPT 제작 전달 문서

이 문서는 데이터 분석 파트를 맡은 팀원이 코드 전체를 읽지 않고도 슬라이드를 만들 수 있도록 구성했습니다.

---

# 1. 데이터 분석 파트의 핵심 메시지

## 문제 정의

기존 WMS에는 피킹해야 할 Wave, 위치, 수량, 순서가 기록되어 있지만, 해당 순서가 이동거리 최소화를 목적으로 생성됐다고 보장할 수 없습니다.

## 분석 목표

1. Wave별 공간적 복잡도와 기존 이동 비효율을 수치화
2. NN + 2-opt로 얻는 실제 거리 절감률을 자동 라벨로 생성
3. 여러 ML 모델이 최적화 효과가 큰 Wave를 사전에 예측할 수 있는지 비교
4. 전체 Wave가 아니라 효과가 큰 일부 Wave만 최적화해 계산비용 절감

## 한 문장 요약

> WMS 데이터를 단순 예측에 사용하는 것이 아니라, 실제 최적화 결과를 자동 라벨로 만들어 “어떤 Wave를 최적화해야 가장 큰 효과가 나는가”를 학습했다.

---

# 2. 권장 슬라이드 구성

## 슬라이드 1 — 데이터 출처와 원본 구조

### 제목

**Actual WMS Data, Not a Synthetic Order List**

### 넣을 내용

- 데이터셋: Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company
- 핵심 원본 파일
  - `Picking_Wave.csv`
  - `Storage_Location.csv`
- 주요 컬럼
  - `waveNumber`
  - `reference`
  - `Size (US)`
  - `quantityToPick (units)`
  - `locations`
  - `operator`

### 시각화

- 두 CSV가 `locations`를 기준으로 결합되는 흐름도
- 원본 컬럼 5~6개 샘플 표

### 발표 문구

> 실제 WMS의 Wave, SKU, 수량, 위치, 작업자 정보를 사용했고, Storage Location 좌표와 결합해 각 피킹 지점을 공간상에 배치했습니다.

---

## 슬라이드 2 — 데이터 품질과 전처리

### 제목

**From Raw WMS Records to Spatial Pick Waves**

### 전처리 단계

```text
원본 컬럼명 정규화
→ Wave·위치·수량 컬럼 자동 탐지
→ 위치 문자열 대문자·공백 정리
→ Storage Location 좌표 파싱
→ Picking Wave와 위치 좌표 결합
→ 원본 좌표를 20×20 walkable grid로 매핑
→ Wave 순서 및 source_sequence 보존
→ 결측·불일치 행 제거
```

### 반드시 넣을 데이터 품질 지표

`runtime/wave_optimizer/presentation_data_quality.csv`에서 가져옵니다.

- 전처리 피킹 라인 수
- 고유 Wave 수
- 고유 위치 수
- 위치 결합률
- 좌표 결측 수
- 중복 task ID 수
- 시간축 생성 방식

### 중요한 주의

원본에 신뢰할 수 있는 실제 timestamp가 없으면:

- 실제 시간 재현이라고 말하지 않음
- `fixed wave sequence` 또는 `capacity-calibrated timeline`이라고 표현

---

## 슬라이드 3 — 공간 분포 EDA

### 제목

**Where Is the Work Concentrated?**

### 사용할 파일

- `presentation_zone_summary.csv`
- `presentation_wave_distribution.csv`

### 권장 그래프

1. Zone A–D 피킹 라인 비율 막대그래프
2. Wave별 `task_count` 히스토그램
3. `unique_grid_stop_count` 분포
4. `original_route_distance` 분포
5. High-load top 20%, Extreme top 5% 강조

### 발표 문구

> Wave마다 작업량뿐 아니라 공간 분산과 Zone 전환 횟수도 크게 달랐기 때문에, 모든 Wave에 동일한 최적화 계산을 적용하는 것은 비효율적일 수 있습니다.

---

## 슬라이드 4 — 공정한 Before/After 정의

### 제목

**Same Pick Stops, Different Visit Order**

### 가장 중요한 설명

Original과 Optimized는 동일한 고유 피킹 지점 집합을 사용합니다.

```text
Original:
Depot → 기록된 첫 방문 순서 → Depot

Optimized:
Depot → Nearest Neighbor → bounded 2-opt → Depot
```

### 공정성 원칙

- 최적화 경로에서만 작업을 삭제하지 않음
- 중복 피킹 라인은 양쪽 모두 동일하게 하나의 물리 방문 지점으로 처리
- 같은 shelf/corridor grid에서 shortest-path 거리 사용
- 최적화 결과가 Original보다 나쁘면 Original 유지

### 거리 라벨

```text
distance_saved
= original_route_distance - optimized_route_distance

saving_pct
= distance_saved / original_route_distance × 100
```

### PPT 시각화

Streamlit 화면의 빨간 Original 경로와 초록 Optimized 경로를 캡처합니다.

---

## 슬라이드 5 — Wave-level Feature Engineering

### 제목

**Features Available Before Optimization**

### 특성 그룹

#### 규모

- task_count
- total_quantity
- unique_sku_count
- unique_location_count
- operator_count

#### Zone 분포

- zone_a/b/c/d_ratio
- dominant_zone_ratio
- active_zone_count
- zone_entropy

#### 공간 분산

- std_x, std_y
- bounding_box_area
- mean_pairwise_distance
- max_pairwise_distance

#### 기존 동선 복잡도

- original_route_distance
- distance_per_stop
- mean_route_jump
- max_route_jump
- zone_transition_count

### 사용할 파일

`presentation_feature_dictionary.csv`

### 누수 방지 설명

다음 값은 입력으로 사용하지 않습니다.

- optimized_route_distance
- distance_saved
- saving_pct

이 값들은 예측 대상이므로 feature에 포함하면 데이터 누수입니다.

---

## 슬라이드 6 — 자동 라벨 생성

### 제목

**Optimization Creates Its Own Supervised Label**

### 핵심 스토리

위험 Wave에 대한 사람이 만든 정답은 없지만, 각 Wave에 실제 경로 최적화를 적용하면 절감률을 계산할 수 있습니다.

```text
원본 Wave
→ NN + 2-opt 실행
→ 실제 saving_pct 계산
→ supervised regression target 생성
```

### Isolation Forest와의 차이

- Isolation Forest: 다른 Wave와 얼마나 특이한가?
- 지도학습: 이 Wave를 최적화하면 실제로 몇 % 절감되는가?

### 발표 문구

> 이상치 여부가 아니라 운영 효과를 직접 예측하기 위해, 최적화 결과 자체를 정답 라벨로 생성했습니다.

---

## 슬라이드 7 — 데이터 분할과 모델 선택

### 제목

**Chronological Holdout Prevents Future Leakage**

### 분할

```text
앞 70%: Train
다음 15%: Validation
마지막 15%: Test
```

### 비교 모델

- Dummy Mean
- Ridge
- Random Forest
- HistGradientBoosting
- XGBoost

### 선택 규칙

1. Validation MAE 최소
2. 동률이면 Captured Savings@20% 최대
3. 다시 동률이면 추론시간 최소
4. Test는 마지막 한 번만 평가

### 사용할 파일

`wave_model_leaderboard.csv`

### 권장 그래프

- 모델별 Validation MAE 막대그래프
- 모델별 Captured Savings@20% 막대그래프

---

## 슬라이드 8 — 비지도 모델을 왜 같이 비교했는가

### 제목

**Supervised Benefit Prediction vs Unsupervised Anomaly Detection**

### 비교 대상

- Isolation Forest
- Local Outlier Factor
- Longest Route Heuristic
- Random Selection
- Selected Supervised Model
- Optimize All

### 공정 비교

모두 동일하게 test Wave의 상위 20%만 선택합니다.

### 질문

> 같은 최적화 예산으로 어떤 방식이 실제 절감량을 가장 많이 포착하는가?

### 사용할 파일

`wave_adaptive_policy_leaderboard.csv`

---

## 슬라이드 9 — 최종 운영 성능

### 제목

**Optimize Fewer Waves, Capture More Savings**

### 핵심 KPI

- Coverage: 최적화한 Wave 비율
- Captured Savings: 전체 가능한 절감량 중 포착 비율
- End-to-End Distance Reduction
- Optimization Workload Reduction

### 수치는 여기서 복사

`PRESENTATION_NUMBERS_KO.md`

### 권장 메시지 형식

> 전체 Wave 중 [X]%만 최적화하면서 전체 가능한 절감량의 [Y]%를 포착했고, 최적화 대상 Wave를 [Z]% 줄였습니다.

실제 생성 숫자로만 채웁니다.

---

## 슬라이드 10 — 설명 가능성과 디지털 트윈

### 제목

**Why Was This Wave Selected?**

### 화면에 보여줄 요소

- Predicted saving
- Actual saving
- Original / Optimized route overlay
- Top reason 3개
  - recorded route distance 상위 몇 %
  - pickup stop 수 상위 몇 %
  - spatial span 상위 몇 %
  - Zone transition 상위 몇 %

### 사용할 파일

- `wave_explanations.csv`
- Streamlit `dashboard/wave_optimizer_app.py`

---

# 3. 전처리 세부 논리

## 컬럼 탐지

원본 컬럼명이 조금 달라도 alias 기반으로 탐지합니다.

예:

```text
waveNumber / wave_number / wave_id
locations / location / storage_location
quantityToPick / quantity / qty
```

## 위치 정리

- 문자열 대문자 변환
- 앞뒤 공백 제거
- 내부 공백 제거
- Storage_Location 좌표와 left join
- 최소 join rate 이하이면 즉시 오류

## 좌표 변환

1. 원본 X/Y 범위를 min-max 기준으로 정규화
2. 20×20 grid에 비례 매핑
3. shelf 위라면 같은 Zone의 가장 가까운 walkable cell로 이동
4. Zone A–D는 grid 사분면 기준

## Wave 순서

- source_sequence가 존재하면 사용
- 없으면 원본 행 순서 기반 순번 생성
- ML split은 wave_order_index 기준 chronological split

---

# 4. 분석에서 반드시 구분해야 하는 값

## 실제 데이터에서 온 값

- Wave ID
- SKU
- 위치 ID
- 좌표
- 수량
- 작업자
- 피킹 순서

## 모델링으로 생성한 값

- 20×20 grid 좌표
- shortest-path grid distance
- NN 경로
- bounded 2-opt 경로
- estimated picking time
- saving_pct label

발표에서는 두 종류를 혼동하지 않습니다.

---

# 5. 발표자가 받을 예상 질문과 답변

## Q. 왜 Isolation Forest로 고정하지 않았나?

A. 위험 라벨은 없지만, 경로 최적화 결과를 통해 실제 절감률 라벨을 자동 생성할 수 있었습니다. 따라서 운영 목표와 직접 연결되는 supervised regression을 중심으로 두고, Isolation Forest는 label-free baseline으로 공정하게 비교했습니다.

## Q. 왜 XGBoost가 아니라 여러 모델을 비교했나?

A. 표형 데이터에서 특정 모델이 항상 최고라고 가정하지 않았습니다. chronological validation MAE를 기준으로 Dummy, linear, bagging, boosting 계열을 비교하고 데이터에 가장 적합한 모델을 선택했습니다.

## Q. Test 성능을 보고 모델을 선택했나?

A. 아닙니다. Validation에서 모델을 선택한 뒤 Train+Validation으로 재학습하고 Test는 마지막 한 번만 평가했습니다.

## Q. 2-opt가 진짜 최적해인가?

A. 아닙니다. 빠르게 실행 가능한 heuristic입니다. 코드에서는 Original, NN, NN+2-opt 중 가장 짧은 경로를 채택해 최소한 기존 경로를 악화시키지 않도록 했습니다.

## Q. 거리 단위가 실제 미터인가?

A. 아닙니다. 원본 좌표를 simulator grid로 매핑한 shortest-path grid step입니다. 동일 기준의 Before/After 상대 비교에 사용했습니다.

## Q. 실제 피킹 시간도 줄어든다고 볼 수 있나?

A. 이동거리 감소는 직접 계산됐지만 시간은 simulation-based estimate입니다. 실제 현장 검증이 필요하다고 명시합니다.

---

# 6. 팀원별 권장 분업

## 데이터 분석 담당

- 데이터 출처, 컬럼, 전처리 flow
- 결측·join rate·분포 그래프
- Feature engineering 슬라이드

## 모델 담당

- 자동 라벨 생성
- Chronological split
- 모델 leaderboard
- Isolation Forest baseline 비교

## 최적화 담당

- NN와 2-opt 설명
- 공정한 동일 지점 집합 비교
- route distance 및 saving 정의

## 데모 담당

- Streamlit 실행
- 최고 predicted wave 선택
- Before/After route
- KPI와 explanation 캡처

---

# 7. PPT 수치 입력 체크리스트

다음 파일이 모두 생성됐는지 확인합니다.

```text
runtime/wave_optimizer/PRESENTATION_NUMBERS_KO.md
runtime/wave_optimizer/presentation_data_quality.csv
runtime/wave_optimizer/presentation_zone_summary.csv
runtime/wave_optimizer/presentation_wave_distribution.csv
runtime/wave_optimizer/presentation_feature_dictionary.csv
runtime/wave_optimizer/presentation_route_summary.csv
runtime/wave_optimizer/wave_model_leaderboard.csv
runtime/wave_optimizer/wave_adaptive_policy_leaderboard.csv
```

수치를 수동으로 추정하거나 예시 숫자를 사용하지 않습니다.
