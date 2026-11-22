# 최신 논문 Future Work → v5 구현 대응표

## 읽는 방법

- **논문이 직접 남긴 과제**: 저자가 논문에서 후속 연구 또는 한계로 명시한 내용
- **v5 반영**: 이번 코드에서 실제로 구현한 부분
- **남은 범위**: 해커톤 이후 또는 실험 단계에서 추가할 부분

| 논문 | 최신 연구가 해결한 것 | 직접 남긴 과제/한계 | v5 반영 | 남은 범위 |
|---|---|---|---|---|
| Felner & Stern, AAAI 2026, MAPFUA | 목표 없는 agent를 정식 문제로 정의, SST/Fuel/NUA 비용 제시 | Lifelong MAPFUA, 계획-실행 교차, 장기 최적화, 동적 sparse/dense 처리 | WMS 연속 작업, idle robot의 STAY/CLEAR/PREPOSITION, 처리량·이동·이동한 유휴 로봇 지표 | 최적/완전 MAPFUA solver, 대규모 밀집도 이론 분석 |
| Giuffrida et al., RAS 2025 | 학습 기반 MAPF를 창고 구조에서 고전 planner와 비교 | hybrid learning+search, ablation, 환경별 tuning, dynamic environment, physical robots | MARL 고수준 정책 + 충돌 인식 planner, baseline ablation, WMS 시간순 replay | 실제 로봇 및 다양한 실제 레이아웃 |
| Phan et al., AAMAS 2026, CACTUS | 경량 MARL과 자동 curriculum | 다른 shielding, 필요한/불필요한 wait 구분, 고밀도, 부분관측, variable agent count | 공유 VDN, action mask, safety shield, 작업 밀도 curriculum, idle move usefulness estimate | 이론적 wait 분류, GNN, 단일 모델의 가변 fleet 일반화 |
| Zheng et al., JAIR 2026, RL-RH-PP | RL로 lifelong planning priority 학습, 고전 planner가 경로 실행 | 작업 할당과 경로의 공동 최적화, 확률 지연/부분 관측 등 | 학습-계획 역할 분리, rolling high-level decision | 통신 지연/고장, task allocation까지 공동학습 |
| Wang et al., AAAI 2025, LNS2+RL | MARL과 LNS를 결합 | — 핵심적으로 hybrid 자체는 이미 연구됨 | 신규성 주장을 hybrid 자체가 아니라 Lifelong MAPFUA로 제한 | LNS/EECBS와 직접 성능 비교 |
| Fu et al., AAAI 2026 | 극고밀도 unassigned-agent block rearrangement | 다른 warehouse operational settings | “unassigned 최초” 주장을 피하고 WMS lifelong fleet 문제로 구분 | 극고밀도 block rearrangement benchmark 비교 |

## 논문별 핵심 이해

### 1. MAPFUA 논문이 가장 중요한 이유

기존 MAPF의 숨은 가정은 모든 로봇에 목표가 있다는 것입니다. 하지만 실제 창고에서는 피킹을 끝낸 뒤 다음 작업을 기다리는 로봇이 생깁니다. 이 로봇도 셀을 점유해 병목을 만들 수 있습니다.

논문은 유휴 로봇을 무조건 움직이라고 하지 않습니다.

- 희소 환경: 정적 장애물로 두어도 우회 가능
- 밀집 환경: 일부 유휴 로봇을 옮기는 것이 중요
- 혼합 환경: 어느 구역이 밀집했는지 동적으로 판단해야 함

논문이 명시한 Lifelong 확장은 이번 프로젝트의 연구 질문과 정확히 맞습니다.

### 2. 창고 실증 논문의 교훈

무작위 맵에서 높은 성능을 낸 MARL이 선반과 좁은 통로가 있는 창고에서도 그대로 잘된다고 볼 수 없습니다. 따라서 이번 프로젝트의 강점은 단순히 MARL을 적용했다는 것이 아니라 실제 WMS 작업의 위치·순서·수량을 보존한 replay에서 비교한다는 점입니다.

### 3. CACTUS의 교훈

복잡한 network와 많은 보상 shaping이 반드시 좋은 것은 아닙니다. 해커톤에서는 경량 공유 정책, curriculum, 명시적 collision prevention이 더 재현 가능하고 설명하기 쉽습니다.

### 4. RL-RH-PP의 교훈

학습 모델에게 모든 공간-시간 계획을 떠넘기지 않고, RL은 고수준 우선순위를 결정하고 planner는 안전 경로를 생성합니다. v5도 동일한 설계 철학을 사용합니다.

## 이번 해커톤에 적절한 연구 범위

### 반드시 포함

1. `Idle-Stay`, `Nearest-Parking`, `Density-Clear`, `Forecast-Rule` baseline
2. `VDN-NoForecast`, `VDN-WithForecast`
3. 같은 WMS window를 정책별로 공유하는 평가
4. average/P95 wait, throughput, total/idle distance, NUA, blocking proxy
5. collision shield 및 vertex/edge conflict 기록
6. 추론시간과 전체 실행시간

### 시간이 남을 때

1. 로봇 수 4/8/12 별도 모델 학습
2. 통로 폐쇄·고장·상태 지연
3. forecast corruption
4. GNN 기반 가변 로봇 수 모델

### 이번 범위에서 제외

1. 수천 대 실시간 최적화
2. 연속 동역학 및 실제 AMR 제어
3. CBS/EECBS 수준의 최적성·완전성 증명
4. 작업 할당과 저수준 경로를 모두 end-to-end MARL로 학습
