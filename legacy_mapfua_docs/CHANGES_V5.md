# Changes in v5 — Lifelong MAPFUA

## 연구 범위 변경

v4의 핵심은 `Reactive + forecast-based prepositioning`이었습니다. v5는 최신 MAPFUA 연구의 Lifelong 확장을 구현하는 방향으로 전환했습니다.

```text
v4: 예측 신호가 강할 때 유휴 로봇 선배치
v5: 유휴 로봇이 STAY / CLEAR / PREPOSITION 중 무엇을 해야 하는지 학습
```

수요 예측은 메인 알고리즘이 아니라 선택적 관측과 ablation 변수가 되었습니다.

## 새 안전 계획 계층

- reservation-table 기반 prioritized space-time A*
- wait action 지원
- vertex conflict 예약
- edge-swap conflict 예약
- 목표 셀 hold horizon
- 동기식 실행 collision shield
- conflict, shield wait, deadlock, path failure 계측

## 새 MAPFUA 정책

- `idle_stay`
- `nearest_parking`
- `density_clear`
- `forecast_rule`
- `external_marl`

고수준 행동:

- `STAY`
- `CLEAR`
- `ZONE_A/B/C/D`

## 새 MARL 환경

- WMS replay의 시간순 70/15/15 분할
- release-tick-aligned episode window
- per-agent observation
- action mask
- team reward
- 저/중/고 workload-density curriculum
- forecast observation on/off

## 새 VDN 학습기

- parameter-sharing agent network
- joint replay buffer
- VDN value factorization
- Double-DQN target action selection
- target network update
- epsilon-greedy exploration
- validation checkpoint
- model metadata에 robot count, observation dimension, forecast 설정 저장

## 새 실험 스크립트

- `scripts/train_mapfua_vdn.py`
- `scripts/run_mapfua_experiment.py`
- `scripts/run_mapfua_ablation.py`
- `run_mapfua_demo.py`

## 새 지표

- idle reposition distance
- number of idle move events
- number of distinct idle robots moved
- clear/preposition/parking event counts
- estimated useful/unnecessary idle moves
- blocking pressure proxy
- vertex/edge conflicts
- shield waits/deadlocks
- path planning failures
- task assignment deferrals
- high-level decision latency 평균/P95

## 호환성

- 기존 Footwear 전처리 및 예측 모델 학습 유지
- 기존 v4 Reactive/Adaptive Hybrid 실험과 Pygame 비교 유지
- 기존 테스트 유지
