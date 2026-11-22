# v5 구현 우선순위 및 상태

## P0 — 충돌 모델 검증 및 안전 계획기: 완료

- [x] vertex conflict 정의와 테스트
- [x] edge-swap conflict 정의와 테스트
- [x] time reservation table
- [x] space-time A* with wait
- [x] prioritized planning
- [x] synchronous safety shield
- [x] 고유한 로봇 초기 위치

현재 제한: prioritized planner이므로 최적성·완전성을 보장하지 않습니다.

## P1 — MARL 없는 MAPFUA baseline: 완료

- [x] Idle-Stay
- [x] Nearest-Parking
- [x] Density-Based Clear
- [x] Forecast-Rule
- [x] matched workload window 실험
- [x] MAPFUA 지표

## P2 — Lifelong MAPFUA 환경: 완료

- [x] CSV replay를 episode 환경에 연결
- [x] 시간순 train/validation/test
- [x] agent observation
- [x] action mask
- [x] 공동 보상
- [x] forecast on/off

## P3 — 경량 MARL: 완료

- [x] parameter-sharing VDN
- [x] joint replay
- [x] target network
- [x] Double-DQN target
- [x] epsilon schedule
- [x] density curriculum
- [x] validation best checkpoint

## P4 — 예측 ablation: 완료

- [x] VDN without forecast
- [x] VDN with forecast
- [x] 동일 test window 비교
- [x] forecast value 결과 JSON

## P5 — 시각화와 팀 문서: 완료

- [x] side-by-side MAPFUA Pygame demo
- [x] 팀 브리핑
- [x] 논문 Future Work 대응표
- [x] 실행 가이드

## P6 — 실제 연구 실험: 팀 환경에서 수행 필요

- [ ] 전체 CSV로 300+ episode 두 모델 학습
- [ ] validation curve 확인
- [ ] test seed 10개 이상
- [ ] reward ablation
- [ ] robot count 4/8/12 실험
- [ ] 정상/밀집/혼합 workload 분석
- [ ] 발표 그래프 생성

## P7 — 선택 확장

- [ ] 로봇 고장
- [ ] 통로 폐쇄
- [ ] 통신·상태 지연
- [ ] forecast corruption
- [ ] GNN 기반 variable fleet
- [ ] stronger MAPF solver baseline
