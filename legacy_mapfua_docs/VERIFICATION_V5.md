# v5 Verification Report

## 자동 검증

- Python source compilation: PASS
- pytest: PASS
- 테스트 수: 27

검증 범위:

- A* 기본 동작
- collision-aware reservation planning
- vertex/edge conflict 검출
- allocator
- 기존 adaptive policy
- Footwear 전처리 alias와 output
- MAPFUA policies와 metrics
- temporal split와 environment
- matched-window experiment logic

## 합성 fixture E2E smoke test

작은 합성 Footwear CSV로 다음 pipeline을 실행했습니다.

1. 데이터 전처리
2. heuristic MAPFUA experiment
3. 2-episode VDN training
4. saved VDN evaluation

모든 단계가 종료되고 output 파일을 생성했습니다.

## 성능 해석 주의

2-episode 모델은 코드 연결을 확인하기 위한 **미학습 smoke model**입니다. baseline 대비 성능을 주장하는 용도가 아닙니다.

## 이 환경에서 검증하지 못한 항목

- 사용자의 전체 191,583-task 데이터로 최종 MARL 학습
- 장시간 300+ episode 학습 결과
- 실제 GPU/CPU 환경의 최종 latency benchmark
- 실제 로봇 또는 ROS 연동
- Pygame 창의 육안 확인: 현재 빌드 컨테이너에는 pygame 런타임이 설치되지 않아 정적 compile만 확인

## 최종 주장 가능 시점

다음이 완료된 뒤에만 MARL 성능 우위를 주장합니다.

- 학습/validation/test workload 분리
- no-forecast와 with-forecast 모델 충분한 학습
- unseen test seed 10개 이상
- seed별 결과와 평균·분산 확인
- 운영 성능과 계산비용 동시 보고
