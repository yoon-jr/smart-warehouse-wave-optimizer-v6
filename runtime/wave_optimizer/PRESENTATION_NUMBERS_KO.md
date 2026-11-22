# 발표용 자동 집계 수치

## 데이터
- 전처리 피킹 라인: **191,583개**
- Wave: **9,143개**
- 위치 결합률: **0.890289**
- 시간축: **fixed_wave_sequence**
- 주의: 실제 timestamp가 없으면 시간 관련 표현은 `capacity-calibrated timeline`으로 표기

## 경로 최적화 라벨
- 평균 실제 거리 절감률: **22.67%**
- 중앙값 절감률: **26.09%**
- 경로가 실제로 개선된 Wave 비율: **78.99%**
- 동일한 고유 피킹 지점 집합을 사용하고 순서만 변경했으므로 비교가 공정함

## 모델 선택
- 선택 모델: **hist_gradient_boosting**
- 선택 기준: validation MAE 최저, 동률 시 captured savings와 추론시간 사용
- Validation MAE: **2.315 percentage points**
- Test MAE: **2.492 percentage points**
- Test Spearman: **0.969**

## 운영 성능
- 최적화 대상 비율: **20.0%**
- 전체 가능한 절감량 중 포착: **49.53%**
- 전체 test 이동거리 절감: **12.22%**
- 최적화 연산을 생략한 Wave 비율: **79.96%**

## 발표에서 피해야 할 표현
- 실제 운영시간을 정확히 재현했다
- 최적 경로의 전역 최적해를 보장한다
- Test set에서 가장 좋은 모델을 선택했다

## 권장 표현
- 실제 WMS 작업 순서·위치·수량을 보존한 held-out chronological evaluation
- NN + bounded 2-opt로 생성한 feasible improvement label
- 모든 Wave가 아니라 예상 절감효과가 큰 Wave에 계산을 집중하는 ML-adaptive optimization
