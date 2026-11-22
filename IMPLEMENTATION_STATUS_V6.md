# v6 구현 상태

## 완료

- Footwear WMS 기존 전처리 재사용
- Grid shortest-path distance oracle
- Recorded route / NN / bounded 2-opt 비교
- Original보다 나빠지지 않는 route fallback
- Wave-level feature engineering
- 실제 route saving label 자동 생성
- Chronological 70/15/15 split
- Dummy / Ridge / Random Forest / HistGradientBoosting / XGBoost 비교
- Validation 기반 최종 모델 선택
- Isolation Forest / LOF / random / heuristic baseline
- Captured Savings / End-to-End Reduction 평가
- Model artifact 저장
- Feature importance와 reason code
- Streamlit Before/After 디지털 트윈
- PPT-ready CSV와 자동 수치 문서 export
- 33개 자동 테스트

## 실제 데이터 실행 후 확인할 것

- 위치 결합률
- 실제 Wave 수
- saving_pct 분포
- 선택된 최종 모델
- Validation/Test MAE 차이
- ML-Adaptive와 longest-route heuristic 차이
- Isolation Forest baseline과 supervised model 차이
- 상위 Wave의 경로가 화면에서 실제로 구분되는지

## 주장할 수 없는 것

- 2-opt가 전역 최적해임
- Grid step이 실제 미터임
- Estimated time이 실제 작업시간임
- 실제 현장 congestion까지 완전 재현함
