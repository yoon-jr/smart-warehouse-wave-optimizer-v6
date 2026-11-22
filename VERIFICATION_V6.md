# v6 검증 기록

## 정적 검증

```text
python -m compileall smart_warehouse scripts dashboard
PASS
```

## 자동 테스트

```text
python -m pytest -q
33 passed
```

검증 항목:

- NN 이후 2-opt가 경로를 악화시키지 않음
- 최종 optimizer가 recorded route를 악화시키지 않음
- target/post-optimization 값이 model feature에 포함되지 않음
- chronological split 순서 보존
- adaptive policy metrics 범위 검증
- end-to-end model artifact 및 결과 CSV 생성
- 기존 v5 테스트 regression 없음

## CLI smoke test

합성 120-Wave fixture에서 다음을 실행했습니다.

```text
python -m scripts.run_wave_optimizer_pipeline ...
PASS
```

확인된 흐름:

```text
tasks CSV
→ Wave dataset
→ model leaderboard
→ selected model refit
→ test evaluation
→ Isolation Forest / LOF comparison
→ adaptive policy leaderboard
→ model artifact
→ PPT export
```

합성 fixture의 성능 수치는 코드 연결 검증용이며 실제 Footwear 성능 주장에 사용하지 않습니다.
