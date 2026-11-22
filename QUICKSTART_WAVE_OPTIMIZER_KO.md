# WareFlow AI v6 실행 순서 — Windows PowerShell

## 0. 프로젝트 폴더 이동

```powershell
Set-Location "$HOME\Desktop\smart-warehouse-wave-optimizer-v6"
```

## 1. 가상환경 생성 및 활성화

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Python 확인:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

## 2. 패키지 설치

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

확인:

```powershell
python -c "import pandas, sklearn, xgboost, streamlit, plotly; print('packages: OK')"
```

## 3. 원본 Footwear CSV 준비

다음 파일이 필요합니다.

```text
 data\raw\footwear\Picking_Wave.csv
 data\raw\footwear\Storage_Location.csv
```

기존 프로젝트에서 복사하는 경우:

```powershell
New-Item -ItemType Directory -Force ".\data\raw\footwear" | Out-Null

Copy-Item `
  "$HOME\Desktop\smart-warehouse-mapfua-v5\data\raw\footwear\Picking_Wave.csv" `
  ".\data\raw\footwear\Picking_Wave.csv" `
  -Force

Copy-Item `
  "$HOME\Desktop\smart-warehouse-mapfua-v5\data\raw\footwear\Storage_Location.csv" `
  ".\data\raw\footwear\Storage_Location.csv" `
  -Force
```

## 4. 기존 WMS 전처리

```powershell
python -m scripts.prepare_footwear_data `
  --dataset-dir ".\data\raw\footwear" `
  --minutes-per-wave 60
```

생성 확인:

```powershell
Get-ChildItem ".\data\footwear_replay_tasks.csv"
Get-Content ".\data\footwear_manifest.json" -Raw
```

주의:

- `time_basis = fixed_wave_sequence`는 오류가 아닙니다.
- 원본에 신뢰할 수 있는 timestamp가 없을 때 Wave 순서를 보존한 보정 시간축을 사용합니다.
- 발표에서는 `capacity-calibrated timeline`이라고 표현합니다.

## 5. 자동 테스트

```powershell
python -m pytest -q .\tests
```

정상 결과:

```text
33 passed
```

## 6. 전체 ML + 최적화 파이프라인 실행

```powershell
python -m scripts.run_wave_optimizer_pipeline `
  --tasks ".\data\footwear_replay_tasks.csv" `
  --dataset-output ".\data\wave_optimization_dataset.csv" `
  --output-dir ".\runtime\wave_optimizer" `
  --model-dir ".\models" `
  --coverage 0.20
```

`coverage 0.20`은 전체 Wave 중 예측 절감률 상위 20%만 최적화한다는 뜻입니다.

시간이 매우 촉박할 때는 먼저 일부 Wave만 smoke test합니다.

```powershell
python -m scripts.run_wave_optimizer_pipeline `
  --tasks ".\data\footwear_replay_tasks.csv" `
  --dataset-output ".\data\wave_optimization_dataset_smoke.csv" `
  --output-dir ".\runtime\wave_optimizer_smoke" `
  --model-dir ".\models\smoke" `
  --coverage 0.20 `
  --max-waves 300
```

최종 발표 수치는 `--max-waves` 없이 전체 데이터로 다시 실행한 결과를 사용합니다.

## 7. 결과 확인

모델 비교:

```powershell
Import-Csv ".\runtime\wave_optimizer\wave_model_leaderboard.csv" |
  Format-Table -AutoSize
```

운영 정책 비교:

```powershell
Import-Csv ".\runtime\wave_optimizer\wave_adaptive_policy_leaderboard.csv" |
  Format-Table -AutoSize
```

최종 보고서:

```powershell
Get-Content ".\runtime\wave_optimizer\wave_optimization_report.json" -Raw
```

## 8. PPT용 표와 자동 문구 생성

```powershell
python -m scripts.export_wave_optimizer_presentation_data
```

가장 먼저 확인할 파일:

```powershell
Get-Content ".\runtime\wave_optimizer\PRESENTATION_NUMBERS_KO.md" -Raw
```

## 9. Streamlit 디지털 트윈

```powershell
streamlit run .\dashboard\wave_optimizer_app.py
```

브라우저에서 확인할 화면:

1. Selected Model과 Test MAE
2. 전체 가능한 절감량 중 ML이 포착한 비율
3. Model Validation Leaderboard
4. Adaptive Policy Leaderboard
5. 특정 Wave의 Original vs Optimized 경로
6. 실제 절감률과 ML 예측 절감률
7. Feature importance와 Why Optimize explanation

## 10. 발표 직전 결과 보관

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = ".\presentation_results_$stamp"
New-Item -ItemType Directory -Force $backup | Out-Null

Copy-Item ".\runtime\wave_optimizer\*" $backup -Recurse -Force
Copy-Item ".\models\wave_benefit_model.joblib" $backup -Force
Copy-Item ".\data\wave_optimization_dataset.csv" $backup -Force

Write-Host "Saved to $backup"
```

## 핵심 결과 판단법

### 모델 성능

- MAE, RMSE: 낮을수록 좋음
- R², Spearman: 높을수록 좋음
- Test 결과는 모델 선택이 끝난 후 마지막 평가로만 사용

### 운영 성능

- `captured_savings_pct`: 높을수록 좋음
- `end_to_end_distance_reduction_pct`: 높을수록 좋음
- `optimization_workload_reduction_pct`: 높을수록 계산을 덜 수행

가장 좋은 메시지 형태:

```text
전체 Wave 중 20%만 최적화하면서,
Optimize-All이 얻을 수 있는 거리 절감의 상당 부분을 포착했고,
최적화 계산 대상은 80% 줄였다.
```

숫자는 `PRESENTATION_NUMBERS_KO.md`에서 실제 결과를 복사합니다.
