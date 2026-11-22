# Lifelong MAPFUA v5 실행 가이드 — Windows PowerShell

모든 명령은 프로젝트 최상위 폴더에서 실행합니다.

## 1. 환경 구성

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-marl.txt
```

새 PowerShell 창에서는 다시 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Footwear 데이터 전처리

원본 파일을 아래에 둔 경우:

```text
data\raw\footwear\Picking_Wave.csv
data\raw\footwear\Storage_Location.csv
```

현재 실험에서 사용한 보정 시간축을 재생성하려면:

```powershell
python -m scripts.prepare_footwear_data `
  --dataset-dir ".\data\raw\footwear" `
  --minutes-per-wave 60
```

확인할 출력:

```text
Prepared tasks
Location join rate
Time basis
```

`footwear_manifest.json`에서 실제 timestamp인지 보정 시간축인지 반드시 확인합니다.

## 3. 자동 테스트

```powershell
python -m pytest -q
```

배포 전에 전체 테스트가 통과해야 합니다.

## 4. 규칙 baseline 먼저 실행

```powershell
python -m scripts.run_mapfua_experiment `
  --ticks 1200 `
  --split test `
  --seeds 42 43 44 45 46 `
  --policies idle_stay nearest_parking density_clear forecast_rule
```

결과:

```text
runtime\mapfua_experiment_detail.csv
runtime\mapfua_experiment_summary.csv
runtime\mapfua_experiment_comparisons.json
```

이 단계에서 확인할 것:

- 모든 정책이 같은 seed에서 같은 WMS window를 사용했는가
- final vertex/edge collision이 발생하지 않았는가
- `density_clear`가 실제로 idle move를 만들었는가
- idle move 비용이 baseline 개선보다 지나치게 크지 않은가

## 5. Forecast 없는 VDN 학습

빠른 기능 확인:

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 10 `
  --episode-ticks 300 `
  --model-out ".\models\mapfua_vdn_no_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_no_forecast_metrics.json"
```

최종 학습 시작점:

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --model-out ".\models\mapfua_vdn_no_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_no_forecast_metrics.json"
```

## 6. Forecast 포함 VDN 학습

먼저 기존 수요 예측 모델이 필요합니다.

```powershell
python -m scripts.train_models
```

그다음:

```powershell
python -m scripts.train_mapfua_vdn `
  --episodes 300 `
  --episode-ticks 600 `
  --validation-interval 25 `
  --include-forecast `
  --model-out ".\models\mapfua_vdn_with_forecast.pt" `
  --metadata-out ".\models\mapfua_vdn_with_forecast_metrics.json"
```

## 7. 전체 ablation

```powershell
python -m scripts.run_mapfua_ablation `
  --ticks 1200 `
  --split test `
  --seeds 100 101 102 103 104 105 106 107 108 109 `
  --no-forecast-model ".\models\mapfua_vdn_no_forecast.pt" `
  --forecast-model ".\models\mapfua_vdn_with_forecast.pt"
```

결과:

```text
runtime\mapfua_ablation_detail.csv
runtime\mapfua_ablation_summary.csv
runtime\mapfua_ablation_results.json
```

평가 seed는 학습 seed와 분리하는 것이 좋습니다.

## 8. 특정 모델 하나를 baseline과 비교

```powershell
python -m scripts.run_mapfua_experiment `
  --ticks 1200 `
  --split test `
  --seeds 100 101 102 103 104 `
  --policies idle_stay density_clear vdn_marl `
  --vdn-model ".\models\mapfua_vdn_no_forecast.pt"
```

신규 v5 checkpoint는 모델 metadata에서 forecast 사용 여부를 자동으로 읽습니다.

## 9. Pygame 데모

규칙 기반 병목 해소:

```powershell
python run_mapfua_demo.py `
  --policy density_clear `
  --split test `
  --seed 42 `
  --max-ticks 1200
```

Forecast 규칙:

```powershell
python run_mapfua_demo.py --policy forecast_rule --split test --seed 42
```

학습된 VDN:

```powershell
python run_mapfua_demo.py `
  --policy vdn_marl `
  --vdn-model ".\models\mapfua_vdn_no_forecast.pt" `
  --split test `
  --seed 42
```

조작:

```text
SPACE  일시정지/재개
R      같은 workload window로 초기화
ESC    종료
```

## 10. 최종 발표 전 체크리스트

- 학습 전용 모델과 test 결과가 분리됨
- seed별 상세 결과를 저장함
- 평균뿐 아니라 P95와 seed별 승률을 확인함
- idle 이동거리와 움직인 idle robot 수도 같이 보고함
- decision latency와 wall time을 같이 보고함
- proxy metric을 실제 지연으로 표현하지 않음
- synthetic/calibrated time이라는 데이터 한계를 공개함
- 최종 충돌 수와 safety shield 개입 횟수를 보고함
