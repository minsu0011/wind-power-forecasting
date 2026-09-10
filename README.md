# BARAM — 기상 예보 기반 풍력 발전량 예측

기상 예보로 KPX 세 그룹의 시간별 풍력 발전량을 예측하는 DACON 대회 프로젝트입니다. 예측 오차뿐 아니라 정산 기준의 경계를 함께 고려하고, 예보가 실제로 공개된 시점과 발전량 target의 단위를 맞추는 데 중점을 뒀습니다.

## 문제와 접근

풍속이 같아도 설비용량과 운전 조건이 다르면 발전량이 달라집니다. 기상 특징으로 각 그룹을 예측하되 여러 그룹을 함께 학습할 때는 capacity factor로 단위를 맞췄습니다. 예측은 다시 그룹 설비용량을 곱해 발전량으로 복원합니다.

## 사용 기술

Python, pandas, NumPy, LightGBM, scikit-learn, SciPy, PyArrow를 중심으로 사용합니다. CatBoost·XGBoost는 추가 후보 탐색에 포함됩니다.

## 데이터

대회에서 제공한 LDAPS/GFS 기상 예보, KPX 그룹별 발전량과 설비 정보를 사용합니다. 풍속·기상 공간 요약·예보 시간·결측 정보를 특징으로 구성합니다. 실제 관측이 나중에 공개된 경우 예측시점에 이용할 수 있었던 예보와 바꿔 쓰지 않습니다.

개발에는 2022 학습→2023 검증을 사용했고, 이후 2024 검증과 2025 공개 평가로 이어졌습니다. 기간과 그룹 구성이 다르므로 이 점수를 하나의 개선표로 합치지 않습니다. 원본 데이터는 저장소에 포함하지 않습니다.

## 모델 구조

- L1 후보: 절대 오차를 줄이는 기본 발전량 예측
- Quantile 후보: 예측 분포의 다른 위치를 학습해 손실·정산 기준의 비대칭을 다루는 후보
- Shared 후보: 그룹을 합쳐 capacity factor로 학습하고 그룹별 용량으로 복원
- 후처리: 고정된 조합과 보정으로 최종 발전량을 구성

```text
공개시점이 맞는 기상 예보 → 특징 → 그룹별 / shared 모델
                                      ↓
                            capacity factor → 발전량
                                      ↓
                              고정 후처리 → 평가
```

## 개발 과정

### 오차 하나에서 정산 기준까지

평가는 정규화 절대오차와 발전량으로 가중한 정산금 비율을 함께 봅니다. 오차가 6%·8% 경계를 넘는지가 중요해 L1과 quantile 후보를 함께 비교했습니다. 정산 지표를 단순히 조건에 든 행의 비율로 계산하지 않도록 metric을 분리했습니다.

### Shared target 단위 불일치 수정

개발 recipe에서는 capacity factor를 사용했지만 후속 trainer가 shared target을 kWh로 학습한 불일치가 발견됐습니다. Shared 모델의 target과 용량 복원을 원래 recipe에 맞췄습니다. 이 수정으로 얻은 2024 결과는 이미 정답을 본 검증 구간의 재계산이지 새로운 독립 검증이 아닙니다.

### 추가 데이터가 항상 도움이 되지는 않음

SCADA 풍속을 보조로 쓰는 경로는 당시 2022→2023 비교에서 기준보다 종합점수가 약 0.000981 낮아 recipe에서 제외했습니다. 특징을 더 넣는 것보다 실제 예측시점에 사용 가능한 정보인지, 같은 검증에서 도움이 되는지가 우선입니다.

### Public 후처리의 경계

2025 공개 평가에서는 전역 scale을 바꾸는 후처리를 살폈습니다. 당시 기록의 0.97 scale 점수는 약 0.61371입니다. 공개 피드백을 사용한 선택이므로 private 구간의 성능을 보장하지 않습니다. 공개 구간의 숨은 구성까지 역산하는 세분화 탐색은 사용하지 않는 방향으로 정리했습니다.

## 결과를 읽는 기준

2023은 개발 선택, 2024는 이미 사용한 검증, 2025는 공개 평가입니다. 보존된 기록을 설명하는 수치이며 현재 순위나 최종 비공개 성능을 뜻하지 않습니다. 그룹별 성능 차이와 기상 자료의 publication 시점 확인이 주요 한계로 남습니다.

## 코드와 실행

[src/metric.py](src/metric.py)는 점수 계산, [scripts/build_features.py](scripts/build_features.py)는 특징 생성의 시작점입니다. [configs](configs)에 실험 설정이 있습니다.

```bash
pip install -r requirements.txt
python scripts/build_features.py --raw-dir data/local/open --cache-dir artifacts/cache
python scripts/train_dev.py --help
python -m pytest tests/test_metric.py -q
```

`scripts/train_dev.py`가 개발 학습의 진입점입니다. 실제 학습에는 대회 원본과 특징 캐시가 필요합니다. 검증 결과를 이미 본 기간을 다시 독립 평가로 사용하지 않습니다.

[개발 과정](docs/wiki/Development-Journey.md) · [모델과 target](docs/wiki/Model-Evolution.md) · [병목](docs/wiki/Bottlenecks-and-Solutions.md) · [평가와 결과](docs/wiki/Validation-and-Results.md)
