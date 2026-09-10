# 실행과 데이터 준비

```bash
pip install -r requirements.txt
python -m pytest tests/test_metric.py -q
```

대회 원본은 info.xlsx와 train/test의 LDAPS·GFS, 발전량 label 등 요구 구조를 맞춰 준비합니다. 특징 생성에는 원본 경로와 새 출력 캐시 경로를 지정합니다.

```bash
python scripts/build_features.py --raw-dir data/local/open --cache-dir artifacts/cache
```

캐시 생성 후 [train_dev.py](../../scripts/train_dev.py)의 인자로 개발 실험을 실행할 수 있습니다. 원본 입력이 없는 상태에서 모델 결과를 자동 생성하지 않습니다.

2024 gate는 이미 사용한 검증입니다. 단순 실행 확인을 위해 새 gate처럼 다시 열지 않습니다. 모델 선택·공개 후처리·최종 성능 검증의 역할을 구분해 사용합니다.

[특징 생성 코드](../../scripts/build_features.py) · [평가 코드](../../src/metric.py) · [기존 모델 계보](../model-evolution.md)
