# 모델과 평가식

## 입력부터 예측까지

기상 예보는 LDAPS/GFS 원본, 설비용량은 그룹 정보에서 가져옵니다. 특징은 풍속·기상 공간 요약, 시간과 예보 horizon, 결측 정보 등을 담습니다. Target은 그룹별 시간 발전량이며 shared 학습에서는 용량으로 나눈 capacity factor입니다.

| 후보 | 역할 |
|---|---|
| lgb_l1 | 절대 오차 중심 기본 모델 |
| lgb_q06 / lgb_q07 | quantile 위치가 다른 후보 |
| shared_l1 / shared_q07 | 그룹 간 학습을 공유하고 용량으로 복원 |
| 후처리 | 고정 조합·보정으로 최종 발전량 구성 |

분위수 하나가 전체 예측분포나 잘 보정된 확률이 되는 것은 아닙니다.

## Metric의 경계

실제 발전량이 설비용량의 10% 이상인 행을 평가합니다. NMAE는 용량으로 정규화한 절대 오차이고, FICR은 실제 발전량으로 가중한 정산금 비율입니다.

```text
정규화 오차 ≤ 6%: 단가 4
6% < 정규화 오차 ≤ 8%: 단가 3
정규화 오차 > 8%: 단가 0

종합 = 0.5 × [그룹 평균(1−NMAE) + 그룹 평균(FICR)]
```

FICR을 해당 구간에 든 행 수의 비율로 계산하면 다른 지표가 됩니다. [metric.py](../../src/metric.py)에서 threshold와 가중 방식을 확인할 수 있습니다.

## Capacity factor의 의미

용량이 다른 그룹의 kWh를 그대로 pooled target으로 쓰면 용량 차이가 학습에 섞입니다. Shared 모델은 target을 용량으로 나눈 뒤 예측을 원래 단위로 복원합니다. [최종 학습 테스트](../../tests/test_final_training.py)는 이 단위 경계를 다룹니다.

## 예보 provenance

원본의 날짜가 맞는 것만으로 예측시점 가용성을 증명할 수는 없습니다. Issue time, publication time과 예보 run을 구분해야 합니다. 사후 개정 자료를 당시 사용 가능 예보로 바꿔 쓰지 않습니다.

[특징 생성](../../scripts/build_features.py) · [개발 학습](../../scripts/train_dev.py) · [설정](../../configs)
