# 예보 시점과 target 단위

공개시점 이전에 이용 가능한 기상 예보에서 특징을 만들고, 그룹별 모델 또는 shared 모델로 발전량을 예측합니다. Shared 모델은 capacity factor로 학습한 값을 설비용량으로 복원한 뒤 고정 후처리를 적용합니다.

예보 날짜뿐 아니라 run·issue time·publication time이 맞아야 합니다. Metric의 용량·threshold 단위 테스트는 수식의 검사이고, 실제 예보가 당시 존재했는지는 원천 기록으로 따로 확인합니다.
