# EDA Step 7: 피처 엔지니어링 탐색 보고서

## 1. 개요
**단계**: EDA Step 7 - 피처 엔지니어링 탐색
**작성일**: 2026-04-06
**목표**: 파생 변수 생성, 피처 중요도 평가, 모델링 전략 수립

## 2. 생성된 파생 변수

| 변수 | 공식 | 생존 신호 |
|------|------|-----------|
| family_size | sibsp + parch + 1 | 2~4명 최적 생존율 |
| is_alone | family_size == 1 | 혼자 탑승 시 낮은 생존율 |
| is_child | age <= 12 | 어린이 높은 생존율 |
| log_fare | log(fare+1) | 왜도 감소 |
| fare_per_person | fare / family_size | 1인당 운임 |
| deck_known | deck 비결측 여부 | 유의한 생존 신호 |
| age_missing | age 결측 여부 | 결측 자체가 정보 |

## 3. 피처 중요도 Top 10

1. sex_encoded - 가장 강력한 예측 변수
2. fare / log_fare - 운임 (log 변환)
3. age - 나이 (대체 후)
4. pclass - 객실 등급
5. family_size - 가족 크기
6. who_encoded - man/woman/child
7. adult_male - 성인 남성 여부
8. fare_per_person - 1인당 운임
9. deck_known - 갑판 정보 유무
10. embarked_encoded - 탑승항

## 4. FN 분석 (놓친 생존자 프로파일)

- 주로 **3등급 남성** (약 65% 이상)
- **혼자 탑승** 비율 높음
- **낮은 운임** (TP보다 현저히 낮음)
- 나이는 TP와 유사한 분포

## 5. Recall 개선 전략 비교

| 전략 | F1 | Precision | Recall |
|------|-----|-----------|--------|
| Default | ~0.72 | ~0.75 | ~0.70 |
| Balanced Weight | 변동 | 감소 | 증가 |
| Threshold=0.4 | 변동 | 감소 | 증가 |
| Threshold=0.35 | 변동 | 감소 | 크게 증가 |

## 6. 최종 피처 추천표

### Essential (필수)
- sex_encoded, pclass, fare/log_fare, age

### High (높음)
- family_size, who_encoded, adult_male

### Medium (중간)
- deck_known, age_missing, embarked_encoded, is_child

### Low (낮음) - 이미 다른 피처에 포함
- fare_per_person, alone, sibsp, parch

### DROP (제거)
- alive (BANNED), class (pclass 중복), embark_town (embarked 중복), deck (deck_known 사용)

## 7. 모델링 전략 제안

1. **Phase 2**: GradientBoosting, XGBoost, LightGBM 실험
2. **Recall 개선**: class_weight 조정 + threshold 최적화
3. **Phase 3**: VotingClassifier, StackingClassifier
4. **목표**: F1 0.76+ 달성
