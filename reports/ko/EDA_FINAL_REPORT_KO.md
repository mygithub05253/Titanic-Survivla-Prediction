# 타이타닉 생존 예측 - EDA 종합 보고서

## 프로젝트 개요

**프로젝트**: BDAI 타이타닉 생존 예측 공모전
**플랫폼**: GSTACK
**데이터**: seaborn titanic 데이터셋 (891행 x 15열)
**평가 지표**: F1 Score (Survived=1 기준)
**Baseline**: F1 0.7151 (RandomForest)
**리더보드 1위**: 0.7594
**목표**: F1 0.76+ (1위 이상)
**작성일**: 2026-04-06

---

## 1. 데이터 요약

- 총 891명의 승객 데이터, 15개 컬럼
- 생존율: 38.4% (342명 생존 / 549명 사망)
- 불균형 비율: 1.61:1 (중간 수준)
- 결측: deck(77.2%), age(19.9%), embarked(0.2%)
- 금지 컬럼: alive (survived와 100% 동일, target leakage)

## 2. 핵심 발견사항

### 2.1 가장 강력한 예측 변수
- **성별(sex)**: 여성 74.2% vs 남성 18.9% (Chi-square 최대)
- **객실 등급(pclass)**: 1등급 63.0% > 2등급 47.3% > 3등급 24.2%
- **역할(who)**: woman 75.6% > child 59.0% > man 16.4%

### 2.2 pclass + sex 상호작용
- 1등급 여성: ~97% 생존 (가장 높음)
- 3등급 남성: ~14% 생존 (가장 낮음)
- 이 교차 효과가 모델링에 매우 중요

### 2.3 family_size 효과
- 2~4명 가족이 최적 생존율
- 혼자(1명) 또는 대가족(5명+)은 낮은 생존율
- family_size = sibsp + parch + 1 파생 변수 필수

### 2.4 fare 특성
- 극심한 오른쪽 치우침 (skew=4.79)
- log 변환으로 왜도 ~0.4로 감소
- 0원 티켓 15건 (대부분 3등급 남성, 생존율 매우 낮음)
- 고운임 = 1등급 = 높은 생존율 (이상치 제거 금지)

## 3. 결측치 처리 전략

| 변수 | 결측% | 패턴 | 최종 전략 |
|------|-------|------|-----------|
| age | 19.9% | MAR (pclass 의존) | pclass+sex 그룹 중앙값 + age_missing dummy |
| deck | 77.2% | MAR (3등급 집중) | deck_known binary |
| embarked | 0.2% | MCAR | Mode(S) |

## 4. 이상치 처리 전략

| 변수 | IQR 이상치 | 전략 | 근거 |
|------|-----------|------|------|
| fare | 116건 (13.0%) | Log 변환 | 고생존율 승객 보존 |
| sibsp | 18건 (2.0%) | 유지 | 가족 크기 신호 |
| parch | 6건 (0.7%) | 유지 | 가족 크기 신호 |
| age | 1건 (0.1%) | 유지 | max=80 합리적 |

## 5. 피처 추천

### Essential (필수)
1. sex_encoded - 가장 강력한 예측 변수
2. pclass - 강한 생존 신호
3. log_fare - 왜도 감소된 운임
4. age - pclass+sex 그룹 중앙값으로 대체

### High (높음)
5. family_size - 2~4명 최적 생존율
6. who_encoded - man/woman/child
7. adult_male - sex와 상관 있으나 추가 가치

### Medium (중간)
8. deck_known - 갑판 정보 유무
9. age_missing - 결측 자체가 정보
10. embarked_encoded - C > Q > S
11. is_child - age <= 12

### DROP (제거)
- alive: BANNED (target leakage)
- class: pclass 중복
- embark_town: embarked 중복
- deck: deck_known 사용

## 6. 모델링 전략 제안

### 핵심 문제: Recall 부족
- 현재 Recall: 0.61 (생존자의 ~39% 미탐지)
- FN(놓친 생존자) 프로파일: 주로 3등급 남성, 혼자 탑승, 낮은 운임
- Precision 0.80 유지 + Recall 0.72+ 달성 시 F1 0.76 가능

### Phase 2: Model Tuning (F1 0.74~0.76)
- GradientBoosting, XGBoost, LightGBM 실험
- class_weight='balanced' 적용
- GridSearchCV / RandomizedSearchCV
- StratifiedKFold(5) 교차 검증

### Phase 3: Ensemble & Optimization (F1 0.76+)
- VotingClassifier, StackingClassifier
- 예측 임계값(threshold) 최적화
- Feature Selection

## 7. EDA 단계별 산출물 목록

| 단계 | 노트북 | PNG | 보고서 |
|------|--------|-----|--------|
| Step 1 | 01_EDA_step1_basic_stats.ipynb | 1 | KO/EN MD+DOCX |
| Step 2 | 02_EDA_step2_class_imbalance.ipynb | 4 | KO/EN MD+DOCX |
| Step 3 | 03_EDA_step3_univariate.ipynb | 6 | KO/EN MD+DOCX |
| Step 4 | 04_EDA_step4_outlier_detection.ipynb | 3 | KO/EN MD+DOCX |
| Step 5 | 05_EDA_step5_multivariate.ipynb | 6 | KO/EN MD+DOCX |
| Step 6 | 06_EDA_step6_missing_data.ipynb | 3 | KO/EN MD+DOCX |
| Step 7 | 07_EDA_step7_feature_engineering.ipynb | 4 | KO/EN MD+DOCX |
| Step 8 | - | - | 대시보드 + 종합 보고서 |

## 8. 결론

타이타닉 데이터셋의 체계적 EDA를 통해 다음을 확인했습니다:
1. **성별과 객실 등급이 생존의 가장 강력한 결정 요인**
2. **family_size, log_fare, deck_known 등 파생 변수가 유용**
3. **Recall 개선이 F1 향상의 핵심** (FN 해소 전략 필요)
4. **pclass+sex 상호작용 패턴이 모델링에 필수**

다음 단계로 Phase 2 (Model Tuning) 진행을 권장합니다.
