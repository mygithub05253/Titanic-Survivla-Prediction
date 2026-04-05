# Titanic 생존 예측 프로젝트 가이드라인

> BDAI 타이타닉 생존 예측 튜토리얼 공모전 참가 가이드라인
> 작성일: 2026-04-05 | 버전: 1.0

---

## 1. 프로젝트 개요

### 1.1 목표

타이타닉 승객 데이터를 기반으로 생존 여부(0: 사망, 1: 생존)를 예측하는 이진 분류 모델을 개발합니다.
베이스라인 F1 Score **0.7151**을 개선하여 **0.76 이상**의 F1 Score를 달성하는 것을 1차 목표로 합니다.

### 1.2 현재 리더보드 현황

| 순위 | F1 Score |
|------|----------|
| 1위  | 0.7594   |
| 베이스라인 | 0.7151 |

### 1.3 규칙 요약

- target 변수 `survived`를 입력 변수로 사용 금지
- 테스트 데이터의 정답(label) 직접 활용 금지
- 제출 파일: `submission.csv` (컬럼: PassengerId, Survived)
- Seed 값(42) 변경 금지

---

## 2. F1 Score 상세 설명

### 2.1 F1 Score란?

F1 Score는 **Precision(정밀도)**과 **Recall(재현율)**의 **조화평균**입니다.
단순 평균이 아닌 조화평균을 사용하는 이유는, 두 지표 중 하나가 극단적으로 낮을 때 이를 강하게 반영하기 위해서입니다.

### 2.2 핵심 개념 정리

이진 분류에서 예측 결과는 4가지로 나뉩니다 (생존=1 기준):

| | 실제 생존(1) | 실제 사망(0) |
|---|---|---|
| **예측 생존(1)** | TP (True Positive) | FP (False Positive) |
| **예측 사망(0)** | FN (False Negative) | TN (True Negative) |

**Precision (정밀도)**: 생존이라고 예측한 것 중 실제로 생존한 비율

```
Precision = TP / (TP + FP)
```

"내가 생존이라고 했을 때, 얼마나 맞았는가?"를 의미합니다.
Precision이 높다면, 모델이 "생존"이라고 판단할 때 꽤 신뢰할 수 있다는 뜻입니다.

**Recall (재현율)**: 실제 생존자 중 모델이 생존으로 예측한 비율

```
Recall = TP / (TP + FN)
```

"실제 생존자를 얼마나 잘 찾아냈는가?"를 의미합니다.
Recall이 높다면, 놓치는 생존자가 적다는 뜻입니다.

**F1 Score**: Precision과 Recall의 조화평균

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 2.3 왜 Accuracy가 아닌 F1 Score인가?

타이타닉 데이터는 **클래스 불균형**이 존재합니다 (사망자가 생존자보다 많음).
이런 경우 모든 승객을 "사망"으로 예측해도 Accuracy는 61%가 나옵니다.
하지만 이 모델은 생존자를 하나도 찾지 못하므로 실질적으로 무의미합니다.

F1 Score는 이런 상황에서 모델이 소수 클래스(생존)를 얼마나 잘 예측하는지를 정확히 평가합니다.

### 2.4 현재 베이스라인 성능 해석

```
              precision    recall  f1-score   support
           0       0.80      0.93      0.86       165
           1       0.84      0.62      0.72       103
```

- 생존(1) 클래스의 Precision 0.84: 생존이라고 예측한 것의 84%가 맞음
- 생존(1) 클래스의 Recall 0.62: 실제 생존자의 62%만 찾아냄
- **Recall이 낮은 것이 핵심 문제** - 생존자 38%를 놓치고 있음

따라서 성능 개선의 핵심은 **Recall을 높이면서 Precision을 유지하는 것**입니다.

### 2.5 F1 Score 개선 전략

| 전략 | 효과 | 방법 |
|------|------|------|
| Recall 향상 | 놓치는 생존자 줄이기 | 피처 엔지니어링, 모델 임계값 조정 |
| Precision 유지 | 오탐 방지 | 강력한 피처, 앙상블 |
| 둘 다 향상 | F1 극대화 | 더 나은 데이터 표현 + 모델 최적화 |

---

## 3. 데이터 분석 가이드

### 3.1 사용 가능한 전체 컬럼

| 컬럼명 | 설명 | 타입 | 결측치 |
|--------|------|------|--------|
| pclass | 객실 등급 (1=1등석, 2=2등석, 3=3등석) | 수치형/순서형 | 없음 |
| sex | 성별 | 범주형 | 없음 |
| age | 나이 | 수치형 | 있음 (~20%) |
| sibsp | 동승한 형제/배우자 수 | 수치형 | 없음 |
| parch | 동승한 부모/자녀 수 | 수치형 | 없음 |
| fare | 운임 | 수치형 | 없음 |
| embarked | 탑승항 (C=Cherbourg, Q=Queenstown, S=Southampton) | 범주형 | 있음 (2건) |
| class | 객실 등급 (문자형) | 범주형 | 없음 |
| who | man/woman/child | 범주형 | 없음 |
| adult_male | 성인 남성 여부 | 불리언 | 없음 |
| deck | 갑판 (A~G) | 범주형 | 있음 (~77%) |
| embark_town | 탑승 도시명 | 범주형 | 있음 (2건) |
| alive | 생존 여부 (yes/no) | 범주형 | 없음 |
| alone | 혼자 탑승 여부 | 불리언 | 없음 |

**주의**: `alive` 컬럼은 `survived`와 동일한 정보이므로 **절대 사용 금지**입니다.
`class`는 `pclass`와 동일 정보이므로 중복 사용에 주의합니다.

### 3.2 핵심 EDA (탐색적 데이터 분석) 포인트

1. **성별별 생존율**: 여성의 생존율이 압도적으로 높음 (74% vs 19%)
2. **객실 등급별 생존율**: 1등석 > 2등석 > 3등석 순으로 생존율 높음
3. **나이별 생존율**: 어린이(~10세)의 생존율이 높음, 나이 결측치 처리가 중요
4. **가족 크기**: 혼자 탑승한 경우 생존율 낮음, 적정 가족 크기에서 생존율 높음
5. **운임**: 높은 운임일수록 생존율 높음 (1등석과 연관)
6. **탑승항**: Cherbourg(C) 탑승객의 생존율이 상대적으로 높음

---

## 4. 단계별 성능 개선 로드맵

### Phase 1: 피처 엔지니어링 (목표: F1 0.73~0.74)

베이스라인에서 사용하지 않는 유용한 피처를 추가하고, 새로운 파생 변수를 생성합니다.

**추가할 피처:**
- `who`: man/woman/child 구분 (성별+나이 결합 정보)
- `adult_male`: 성인 남성 여부 (생존율과 강한 상관)
- `deck`: 갑판 정보 (결측치가 많지만 정보가 유용)

**새로 만들 파생 변수:**
- `family_size`: sibsp + parch + 1 (총 가족 크기)
- `is_child`: age < 10 여부
- `fare_per_person`: fare / family_size (1인당 운임)
- `age_group`: 나이 구간화 (child/teen/adult/senior)

**결측치 처리 개선:**
- age: 단순 중앙값 대신 pclass + sex 그룹별 중앙값 적용
- deck: 'Unknown' 카테고리로 처리 또는 pclass 기반 추정

### Phase 2: 모델 다양화 및 하이퍼파라미터 튜닝 (목표: F1 0.74~0.76)

**시도할 모델들:**
- GradientBoostingClassifier: 부스팅 기반으로 RandomForest보다 높은 성능 기대
- XGBoost / LightGBM: 더 고도화된 부스팅 모델
- LogisticRegression: 간단하지만 강력한 베이스라인

**하이퍼파라미터 튜닝:**
- GridSearchCV 또는 RandomizedSearchCV 활용
- 교차 검증(Cross-Validation) 적용으로 과적합 방지

### Phase 3: 앙상블 및 최종 최적화 (목표: F1 0.76+)

**앙상블 기법:**
- VotingClassifier: 여러 모델의 다수결 투표
- StackingClassifier: 메타 학습기를 통한 모델 결합

**추가 최적화:**
- 예측 임계값(threshold) 조정: 기본 0.5에서 조정하여 Recall/Precision 밸런스 최적화
- 특성 선택(Feature Selection): 중요도 낮은 피처 제거
- K-Fold 교차 검증으로 안정적인 성능 확인

---

## 5. 프로젝트 폴더 구조

```
Titanic-Survival-Prediction/
├── README.md                    # 프로젝트 전체 설명 (영문)
├── configs/                     # 설정 파일
│   └── config.yaml             # 모델/전처리 하이퍼파라미터
├── docs/                        # 문서
│   ├── ko/                     # 한국어 문서
│   │   └── GUIDELINE_KO.md    # 가이드라인 (한국어)
│   └── en/                     # 영어 문서
│       └── GUIDELINE_EN.md    # 가이드라인 (영어)
├── notebooks/                   # Jupyter 노트북
│   ├── 01_EDA.ipynb            # 탐색적 데이터 분석
│   ├── 02_baseline.ipynb       # 베이스라인 코드
│   ├── 03_feature_eng.ipynb    # 피처 엔지니어링
│   ├── 04_model_tuning.ipynb   # 모델 튜닝
│   └── 05_ensemble.ipynb       # 앙상블 및 최종 모델
├── src/                         # 소스 코드
│   ├── preprocessing/          # 전처리 모듈
│   │   ├── __init__.py
│   │   └── feature_engineer.py # 피처 엔지니어링 함수
│   ├── models/                 # 모델 모듈
│   │   ├── __init__.py
│   │   └── trainer.py          # 모델 학습/평가 함수
│   └── utils/                  # 유틸리티
│       ├── __init__.py
│       └── helpers.py          # 공통 유틸리티 함수
├── outputs/                     # 출력 파일
│   └── submission.csv          # 최종 제출 파일
├── reports/                     # 보고서
│   ├── ko/                     # 한국어 보고서
│   └── en/                     # 영어 보고서
├── .github/                     # GitHub 설정
├── .gitignore                   # Git 무시 파일
└── requirements.txt             # 의존성 패키지
```

---

## 6. 버전 관리 전략

### 6.1 브랜치 전략

```
main ─────────────────────────────────────────────
  │
  ├── feature/eda ──────────── (EDA 분석)
  │
  ├── feature/preprocessing ── (전처리 개선)
  │
  ├── feature/model-tuning ─── (모델 튜닝)
  │
  └── feature/ensemble ──────── (앙상블)
```

### 6.2 커밋 컨벤션

```
feat: 새 기능 추가
fix: 버그 수정
docs: 문서 변경
refactor: 코드 리팩토링
perf: 성능 개선
test: 테스트 추가/수정
```

### 6.3 버전별 성능 기록

각 단계마다 F1 Score를 기록하고, README.md에 업데이트합니다.

| 버전 | 설명 | F1 Score | 날짜 |
|------|------|----------|------|
| v0.1 | 베이스라인 | 0.7151 | 2026-04-05 |
| v0.2 | 피처 엔지니어링 | - | - |
| v0.3 | 모델 튜닝 | - | - |
| v1.0 | 최종 제출 | - | - |

---

## 7. 다음 단계 (Action Items)

1. EDA 노트북 작성 - 데이터 시각화 및 인사이트 도출
2. 피처 엔지니어링 실험 - 파생 변수 생성 및 효과 검증
3. 모델 비교 실험 - 다양한 모델 성능 비교
4. 하이퍼파라미터 튜닝 - 최적 파라미터 탐색
5. 앙상블 적용 - 최종 모델 구성
6. 제출 파일 생성 - submission.csv 생성 및 제출

---

## 8. 참고 자료

- scikit-learn 공식 문서: https://scikit-learn.org/stable/
- F1 Score 설명: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- seaborn Titanic 데이터셋 정보: https://github.com/mwaskom/seaborn-data
