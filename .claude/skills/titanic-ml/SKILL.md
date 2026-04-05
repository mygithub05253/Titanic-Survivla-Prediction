---
name: titanic-ml
description: "Titanic 생존 예측 ML 프로젝트 작업 가이드. 데이터 분석, 피처 엔지니어링, 모델 학습, 성능 평가, 제출 파일 생성 시 사용합니다. Trigger: titanic, 생존 예측, EDA, feature engineering, model training, submission, F1 score, 공모전"
---

# Titanic Survival Prediction ML Workflow Guide

## 이 스킬의 목적

이 스킬은 BDAI 타이타닉 생존 예측 공모전 프로젝트에서 일관된 작업을 보장합니다.
새 세션을 시작할 때 반드시 CLAUDE.md를 먼저 읽고 현재 버전과 진행 상황을 파악하세요.

## Session Start Checklist / 세션 시작 체크리스트

1. `CLAUDE.md` 읽기 → 현재 프로젝트 상태 파악
2. `configs/config.yaml` 읽기 → 현재 설정 확인
3. `outputs/` 디렉토리의 최신 submission.csv 확인
4. Version History에서 현재 버전 및 F1 Score 확인

## Mandatory Code Template / 필수 코드 템플릿

모든 노트북과 스크립트는 반드시 아래 코드로 시작:

```python
import random
import numpy as np
import pandas as pd
import seaborn as sns

# SEED 고정 (절대 변경 금지)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 데이터 로드
df = sns.load_dataset("titanic")

# Target 분리
X = df.drop(columns=["survived"])
y = df["survived"]
```

## Feature Engineering Rules / 피처 엔지니어링 규칙

### 사용 가능 피처

```python
# 핵심 수치형
numeric_core = ["age", "sibsp", "parch", "fare", "pclass"]

# 핵심 범주형
categorical_core = ["sex", "embarked", "alone", "who", "adult_male"]

# 선택적 (결측치 처리 필요)
optional = ["deck"]  # ~77% 결측

# 파생 변수 (생성 필요)
derived = ["family_size", "is_child", "fare_per_person", "age_group"]
```

### 금지 피처

```python
BANNED_FEATURES = ["alive"]  # target leakage - survived와 동일 정보
DUPLICATE_FEATURES = ["class", "embark_town"]  # pclass, embarked와 중복
```

### 파생 변수 생성 표준 코드

```python
def create_features(df):
    """표준 파생 변수 생성 함수"""
    df = df.copy()

    # 가족 크기
    df["family_size"] = df["sibsp"] + df["parch"] + 1

    # 혼자 여부 (family_size 기반)
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # 어린이 여부
    df["is_child"] = (df["age"] < 10).astype(int)

    # 1인당 운임
    df["fare_per_person"] = df["fare"] / df["family_size"]

    # 나이 구간화
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                              labels=["child", "teen", "adult", "middle", "senior"])

    return df
```

### 결측치 처리 표준

```python
def impute_age(df):
    """pclass + sex 그룹별 중앙값으로 age 결측치 처리"""
    df = df.copy()
    age_median = df.groupby(["pclass", "sex"])["age"].transform("median")
    df["age"] = df["age"].fillna(age_median)
    # 그룹별 중앙값도 NaN인 경우 전체 중앙값 사용
    df["age"] = df["age"].fillna(df["age"].median())
    return df
```

## Model Training Standards / 모델 학습 표준

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=SEED  # 반드시 SEED 사용
)
```

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

### Performance Evaluation

```python
from sklearn.metrics import f1_score, classification_report

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)  # survived=1 기준 F1

print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred))
```

## Submission File Generation / 제출 파일 생성 표준

```python
def create_submission(X_test, y_pred, filename="submission.csv"):
    """표준 제출 파일 생성"""
    submission = pd.DataFrame({
        "PassengerId": range(1, len(y_pred) + 1),
        "Survived": y_pred
    })
    submission.to_csv(f"outputs/{filename}", index=False)
    print(f"Submission saved: {len(submission)} rows")
    return submission
```

## Performance Tracking / 성능 추적 규칙

새로운 실험 결과가 나올 때마다:

1. CLAUDE.md의 Version History 테이블에 결과 추가
2. configs/config.yaml에 최적 하이퍼파라미터 업데이트
3. F1 Score가 이전보다 높으면 outputs/submission.csv 갱신
4. GitHub에 커밋 (feat: 또는 perf: 접두사 사용)

## Document Generation Rules / 문서 생성 규칙

- **모든 문서**: 한국어(docs/ko/) + 영어(docs/en/) 두 버전 필수
- **형식**: MD + DOCX 모두 생성
- **계획서**: docs/{ko,en}/plan_*.{md,docx}
- **명세서**: docs/{ko,en}/spec_*.{md,docx}
- **설계서**: docs/{ko,en}/design_*.{md,docx}
- **보고서**: reports/{ko,en}/report_*.{md,docx}
- **DOCX 생성 시**: 반드시 docx 스킬 참조하여 전문적 포맷 적용
- **CSV/Excel 생성 시**: 반드시 xlsx 스킬 참조

## GitHub Workflow / GitHub 작업 흐름

```bash
# 브랜치 생성
git checkout -b feature/<task-name>

# 작업 후 커밋
git add <files>
git commit -m "feat: <설명>"

# Push
git push origin feature/<task-name>

# main 병합 (PR 생성 권장)
```

### 커밋 메시지 규칙

```
feat: 피처 엔지니어링 - family_size 추가 (F1: 0.7151 → 0.7350)
perf: XGBoost 하이퍼파라미터 튜닝 (F1: 0.7350 → 0.7550)
docs: 가이드라인 문서 한국어/영어 추가
data: age 결측치 처리 개선
exp: LightGBM vs XGBoost 비교 실험
```

**커밋 메시지에 F1 Score 변화를 반드시 포함**하여 성능 추적이 가능하도록 합니다.

## Safety Checks / 안전 점검

코드 실행 전 반드시 확인:

```python
# 1. alive 컬럼이 피처에 포함되지 않았는지 확인
assert "alive" not in X.columns, "alive 컬럼 사용 금지!"

# 2. survived가 입력에 없는지 확인
assert "survived" not in X.columns, "survived 컬럼 사용 금지!"

# 3. SEED가 42인지 확인
assert SEED == 42, "SEED는 42로 고정!"
```
