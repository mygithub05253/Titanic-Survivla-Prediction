# Titanic Survival Prediction - Project Context

## Project Overview / 프로젝트 개요

BDAI 타이타닉 생존 예측 튜토리얼 공모전 프로젝트입니다.
seaborn의 titanic 데이터셋을 활용하여 승객 생존 여부를 예측하는 이진 분류 모델을 개발합니다.

- **Platform**: GSTACK (공모전 플랫폼)
- **Evaluation Metric**: F1 Score (Survived=1 기준)
- **Current Leaderboard 1st**: 0.7594
- **Baseline F1 Score**: 0.7151
- **Target**: F1 0.76+ (1등 이상)
- **GitHub**: https://github.com/mygithub05253/Titanic-Survivla-Prediction.git

## Critical Rules / 절대 규칙

1. **SEED 고정**: `SEED = 42` - 변경 금지
2. **Target Leakage 금지**: `survived` 컬럼을 입력 변수로 사용 금지
3. **`alive` 컬럼 사용 금지**: `survived`와 동일한 정보 (yes/no)
4. **Test Label 직접 활용 금지**: 테스트 데이터의 정답을 직접 사용하면 안 됨
5. **제출 형식**: `submission.csv` (컬럼: PassengerId, Survived)

## Data Source / 데이터 소스

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

별도 파일 제공 없음. 코드를 통해 직접 불러옴.

## Available Columns / 사용 가능 컬럼

| Column | Description | Type | Missing | Usage Note |
|--------|-------------|------|---------|------------|
| pclass | 객실 등급 (1/2/3) | Numeric | No | USE - 핵심 피처 |
| sex | 성별 | Categorical | No | USE - 핵심 피처 |
| age | 나이 | Numeric | ~20% | USE - 결측치 처리 중요 |
| sibsp | 형제/배우자 수 | Numeric | No | USE |
| parch | 부모/자녀 수 | Numeric | No | USE |
| fare | 운임 | Numeric | No | USE |
| embarked | 탑승항 (C/Q/S) | Categorical | 2건 | USE |
| class | 객실 등급 (문자형) | Categorical | No | CAUTION - pclass 중복 |
| who | man/woman/child | Categorical | No | USE - 유용 |
| adult_male | 성인 남성 여부 | Boolean | No | USE |
| deck | 갑판 (A~G) | Categorical | ~77% | USE - 결측 많지만 유용 |
| embark_town | 탑승 도시명 | Categorical | 2건 | CAUTION - embarked 중복 |
| alive | 생존 여부 (yes/no) | Categorical | No | **BANNED** - target leak |
| alone | 혼자 탑승 여부 | Boolean | No | USE |

## Project Structure / 폴더 구조

```
Titanic-Survival-Prediction/
├── CLAUDE.md              # 이 파일 - 프로젝트 컨텍스트
├── README.md              # GitHub README
├── .gitignore
├── requirements.txt
├── configs/
│   └── config.yaml        # 모델/전처리 하이퍼파라미터
├── docs/
│   ├── ko/                # 한국어 문서 (MD + DOCX)
│   └── en/                # 영어 문서 (MD + DOCX)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_eng.ipynb
│   ├── 04_model_tuning.ipynb
│   └── 05_ensemble.ipynb
├── src/
│   ├── preprocessing/
│   │   └── feature_engineer.py
│   ├── models/
│   │   └── trainer.py
│   └── utils/
│       └── helpers.py
├── outputs/
│   └── submission.csv     # 최종 제출 파일
└── reports/
    ├── ko/
    └── en/
```

## Development Phases / 개발 단계

### Phase 1: Feature Engineering (현재 → F1 0.73~0.74)
- 파생 변수 생성: family_size, is_child, fare_per_person, age_group
- who, adult_male, deck 피처 추가
- pclass + sex 그룹별 age 결측치 처리

### Phase 2: Model Tuning (F1 0.74~0.76)
- GradientBoosting, XGBoost, LightGBM 실험
- GridSearchCV / RandomizedSearchCV
- 교차 검증 (StratifiedKFold)

### Phase 3: Ensemble & Optimization (F1 0.76+)
- VotingClassifier, StackingClassifier
- 예측 임계값(threshold) 조정
- Feature Selection

## Version History / 버전 기록

| Version | Description | F1 Score | Date |
|---------|-------------|----------|------|
| v0.1 | Baseline (RandomForest) | 0.7151 | 2026-04-05 |

## Git Convention / 커밋 컨벤션

```
feat: 새 기능 추가
fix: 버그 수정
docs: 문서 변경
refactor: 코드 리팩토링
perf: 성능 개선
test: 테스트 추가/수정
data: 데이터 처리 관련
exp: 실험 관련
```

## Workflow Convention / 작업 흐름 규칙

모든 단계의 작업은 반드시 아래 3단계 흐름을 따릅니다:

### Step 1: Plan (계획 수립)
- 다음 작업을 시작하기 전에 **반드시 사용자와 대화**로 가이드라인을 먼저 합의
- Claude Code의 Plan 모드를 활용하여 목표, 접근 방식, 예상 결과를 논의
- 사용자가 계획을 승인한 후에만 실행 단계로 진행

### Step 2: Execute (실행)
- 합의된 계획에 따라 작업 수행
- 활용 가능한 도구:
  - **GSTACK**: 공모전 플랫폼 (제출, 리더보드 확인)
  - **data 스킬**: 데이터 분석, 시각화, 통계 분석, 대시보드
  - **engineering 스킬**: GitHub 관리, 코드 리뷰, 문서화
  - **docx 스킬**: Word 문서 생성 (계획서, 명세서, 보고서)
  - **xlsx 스킬**: Excel/CSV 파일 처리
  - **titanic-ml 스킬**: ML 작업 전용 (코드 템플릿, 안전 점검)
- 작업 중 TodoList로 진행 상황 추적

### Step 3: Verify & Commit (검증 및 버전 관리)
- 결과 확인 (F1 Score, 문서 품질 등)
- CLAUDE.md의 Version History 업데이트
- Git 커밋 및 push (사용자 승인 후)
- 다음 단계 Plan 모드로 전환

### 중요 원칙
- **절대 사용자 승인 없이 다음 단계로 넘어가지 않음**
- **계획 없이 코드를 작성하지 않음**
- **매 실행 후 반드시 결과를 사용자에게 보고**

## Key Insights / 핵심 인사이트

- **Recall이 낮음** (0.62): 생존자 38%를 놓치고 있음 → Recall 개선이 1순위
- **성별**: 여성 생존율 74% vs 남성 19%
- **객실 등급**: 1등석 > 2등석 > 3등석
- **나이**: 10세 이하 어린이 생존율 높음
- **가족 크기**: 2~4명이 최적 생존율

## Document Standards / 문서 기준

- 모든 문서는 한국어(ko/)와 영어(en/) 두 버전 생성
- 계획서, 명세서, 설계서, 보고서 모두 DOCX + MD 형식
- DOCX 생성 시 반드시 docx 스킬 참조
- CSV/Excel 생성 시 반드시 xlsx 스킬 참조

## Submission Format / 제출 형식

```csv
PassengerId,Survived
1,0
2,1
3,0
...
```

- PassengerId: 1부터 시작하는 순차 번호 (seaborn 데이터에 없으므로 index 기반 생성)
- Survived: 0 또는 1
