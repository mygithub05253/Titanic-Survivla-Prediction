# Titanic Survival Prediction Project Guideline

> BDAI Titanic Survival Prediction Tutorial Competition Guide
> Created: 2026-04-05 | Version: 1.0

---

## 1. Project Overview

### 1.1 Objective

Develop a binary classification model to predict passenger survival (0: deceased, 1: survived) based on Titanic passenger data. The primary goal is to improve from the baseline F1 Score of **0.7151** to **0.76+**.

### 1.2 Current Leaderboard

| Rank | F1 Score |
|------|----------|
| 1st Place | 0.7594 |
| Baseline | 0.7151 |

### 1.3 Rules Summary

- Target variable `survived` must NOT be used as an input feature
- Direct use of test data labels is prohibited
- Submission file: `submission.csv` (columns: PassengerId, Survived)
- Seed value (42) must not be changed

---

## 2. Understanding F1 Score

### 2.1 What is F1 Score?

F1 Score is the **harmonic mean** of Precision and Recall. The harmonic mean (rather than arithmetic mean) is used because it penalizes extreme imbalances between the two metrics more heavily.

### 2.2 Core Concepts

In binary classification, predictions fall into 4 categories (with Survived=1 as positive):

| | Actually Survived (1) | Actually Deceased (0) |
|---|---|---|
| **Predicted Survived (1)** | TP (True Positive) | FP (False Positive) |
| **Predicted Deceased (0)** | FN (False Negative) | TN (True Negative) |

**Precision**: Among those predicted as survived, what fraction actually survived?

```
Precision = TP / (TP + FP)
```

High precision means the model is reliable when it says "survived."

**Recall**: Among actual survivors, what fraction did the model identify?

```
Recall = TP / (TP + FN)
```

High recall means fewer survivors are missed by the model.

**F1 Score**: Harmonic mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 2.3 Why F1 Score Instead of Accuracy?

The Titanic dataset has **class imbalance** (more deceased than survived). Predicting all passengers as "deceased" yields ~61% accuracy but completely fails to identify survivors. F1 Score properly evaluates how well the model handles the minority class (survived).

### 2.4 Baseline Performance Analysis

```
              precision    recall  f1-score   support
           0       0.80      0.93      0.86       165
           1       0.84      0.62      0.72       103
```

- Survived(1) Precision 0.84: 84% of survival predictions are correct
- Survived(1) Recall 0.62: Only 62% of actual survivors are identified
- **Low Recall is the key issue** - 38% of survivors are being missed

The core improvement strategy: **increase Recall while maintaining Precision**.

### 2.5 F1 Score Improvement Strategies

| Strategy | Effect | Method |
|----------|--------|--------|
| Improve Recall | Catch more survivors | Feature engineering, threshold tuning |
| Maintain Precision | Avoid false positives | Strong features, ensemble |
| Improve Both | Maximize F1 | Better data representation + model optimization |

---

## 3. Data Analysis Guide

### 3.1 Available Columns

| Column | Description | Type | Missing |
|--------|-------------|------|---------|
| pclass | Ticket class (1=1st, 2=2nd, 3=3rd) | Numeric/Ordinal | None |
| sex | Gender | Categorical | None |
| age | Age | Numeric | ~20% |
| sibsp | Siblings/Spouses aboard | Numeric | None |
| parch | Parents/Children aboard | Numeric | None |
| fare | Passenger fare | Numeric | None |
| embarked | Port of embarkation (C/Q/S) | Categorical | 2 |
| class | Ticket class (string) | Categorical | None |
| who | man/woman/child | Categorical | None |
| adult_male | Whether adult male | Boolean | None |
| deck | Deck (A-G) | Categorical | ~77% |
| embark_town | Embarkation city name | Categorical | 2 |
| alive | Survival status (yes/no) | Categorical | None |
| alone | Whether traveling alone | Boolean | None |

**Warning**: `alive` column contains the same information as `survived` - **absolutely must NOT be used**.
`class` duplicates `pclass` information - avoid redundant usage.

### 3.2 Key EDA Points

1. **Gender vs Survival**: Female survival rate is dramatically higher (74% vs 19%)
2. **Class vs Survival**: 1st > 2nd > 3rd class in survival rate
3. **Age vs Survival**: Children (~10 and under) have higher survival rates; age imputation matters
4. **Family Size**: Solo travelers have lower survival; moderate family size shows higher survival
5. **Fare**: Higher fares correlate with higher survival (linked to class)
6. **Embarkation**: Cherbourg (C) passengers have relatively higher survival

---

## 4. Step-by-Step Improvement Roadmap

### Phase 1: Feature Engineering (Target: F1 0.73~0.74)

Add useful features not in the baseline and create derived variables.

**Features to add:**
- `who`: man/woman/child classification (combined sex + age info)
- `adult_male`: adult male indicator (strong correlation with survival)
- `deck`: deck information (useful despite high missingness)

**New derived variables:**
- `family_size`: sibsp + parch + 1 (total family size)
- `is_child`: age < 10 flag
- `fare_per_person`: fare / family_size
- `age_group`: age binning (child/teen/adult/senior)

**Improved missing value handling:**
- age: group-wise median by pclass + sex instead of global median
- deck: treat as 'Unknown' category or estimate from pclass

### Phase 2: Model Diversification & Hyperparameter Tuning (Target: F1 0.74~0.76)

**Models to try:**
- GradientBoostingClassifier: boosting-based, potentially outperforms RandomForest
- XGBoost / LightGBM: advanced gradient boosting
- LogisticRegression: simple yet powerful baseline

**Hyperparameter tuning:**
- GridSearchCV or RandomizedSearchCV
- Cross-validation to prevent overfitting

### Phase 3: Ensemble & Final Optimization (Target: F1 0.76+)

**Ensemble techniques:**
- VotingClassifier: majority voting across multiple models
- StackingClassifier: meta-learner combining model predictions

**Additional optimization:**
- Prediction threshold tuning: adjust from default 0.5 for optimal Recall/Precision balance
- Feature Selection: remove low-importance features
- K-Fold cross-validation for stable performance estimation

---

## 5. Project Folder Structure

```
Titanic-Survival-Prediction/
├── README.md                    # Project overview
├── configs/                     # Configuration files
│   └── config.yaml             # Model/preprocessing hyperparameters
├── docs/                        # Documentation
│   ├── ko/                     # Korean documents
│   │   └── GUIDELINE_KO.md
│   └── en/                     # English documents
│       └── GUIDELINE_EN.md
├── notebooks/                   # Jupyter notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_baseline.ipynb       # Baseline code
│   ├── 03_feature_eng.ipynb    # Feature engineering
│   ├── 04_model_tuning.ipynb   # Model tuning
│   └── 05_ensemble.ipynb       # Ensemble & final model
├── src/                         # Source code modules
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── feature_engineer.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── outputs/                     # Output files
│   └── submission.csv
├── reports/                     # Reports
│   ├── ko/
│   └── en/
├── .github/                     # GitHub configuration
├── .gitignore
└── requirements.txt
```

---

## 6. Version Control Strategy

### 6.1 Branch Strategy

```
main ─────────────────────────────────────────────
  ├── feature/eda
  ├── feature/preprocessing
  ├── feature/model-tuning
  └── feature/ensemble
```

### 6.2 Commit Convention

```
feat: add new feature
fix: bug fix
docs: documentation changes
refactor: code refactoring
perf: performance improvement
test: add/modify tests
```

### 6.3 Performance Tracking

| Version | Description | F1 Score | Date |
|---------|-------------|----------|------|
| v0.1 | Baseline | 0.7151 | 2026-04-05 |
| v0.2 | Feature Engineering | - | - |
| v0.3 | Model Tuning | - | - |
| v1.0 | Final Submission | - | - |

---

## 7. Next Steps (Action Items)

1. Create EDA notebook - data visualization and insight extraction
2. Feature engineering experiments - derived variables and effect validation
3. Model comparison experiments - benchmark multiple models
4. Hyperparameter tuning - optimal parameter search
5. Ensemble application - final model composition
6. Generate submission file - create and submit submission.csv

---

## 8. References

- scikit-learn documentation: https://scikit-learn.org/stable/
- F1 Score reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- seaborn Titanic dataset: https://github.com/mwaskom/seaborn-data
