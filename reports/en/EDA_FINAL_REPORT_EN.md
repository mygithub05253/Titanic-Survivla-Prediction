# Titanic Survival Prediction - EDA Final Report

## Project Overview

**Project**: BDAI Titanic Survival Prediction Competition
**Platform**: GSTACK
**Data**: seaborn titanic dataset (891 rows x 15 columns)
**Metric**: F1 Score (Survived=1)
**Baseline**: F1 0.7151 (RandomForest)
**Leaderboard 1st**: 0.7594
**Target**: F1 0.76+ (beat leaderboard)
**Date**: 2026-04-06

---

## 1. Data Summary

- 891 passenger records, 15 columns
- Survival rate: 38.4% (342 survived / 549 died)
- Imbalance ratio: 1.61:1 (moderate)
- Missing: deck(77.2%), age(19.9%), embarked(0.2%)
- Banned column: alive (identical to survived, target leakage)

## 2. Key Findings

### 2.1 Strongest Predictors
- **Sex**: Female 74.2% vs Male 18.9% (highest Chi-square)
- **Pclass**: 1st 63.0% > 2nd 47.3% > 3rd 24.2%
- **Who**: woman 75.6% > child 59.0% > man 16.4%

### 2.2 Pclass + Sex Interaction
- 1st class female: ~97% survival (highest)
- 3rd class male: ~14% survival (lowest)
- This interaction is critical for modeling

### 2.3 Family Size Effect
- Families of 2-4 have optimal survival
- Solo travelers (1) and large families (5+) have lower survival
- Derived feature: family_size = sibsp + parch + 1

### 2.4 Fare Characteristics
- Extremely right-skewed (skew=4.79)
- Log transform reduces skew to ~0.4
- 15 zero-fare tickets (mostly 3rd class males, very low survival)
- High fare = 1st class = high survival (do NOT remove outliers)

## 3. Missing Data Strategy

| Variable | Missing% | Pattern | Strategy |
|----------|----------|---------|----------|
| age | 19.9% | MAR (depends on pclass) | pclass+sex group median + age_missing dummy |
| deck | 77.2% | MAR (concentrated in 3rd) | deck_known binary |
| embarked | 0.2% | MCAR | Mode(S) |

## 4. Outlier Handling Strategy

| Variable | IQR Outliers | Strategy | Rationale |
|----------|-------------|----------|-----------|
| fare | 116 (13.0%) | Log transform | Preserve high-survival passengers |
| sibsp | 18 (2.0%) | Keep | Family size signal |
| parch | 6 (0.7%) | Keep | Family size signal |
| age | 1 (0.1%) | Keep | max=80 is reasonable |

## 5. Feature Recommendations

### Essential
1. sex_encoded - Most powerful predictor
2. pclass - Strong survival signal
3. log_fare - Skew-reduced fare
4. age - Imputed with pclass+sex group median

### High Priority
5. family_size - Optimal survival at 2-4
6. who_encoded - man/woman/child
7. adult_male - Additional value beyond sex

### Medium Priority
8. deck_known - Deck info availability
9. age_missing - Missing itself is informative
10. embarked_encoded - C > Q > S
11. is_child - age <= 12

### DROP
- alive: BANNED (target leakage)
- class: duplicate of pclass
- embark_town: duplicate of embarked
- deck: use deck_known instead

## 6. Modeling Strategy

### Core Problem: Low Recall
- Current Recall: 0.61 (~39% of survivors missed)
- FN profile: Mostly 3rd class males, solo travelers, low fare
- With Precision 0.80 + Recall 0.72+ = F1 0.76

### Phase 2: Model Tuning (F1 0.74~0.76)
- Experiment: GradientBoosting, XGBoost, LightGBM
- Apply class_weight='balanced'
- GridSearchCV / RandomizedSearchCV
- StratifiedKFold(5) cross-validation

### Phase 3: Ensemble & Optimization (F1 0.76+)
- VotingClassifier, StackingClassifier
- Threshold optimization
- Feature Selection

## 7. EDA Deliverables

| Step | Notebook | PNGs | Reports |
|------|----------|------|---------|
| Step 1 | 01_EDA_step1_basic_stats.ipynb | 1 | KO/EN MD+DOCX |
| Step 2 | 02_EDA_step2_class_imbalance.ipynb | 4 | KO/EN MD+DOCX |
| Step 3 | 03_EDA_step3_univariate.ipynb | 6 | KO/EN MD+DOCX |
| Step 4 | 04_EDA_step4_outlier_detection.ipynb | 3 | KO/EN MD+DOCX |
| Step 5 | 05_EDA_step5_multivariate.ipynb | 6 | KO/EN MD+DOCX |
| Step 6 | 06_EDA_step6_missing_data.ipynb | 3 | KO/EN MD+DOCX |
| Step 7 | 07_EDA_step7_feature_engineering.ipynb | 4 | KO/EN MD+DOCX |
| Step 8 | - | - | Dashboard + Final Report |

## 8. Conclusion

Through systematic EDA of the Titanic dataset:
1. **Sex and cabin class are the strongest survival determinants**
2. **Derived features (family_size, log_fare, deck_known) add value**
3. **Recall improvement is the key to F1 gains** (FN reduction strategy needed)
4. **pclass+sex interaction pattern is essential for modeling**

Recommended next step: Proceed to Phase 2 (Model Tuning).
