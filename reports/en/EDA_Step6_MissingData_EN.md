# EDA Step 6: Missing Data Deep Analysis Report

**BDAI Titanic Survival Prediction Competition | 2026-04-06**

---

## 1. Overview

- **Project**: BDAI Titanic Survival Prediction Competition
- **Phase**: EDA Step 6 - Missing Data Deep Analysis
- **Date**: 2026-04-06
- **Objective**: Identify missing data mechanisms (MCAR/MAR/MNAR) and experimentally validate optimal imputation strategies

---

## 2. Missing Data Summary

| Variable | Missing Count | Missing Rate | Mechanism | Evidence |
|----------|--------------|-------------|-----------|----------|
| age | ~177 | ~19.9% | MAR | Depends on pclass (3rd class has highest rate) |
| deck | ~688 | ~77.2% | MAR | 3rd class ~100% missing, 1st class much lower |
| embarked | 2 | ~0.2% | MCAR | Only 2 rows, no discernible pattern |

---

## 3. Age Missing Data Analysis

### 3.1 Missing Mechanism: MAR (Missing At Random)

Age missingness depends on pclass and sex, exhibiting a clear MAR pattern.

| pclass | Total | Missing | Missing Rate |
|--------|-------|---------|-------------|
| 1st | 216 | ~30 | ~14% |
| 2nd | 184 | ~11 | ~6% |
| 3rd | 491 | ~136 | ~28% |

- 3rd class missing rate (~28%) is ~2x that of 1st class (~14%) and ~5x that of 2nd class (~6%)
- Chi-square test confirms significant association between pclass and age missingness (p < 0.001)

### 3.2 Age Imputation Strategy Comparison (5-Fold CV, F1 Score)

| Strategy | Description | F1 Score | Notes |
|----------|------------|----------|-------|
| Global median | Use overall median(age) | ~0.710 | Simple but ignores group differences |
| Group median | pclass+sex group median | ~0.718 | Reflects MAR structure |
| Group median + age_missing | Group median + missing indicator dummy | ~0.723 | **Optimal strategy** |
| KNN Imputer | K=5 nearest neighbor imputation | ~0.715 | Marginal gain vs computational cost |

### 3.3 Optimal Strategy: pclass+sex Group Median + age_missing Dummy

- **Method**: Impute missing ages with pclass+sex group median, then add `age_missing` (0/1) indicator
- **Rationale**: Captures MAR structure while preserving the signal that "age is unknown" itself carries predictive value
- **F1 improvement**: +0.013 over global median

---

## 4. Deck Missing Data Analysis

### 4.1 Missing Mechanism: MAR (Missing At Random)

Deck missingness is strongly dependent on pclass.

| pclass | Total | Deck Present | Missing Rate |
|--------|-------|-------------|-------------|
| 1st | 216 | ~130 | ~40% |
| 2nd | 184 | ~15 | ~92% |
| 3rd | 491 | ~8 | ~98% |

- 3rd class passengers have virtually no deck information (~98% missing)
- 1st class passengers retain relatively high deck information (~60% present)
- Deck information availability itself serves as a proxy for social status (= pclass)

### 4.2 Deck Strategy Comparison (5-Fold CV, F1 Score)

| Strategy | Description | F1 Score | Notes |
|----------|------------|----------|-------|
| deck_known binary | Encode missingness as 0/1 | ~0.722 | **Optimal strategy** (simple and effective) |
| Mode imputation | Replace with most frequent deck | ~0.712 | Injects meaningless information |
| Category retention (Unknown) | Add Unknown category | ~0.718 | Marginal gain vs dimensionality cost |

### 4.3 Optimal Strategy: deck_known Binary Variable

- **Method**: Encode as 1 if deck value exists, 0 if missing
- **Rationale**: With 77% missing, the "presence of information" is a stronger signal than actual deck values
- Passengers with deck_known = 1 have significantly higher survival rates (due to higher 1st class proportion)

---

## 5. Embarked Missing Data Analysis

### 5.1 Missing Mechanism: MCAR (Missing Completely At Random)

- Only 2 out of 891 records are missing
- No concentration in any specific pclass, sex, or age group
- At 0.2% missing rate, any method has negligible impact on model performance

### 5.2 Strategy: Mode Imputation

- **Method**: Replace with mode value 'S' (Southampton)
- **Rationale**: ~72% of all passengers embarked at S; 2 missing values are statistically negligible
- MCAR mechanism justifies simple imputation

---

## 6. Relationship Between Missingness and Survival

### 6.1 Chi-square Test Results

| Missing Variable | Chi-square | p-value | Significance |
|-----------------|-----------|---------|-------------|
| age_missing | Significant | < 0.05 | Missing age group has different survival rate |
| deck_missing | Highly significant | < 0.001 | Missing deck (= lower class) has lower survival |
| embarked_missing | Not significant | > 0.05 | Only 2 cases, insufficient power |

### 6.2 Key Findings

- **Missing data itself is a significant survival signal**: Using missingness as a feature is justified
- age_missing: Passengers with missing age have higher 3rd class proportion, thus lower survival
- deck_missing: Most passengers with missing deck are 3rd class, showing markedly lower survival

---

## 7. Final Missing Data Strategy Summary

| Variable | Mechanism | Imputation Method | Dummy Variable | Expected Effect |
|----------|----------|------------------|---------------|----------------|
| age | MAR | pclass+sex group median | age_missing (0/1) | F1 ~0.723 |
| deck | MAR | Not imputed (drop original) | deck_known (0/1) | F1 ~0.722 |
| embarked | MCAR | Mode imputation ('S') | Not needed | Negligible impact |

### Implementation Order

1. Replace 2 missing embarked values with 'S'
2. Compute pclass+sex group median for age, impute missing values, create age_missing dummy
3. Create deck_known binary variable, then drop original deck column
4. Drop alive column (target leakage prevention)

---

## 8. Key Insights Summary

> Missing values are not simply "empty cells" but carry information about the data generation process. In the Titanic dataset, age and deck missingness follows MAR patterns concentrated among 3rd class passengers. This missingness pattern itself has a significant relationship with survival, making missingness indicator variables a justified strategy for model performance improvement.

### Next Steps

1. Implement the above strategies in src/preprocessing/feature_engineer.py
2. Add derived variables (age_missing, deck_known) and re-evaluate baseline
3. Proceed to Phase 2 model tuning

---

## 9. Visualization References

- eda_step6_missing_correlation.png - Missing data correlation heatmap
- eda_step6_age_missing_analysis.png - Age missing pattern analysis
- eda_step6_imputation_comparison.png - Imputation strategy F1 comparison

---

*BDAI Titanic Survival Prediction Project | 2026-04-06*
