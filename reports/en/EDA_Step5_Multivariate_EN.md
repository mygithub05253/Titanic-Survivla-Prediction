# EDA Step 5: Multivariate Analysis Report

**BDAI Titanic Survival Prediction Competition | 2026-04-06**

---

## 1. Overview

- **Project**: BDAI Titanic Survival Prediction Competition
- **Phase**: EDA Step 5 - Multivariate Analysis
- **Date**: 2026-04-06
- **Objective**: Analyze variable interactions and correlations to guide feature engineering strategy and assess multicollinearity

---

## 2. Correlation Analysis

### 2.1 Pearson Correlation with Survival

| Variable | Correlation | Direction | Strength |
|----------|------------|-----------|----------|
| sex_male | -0.54 | Negative | Strong |
| pclass | -0.34 | Negative | Moderate |
| fare | +0.26 | Positive | Weak-Moderate |
| parch | +0.08 | Positive | Weak |
| sibsp | -0.04 | Negative | Very Weak |
| age | -0.08 | Negative | Weak |

### Key Findings
- **sex_male (-0.54)**: Being male strongly reduces survival probability - the single strongest correlated variable
- **pclass (-0.34)**: Higher class (lower number) correlates with higher survival
- **fare (+0.26)**: Higher fare correlates with better survival chances - linked to pclass
- sibsp, parch, and age show weak correlations individually

---

## 3. Cramer's V Analysis (Categorical Variable Associations)

| Variable | Cramer's V | Association Strength |
|----------|-----------|---------------------|
| sex | ~0.54 | Strong |
| who | ~0.51 | Strong |
| adult_male | ~0.48 | Moderate-Strong |
| pclass | ~0.34 | Moderate |
| embarked | ~0.17 | Weak |
| alone | ~0.16 | Weak |

### Key Findings
- **sex (~0.54) and who (~0.51)**: Strongest categorical associations with survival
- who encapsulates combined sex+age information, achieving association levels close to sex
- adult_male (~0.48) also shows strong association as a sex+age composite variable
- embarked and alone provide supplementary-level associations

---

## 4. Interaction Analysis

### 4.1 Pclass + Sex Interaction

| Combination | Survival Rate | Count | Notes |
|-------------|--------------|-------|-------|
| 1st class female | ~97% | 94 | Nearly all survived |
| 2nd class female | ~92% | 76 | Very high survival |
| 3rd class female | ~50% | 144 | Female but only half survived |
| 1st class male | ~37% | 122 | 1st class but low survival |
| 2nd class male | ~16% | 108 | Very low |
| 3rd class male | ~14% | 347 | Lowest survival rate |

### Key Findings
- **1st class female (~97%) vs 3rd class male (~14%)**: Survival gap of approximately 7x
- 3rd class female survival (~50%) exceeds 1st class male (~37%) - **sex is a stronger predictor than class**
- The pclass-sex interaction effect is pronounced, making **pclass*sex interaction feature creation essential**

### 4.2 Age + Sex Interaction

| Combination | Survival Rate | Notes |
|-------------|--------------|-------|
| Female child (age<15) | ~65-70% | High survival |
| Male child (age<15) | ~40-45% | Lower than female children |
| Adult female (age>=15) | ~75% | At overall female average |
| Adult male (age>=15) | ~16% | Lowest level |

### Key Findings
- **Male children (~40-45%) survived at notably lower rates than female children (~65-70%)**
- Gender bias existed even within the "women and children first" evacuation principle
- The who variable captures this pattern well (man/woman/child classification)
- When creating is_child feature, consider the interaction with sex

### 4.3 Family Size and Survival

| family_size | Survival Rate | Proportion | Interpretation |
|-------------|--------------|------------|----------------|
| 1 (alone) | ~30% | 60.3% | Most common but low survival |
| 2 | ~55% | 16.1% | High survival |
| 3 | ~58% | 11.2% | Optimal survival range |
| 4 | ~72% | 5.0% | Very high survival |
| 5+ (large) | ~16% | 7.4% | Sharp decline |

### Key Findings
- **Family size 2-4 is the optimal survival range** - appropriately sized families that could help each other
- **Solo travelers (~30%) and large families (~16%)** both show low survival rates
- family_size = sibsp + parch + 1
- Both categorized (alone/small/large) and continuous family_size are valid approaches

### 4.4 Fare + Pclass Interaction

| Class | Median Fare | High Fare Survival | Low Fare Survival | Difference |
|-------|------------|-------------------|-------------------|------------|
| 1st | ~60 | High | Relatively lower | Significant |
| 2nd | ~15 | Medium-high | Medium-low | Significant |
| 3rd | ~8 | Medium | Low | Significant |

### Key Findings
- **Within the same class, higher fare = higher survival** pattern confirmed
- Fare provides independent additional information beyond pclass
- fare_per_person (shared ticket adjustment) or log_fare derived features are useful
- pclass and fare are correlated (-0.55) but not fully redundant

---

## 5. Multicollinearity Analysis (VIF)

| Variable | VIF | Assessment |
|----------|-----|-----------|
| pclass | ~1.5 | Normal |
| age | ~1.2 | Normal |
| sibsp | ~1.4 | Normal |
| parch | ~1.3 | Normal |
| fare | ~1.8 | Normal |

### Key Findings
- **All numeric variables have VIF < 5** - no severe multicollinearity detected
- pclass-fare correlation (-0.55) exists but is within acceptable VIF thresholds
- Tree-based models (GBM, XGBoost, LightGBM) are robust to multicollinearity, so all variables can be safely used
- For linear models, remove redundant variables such as class and embark_town

---

## 6. Feature Engineering Recommendations

### 6.1 Essential Features to Create

| Feature | Method | Rationale |
|---------|--------|-----------|
| pclass_sex | pclass * sex_encoded | pclass+sex interaction is extremely powerful |
| family_size | sibsp + parch + 1 | Optimal 2-4 member survival pattern |
| is_alone | family_size == 1 | Solo travelers show low survival |
| log_fare | log1p(fare) | Removes skewness, adds within-class information |

### 6.2 Recommended Features

| Feature | Method | Rationale |
|---------|--------|-----------|
| is_child | age < 15 | Children show higher survival rates |
| fare_per_person | fare / family_size | Adjusts for shared tickets |
| age_group | Age binning | Captures non-linear age effects |
| deck_known | deck missingness | Missingness itself is a survival signal |

### 6.3 Cautions
- **Drop class column**: Fully redundant with pclass (categorical vs numeric)
- **Drop embark_town**: Same information as embarked
- **Never use alive**: Target leakage (identical to survived)
- who and adult_male are useful as sex+age composites but check for redundancy

---

## 7. Key Insights Summary

> **Core Conclusion**: Variables that show weak individual correlations reveal powerful patterns in interaction analysis. The pclass*sex interaction is the most important cross-effect for survival prediction, while family_size (2-4 optimal) and log_fare are essential derived features for model improvement. Multicollinearity is not severe, allowing safe use of most variables in tree-based models.

### Next Steps (Phase 1 to Phase 2)
1. Implement recommended features in `src/preprocessing/feature_engineer.py`
2. Apply to baseline model and verify F1 Score improvement
3. Begin Phase 2 with GradientBoosting / XGBoost / LightGBM experiments

---

## 8. Visualization References

- `eda_step5_correlation_heatmap.png` - Correlation coefficient heatmap
- `eda_step5_pclass_sex.png` - Pclass + Sex interaction analysis
- `eda_step5_age_sex.png` - Age + Sex interaction analysis
- `eda_step5_family_survival.png` - Survival rate by family size
- `eda_step5_fare_pclass.png` - Fare + Pclass interaction analysis
- `eda_step5_cramers_v.png` - Cramer's V associations

---

*BDAI Titanic Survival Prediction Project | 2026-04-06*
