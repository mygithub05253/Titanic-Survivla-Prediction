# EDA Step 3: Univariate Analysis Report

## 1. Overview

**Project**: BDAI Titanic Survival Prediction Competition
**Phase**: EDA Step 3 - Univariate Analysis
**Date**: 2026-04-06
**Objective**: Examine individual variable distributions and validate statistical associations with survival

## 2. Numeric Variable Distributions

### 2.1 Age

| Statistic | Value |
|-----------|-------|
| Mean | 29.7 years |
| Median | 28.0 years |
| Std Dev | 14.5 |
| Missing Rate | ~19.9% (177 records) |
| Distribution | Slightly right-skewed |

**Survival Rate by Age Group**:

| Age Group | Survival Rate | Notes |
|-----------|---------------|-------|
| 0-5 | ~67% | Highest survival - children first policy |
| 6-10 | ~40% | Relatively high |
| 11-15 | ~35% | Average level |
| 16-30 | ~35% | Young adults, average |
| 31-50 | ~40% | Middle-aged |
| 51+ | ~35% | Elderly |

**Key Finding**: Children aged 0-5 had ~67% survival rate, far exceeding the overall average of 38.4%. This reflects the "women and children first" evacuation principle.

### 2.2 Fare

| Statistic | Value |
|-----------|-------|
| Mean | 32.2 |
| Median | 14.5 |
| Std Dev | 49.7 |
| Skewness | 4.79 |
| Maximum | 512.3 |
| Zero-fare tickets | 15 |

**Distribution Characteristics**:
- Extreme right skew (skewness=4.79) - log transformation required
- Mean (32.2) is ~2.2x the median (14.5) - indicates extreme outliers
- 15 zero-fare tickets - possibly crew or special passengers
- Log transformation produces near-normal distribution, expected to improve model performance

### 2.3 SibSp / Parch (Family Relations)

| Variable | Mode | Range | Notes |
|----------|------|-------|-------|
| sibsp | 0 (68%) | 0-8 | Majority traveled alone |
| parch | 0 (76%) | 0-6 | Majority without parents/children |

## 3. Categorical Variable Distributions

### 3.1 Sex - Strongest Predictor

| Sex | Count | Proportion | Survival Rate |
|-----|-------|------------|---------------|
| male | 577 | 64.8% | 18.9% |
| female | 314 | 35.2% | 74.2% |

- Highest Chi-square statistic among all categorical variables
- Sex alone provides powerful predictive capability

### 3.2 Pclass (Passenger Class) - Key Predictor

| Class | Count | Proportion | Survival Rate |
|-------|-------|------------|---------------|
| 1st | 216 | 24.2% | 63.0% |
| 2nd | 184 | 20.7% | 47.3% |
| 3rd | 491 | 55.1% | 24.2% |

- Survival gap between 1st (63%) and 3rd (24%) class is approximately 2.6x

### 3.3 Who (Role Classification)

| Role | Count | Survival Rate |
|------|-------|---------------|
| man | 537 | 16.4% |
| woman | 271 | 75.6% |
| child | 83 | 59.0% |

### 3.4 Embarked (Port of Embarkation)

| Port | Count | Survival Rate |
|------|-------|---------------|
| S (Southampton) | 644 | 33.7% |
| C (Cherbourg) | 168 | 55.4% |
| Q (Queenstown) | 77 | 39.0% |

- 2 missing values (can be imputed with mode: S)
- Higher survival rate at Cherbourg due to higher proportion of 1st class passengers

### 3.5 Deck

| Item | Value |
|------|-------|
| Missing Rate | 77.2% (688 records) |
| Valid Data | 203 records |
| deck_known survival | Significantly higher |
| Strategy | Create binary deck_known variable |

- Despite 77.2% missing rate, passengers with deck information show significantly higher survival
- **deck_known** serves as a strong survival signal (1st/2nd class passengers more likely to have deck info)

## 4. Statistical Test Results

### 4.1 Chi-Square Tests - Categorical Variables

All categorical variables show **statistically significant associations** with survival (p < 0.05).

| Variable | Chi-Square | p-value | Significance |
|----------|------------|---------|--------------|
| sex | Highest | < 0.001 | *** (Highly significant) |
| pclass | High | < 0.001 | *** (Highly significant) |
| who | High | < 0.001 | *** (Highly significant) |
| adult_male | High | < 0.001 | *** (Highly significant) |
| alone | Significant | < 0.001 | *** (Significant) |
| embarked | Significant | < 0.05 | * (Significant) |

**Interpretation**: Sex is the single strongest predictor of survival. All categorical variables are significant, warranting inclusion in the model.

### 4.2 Mann-Whitney U Tests - Numeric Variables

| Variable | U-statistic | p-value | Significance |
|----------|-------------|---------|--------------|
| fare | Most significant | < 0.001 | *** |
| pclass | Highly significant | < 0.001 | *** |
| age | Significant | < 0.05 | * |
| sibsp | Significant | < 0.05 | * |
| parch | Significant | < 0.05 | * |

**Interpretation**: Fare and pclass are the most powerful numeric discriminators of survival. Non-parametric Mann-Whitney U test was used due to non-normal distributions.

## 5. Key Insights Summary

### 5.1 Variable Importance Ranking

1. **sex** - Strongest predictor (highest Chi-square)
2. **pclass / fare** - Reflects socioeconomic status
3. **who / adult_male** - Combined sex + age information
4. **deck_known** - Missingness itself is a survival signal
5. **age** - 0-5 year olds show highest survival rate
6. **alone / embarked** - Supplementary predictors

### 5.2 Preprocessing Recommendations

1. **fare**: Log transformation essential (skewness 4.79)
2. **age**: Impute with median by pclass+sex groups
3. **deck**: Create binary deck_known variable
4. **embarked**: Impute 2 missing values with mode (S)
5. **Zero fares**: Consider separate flag or group median imputation

### 5.3 Core Conclusion

> **All variables show statistically significant relationships with survival.**
> Sex and pclass are the strongest predictors, while deck_known reveals that missingness patterns themselves serve as powerful survival signals.
> Log transformation of fare and group-based age imputation are critical preprocessing steps for model performance improvement.

## 6. Visualization References

- `eda_step3_numeric_dist.png` - Numeric variable distributions
- `eda_step3_age_detail.png` - Detailed age analysis
- `eda_step3_fare_detail.png` - Detailed fare analysis
- `eda_step3_categorical_dist.png` - Categorical variable distributions
- `eda_step3_deck_analysis.png` - Deck analysis
- `eda_step3_feature_vs_target.png` - Feature vs. survival relationships

---
*Author: BDAI Titanic Survival Prediction Project | 2026-04-06*
