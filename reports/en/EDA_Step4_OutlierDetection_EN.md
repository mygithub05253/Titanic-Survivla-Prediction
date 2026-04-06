# EDA Step 4: Outlier Detection Report

**BDAI Titanic Survival Prediction Competition | 2026-04-06**

---

## 1. Overview

- **Project**: BDAI Titanic Survival Prediction Competition
- **Phase**: EDA Step 4 - Outlier Detection
- **Date**: 2026-04-06
- **Objective**: Identify outliers in numeric variables, analyze their impact on survival prediction, and establish optimal handling strategies

---

## 2. IQR Method Analysis

Outlier detection results using the IQR (Interquartile Range) method:

| Variable | Outlier Count | Percentage | Q1 | Q3 | IQR | Lower | Upper |
|----------|--------------|------------|-----|-----|-----|-------|-------|
| fare | 116 | 13.0% | 7.91 | 31.0 | 23.09 | -26.7 | 65.6 |
| sibsp | 18 | 2.0% | 0 | 1 | 1 | -1.5 | 2.5 |
| parch | 6 | 0.7% | 0 | 0 | 0 | 0 | 0 |
| age | 1 | 0.1% | 20.1 | 38.0 | 17.9 | -6.7 | 64.8 |

### Key Findings
- **Fare has the most outliers** (116, 13.0%) - extreme right-tail distribution
- **Age has virtually no outliers** (1 case) - relatively normal distribution
- **sibsp/parch** - a small number of large families detected as outliers

---

## 3. Z-Score Method Analysis

Outlier detection results using |Z| > 3 threshold:

| Variable | |Z|>3 Outliers | Max |Z| Value |
|----------|---------------|---------------|
| fare | Most | Very high |
| sibsp | Few | High |
| parch | Few | High |
| age | None | Normal range |

- Z-score method also identifies fare as having the most extreme outliers
- The extreme skewness (4.79) of fare greatly amplifies Z-scores

---

## 4. Fare Detailed Analysis

### 4.1 Basic Statistics

| Item | Value |
|------|-------|
| Maximum | 512.33 |
| Mean | 32.2 |
| Median | 14.5 |
| Skewness | 4.79 |

### 4.2 Zero-Fare Tickets (15 cases)

| Characteristic | Description |
|---------------|-------------|
| Count | 15 cases |
| Passenger Class | Mostly 3rd class males |
| Survival Rate | **Very low** |
| Probable Cause | Crew members or special passengers |

- Zero-fare passengers are predominantly 3rd class males
- Their survival rate is well below the overall average, providing meaningful model information

### 4.3 High-Fare Passengers

- High-fare passengers (fare > 100) are **predominantly 1st class**
- Their survival rate is **very high**
- **DO NOT remove** - critical information for survival prediction

---

## 5. SibSp / Parch Detailed Analysis

| Variable | Maximum | Outlier Characteristics | Strategy |
|----------|---------|----------------------|----------|
| sibsp | 8 | Large families (rare but informative) | Keep |
| parch | 6 | Large families (rare but informative) | Keep |

- Extreme values are rare but carry important information for family_size feature engineering
- Large families (family_size > 4) show distinct survival patterns; removal causes information loss

---

## 6. Outlier Handling Strategy

| Variable | Strategy | Rationale |
|----------|----------|-----------|
| age | **Keep as-is** | Only 1 outlier, represents actual elderly passenger |
| fare | **Log transform** | Removes skewness, approximates normal distribution, preserves information |
| sibsp | **Keep as-is** | Large family info is critical for family_size feature |
| parch | **Keep as-is** | Large family info is critical for family_size feature |

---

## 7. Key Insight

> **Outlier removal would hurt Recall.** Removing high-fare passengers (1st class, high survival) and large family data would lose positive class (survived) information, causing Recall to drop. Given the current model's low Recall (0.62), outlier removal would be counterproductive.

### Recommendations
1. **fare**: Log transform to normalize distribution (naturally reduces outlier impact)
2. **age**: No outlier treatment needed
3. **sibsp/parch**: Keep original values, leverage via family_size feature
4. **Zero fares**: Consider creating a separate flag (is_zero_fare)

---

## 8. Visualization References

- `eda_step4_iqr_boxplot.png` - IQR-based box plots
- `eda_step4_zscore_dist.png` - Z-score distributions
- `eda_step4_outlier_vs_survival.png` - Outliers vs. survival rates

---

*BDAI Titanic Survival Prediction Project | 2026-04-06*
