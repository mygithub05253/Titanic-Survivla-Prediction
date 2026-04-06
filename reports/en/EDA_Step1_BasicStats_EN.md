# EDA Step 1: Basic Statistics Report

## 1. Overview

**Project**: BDAI Titanic Survival Prediction Competition
**Phase**: EDA Step 1 - Basic Statistics Analysis
**Date**: 2026-04-06
**Objective**: Understand overall data structure and fundamental statistics

## 2. Data Summary

| Item | Value |
|------|-------|
| Total Rows | 891 |
| Total Columns | 15 |
| Memory Usage | 278.9 KB |
| Numeric Columns | 6 (survived, pclass, age, sibsp, parch, fare) |
| Categorical Columns | 7 (sex, embarked, class, who, deck, embark_town, alive) |
| Boolean Columns | 2 (adult_male, alone) |

## 3. Column Classification & Roles

### Usable Features (11)
- **Numeric**: pclass (cabin class), age, sibsp (siblings/spouse), parch (parents/children), fare
- **Categorical**: sex, embarked, who (man/woman/child), deck
- **Boolean**: adult_male, alone

### Caution Columns
- **class**: Duplicate of pclass
- **embark_town**: Duplicate of embarked

### Banned Columns
- **alive**: Identical to survived (target leakage) - NEVER use

## 4. Missing Data Status

| Column | Missing Count | Missing % | Severity |
|--------|--------------|-----------|----------|
| deck | 688 | 77.2% | Red (Critical) |
| age | 177 | 19.9% | Orange (Warning) |
| embarked | 2 | 0.2% | Yellow (Minor) |
| embark_town | 2 | 0.2% | Yellow (Minor) |
| Other 11 cols | 0 | 0.0% | Green (Complete) |

## 5. Numeric Variable Descriptive Statistics

### age
- Mean: 29.7, Median: 28.0
- Range: 0.42 - 80
- Skewness: 0.389 (slightly right-skewed)
- 177 missing values (19.9%) - imputation required

### fare
- Mean: 32.2, Median: 14.5
- Range: 0 - 512.3
- Skewness: 4.787 (extremely right-skewed)
- 15 zero-fare tickets - investigation needed

### sibsp / parch
- Mostly 0 (traveling alone): sibsp 68.2%, parch 76.1%
- High skewness: sibsp 3.695, parch 2.749

## 6. Categorical Variable Distributions

| Variable | Categories | Note |
|----------|-----------|------|
| sex | male 64.8%, female 35.2% | Male-dominated |
| embarked | S 72.3%, C 18.9%, Q 8.6% | Southampton dominant |
| who | man 60.3%, woman 30.4%, child 9.3% | Adult males majority |
| class | Third 55.1%, First 24.2%, Second 20.7% | 3rd class majority |
| deck | C 6.6%, B 5.3%, D-G total 10.8% | 77.2% missing |

## 7. Key Findings

1. **Class Imbalance**: 38.4% survival rate (342 survived / 549 died)
2. **Age Missing**: 19.9% - recommend pclass+sex group median imputation
3. **Deck Missing**: 77.2% - usable but requires careful handling
4. **Fare Anomaly**: Extreme skewness (4.79), log transformation recommended
5. **Zero Fare**: 15 tickets with fare=0 - outliers or special cases
6. **Duplicate Columns**: class=pclass, embark_town=embarked
7. **Target Leakage Confirmed**: alive=survived (banned from use)

## 8. Next Step

Proceed to **EDA Step 2: Target Class Imbalance Analysis**
- Analyze impact of 38.4% imbalance on F1 Score
- Compare survival rates across groups
