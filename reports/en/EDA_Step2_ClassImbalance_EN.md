# EDA Step 2: Target Class Imbalance Analysis Report

## 1. Overview

**Project**: BDAI Titanic Survival Prediction Competition
**Phase**: EDA Step 2 - Target Class Imbalance Analysis
**Date**: 2026-04-06
**Objective**: Analyze impact of survival/death ratio imbalance on F1 Score

## 2. Overall Survival Rate

| Item | Value |
|------|-------|
| Survived | 342 (38.4%) |
| Died | 549 (61.6%) |
| Imbalance Ratio | 1.61:1 (Died:Survived) |
| Severity | Moderate (not severe) |

## 3. Group-Level Survival Rates

### Sex (Largest Gap)
| Sex | Survival Rate | Count | Survived |
|-----|--------------|-------|----------|
| female | 74.2% | 314 | 233 |
| male | 18.9% | 577 | 109 |

### Passenger Class
| Class | Survival Rate | Count | Survived |
|-------|--------------|-------|----------|
| 1st | 63.0% | 216 | 136 |
| 2nd | 47.3% | 184 | 87 |
| 3rd | 24.2% | 491 | 119 |

### Who (Role)
| Role | Survival Rate | Count |
|------|--------------|-------|
| woman | 75.6% | 271 |
| child | 59.0% | 83 |
| man | 16.4% | 537 |

### Embarkation Port
| Port | Survival Rate | Count |
|------|--------------|-------|
| C (Cherbourg) | 55.4% | 168 |
| Q (Queenstown) | 39.0% | 77 |
| S (Southampton) | 33.7% | 644 |

### Traveling Alone
| Status | Survival Rate | Count |
|--------|--------------|-------|
| With Family (alone=False) | 50.6% | 354 |
| Alone (alone=True) | 30.4% | 537 |

## 4. DummyClassifier Baselines

| Strategy | F1 | Precision | Recall |
|----------|-----|-----------|--------|
| Most Frequent (All Died) | 0.0000 | 0.0000 | 0.0000 |
| Stratified (Random by ratio) | 0.3429 | 0.3602 | 0.3272 |
| Uniform (Random 50/50) | 0.4260 | 0.3681 | 0.5057 |

## 5. Baseline Model Performance

**Model**: RandomForest (n_estimators=200, max_depth=5)

| Metric | Value |
|--------|-------|
| F1 Score | 0.6885 |
| Precision | 0.7875 |
| Recall | 0.6117 |
| Accuracy | 0.79 |

### Confusion Matrix

| | Pred: Died | Pred: Survived |
|---|-----------|---------------|
| Actual: Died | TN=148 | FP=17 |
| Actual: Survived | **FN=40** | TP=63 |

**Key Issue**: FN=40 - 40 out of 103 survivors (38.8%) were missed

## 6. F1=0.76 Achievement Scenarios

| Precision | Required Recall |
|-----------|----------------|
| 0.75 | 0.7703 |
| 0.80 | 0.7238 |
| 0.84 | 0.6939 |

## 7. Key Findings

1. Imbalance ratio 1.61:1 is moderate (not a severe imbalance issue)
2. **Sex is the strongest survival predictor** (female 74.2% vs male 18.9%)
3. Cabin class, who, and traveling-alone all show significant survival differences
4. **Current Recall (0.61) is the bottleneck** - resolving 40 FN cases is key
5. With Precision 0.80 maintained, achieving Recall 0.72+ yields F1 0.76

## 8. Strategic Direction

1. **Prioritize Recall improvement**: class_weight='balanced', lower threshold
2. **Feature enhancement**: Leverage features with large survival gaps (sex, pclass, who)
3. **FN profile analysis**: Identify common patterns in the 40 missed survivors (Step 7)

## 9. Next Step

Proceed to **EDA Step 3: Univariate Analysis**
- Detailed distribution analysis for each variable
- In-depth feature-target relationship exploration
