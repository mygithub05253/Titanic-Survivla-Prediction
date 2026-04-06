# EDA Step 7: Feature Engineering Exploration Report

## 1. Overview
**Phase**: EDA Step 7 - Feature Engineering Exploration
**Date**: 2026-04-06
**Objective**: Create derived features, evaluate importance, establish modeling strategy

## 2. Derived Features Created

| Feature | Formula | Survival Signal |
|---------|---------|----------------|
| family_size | sibsp + parch + 1 | Optimal survival at 2-4 |
| is_alone | family_size == 1 | Lower survival when alone |
| is_child | age <= 12 | Higher survival for children |
| log_fare | log(fare+1) | Reduces skewness |
| fare_per_person | fare / family_size | Per-person fare |
| deck_known | deck is not null | Significant survival signal |
| age_missing | age is null | Missing itself is informative |

## 3. Feature Importance Top 10

1. sex_encoded - Most powerful predictor
2. fare / log_fare - Fare (log-transformed)
3. age - Age (after imputation)
4. pclass - Cabin class
5. family_size - Family size
6. who_encoded - man/woman/child
7. adult_male - Adult male indicator
8. fare_per_person - Per-person fare
9. deck_known - Deck info availability
10. embarked_encoded - Embarkation port

## 4. FN Analysis (Missed Survivors Profile)

- Primarily **3rd class males** (~65%+)
- High proportion of **solo travelers**
- **Lower fares** (significantly lower than TP)
- Age distribution similar to TP

## 5. Recall Improvement Strategy Comparison

| Strategy | F1 | Precision | Recall |
|----------|-----|-----------|--------|
| Default | ~0.72 | ~0.75 | ~0.70 |
| Balanced Weight | Varies | Decreases | Increases |
| Threshold=0.4 | Varies | Decreases | Increases |
| Threshold=0.35 | Varies | Decreases | Increases significantly |

## 6. Final Feature Recommendation

### Essential
- sex_encoded, pclass, fare/log_fare, age

### High Priority
- family_size, who_encoded, adult_male

### Medium Priority
- deck_known, age_missing, embarked_encoded, is_child

### Low Priority (captured by other features)
- fare_per_person, alone, sibsp, parch

### DROP
- alive (BANNED), class (=pclass), embark_town (=embarked), deck (use deck_known)

## 7. Modeling Strategy Recommendations

1. **Phase 2**: Experiment with GradientBoosting, XGBoost, LightGBM
2. **Recall improvement**: Adjust class_weight + optimize threshold
3. **Phase 3**: VotingClassifier, StackingClassifier
4. **Target**: Achieve F1 0.76+
