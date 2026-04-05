# Titanic Survival Prediction

BDAI 타이타닉 생존 예측 튜토리얼 공모전 프로젝트

## Overview

seaborn의 Titanic 데이터셋을 활용하여 승객 생존 여부를 예측하는 이진 분류 모델을 개발합니다.

- **Platform**: GSTACK
- **Metric**: F1 Score (Survived=1)
- **Baseline**: F1 0.7151 (RandomForest)
- **Target**: F1 0.76+

## Project Structure

```
Titanic-Survival-Prediction/
├── CLAUDE.md              # Project context for AI-assisted development
├── configs/config.yaml    # Hyperparameters & settings
├── docs/
│   ├── ko/                # Korean documents (MD + DOCX)
│   └── en/                # English documents (MD + DOCX)
├── notebooks/             # Jupyter notebooks (EDA → Ensemble)
├── src/
│   ├── preprocessing/     # Feature engineering
│   ├── models/            # Model training & evaluation
│   └── utils/             # Helper functions
├── outputs/               # submission.csv
└── reports/               # Analysis reports (ko/en)
```

## Development Roadmap

| Phase | Description | Target F1 |
|-------|-------------|-----------|
| Phase 1 | Feature Engineering | 0.73~0.74 |
| Phase 2 | Model Tuning (XGBoost, LightGBM) | 0.74~0.76 |
| Phase 3 | Ensemble & Optimization | 0.76+ |

## Version History

| Version | Description | F1 Score | Date |
|---------|-------------|----------|------|
| v0.1 | Baseline (RandomForest) | 0.7151 | 2026-04-05 |

## Rules

- Seed: `SEED = 42` (fixed)
- `survived` column must NOT be used as input feature
- `alive` column is BANNED (target leakage)
- Test labels must not be directly used
- Submission: `submission.csv` (PassengerId, Survived)

## Tech Stack

- Python 3.x
- scikit-learn, XGBoost, LightGBM
- pandas, numpy, seaborn, matplotlib

## Setup

```bash
pip install -r requirements.txt
```

## License

This project is for educational purposes (BDAI Tutorial Competition).
