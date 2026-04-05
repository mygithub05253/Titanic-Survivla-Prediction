"""
Model training and evaluation functions for Titanic Survival Prediction project.
타이타닉 생존 예측 프로젝트를 위한 모델 훈련 및 평가 함수들입니다.

This module provides functions for:
- Training machine learning models
- Evaluating model performance with various metrics
- Cross-validation with different strategies

이 모듈은 다음을 위한 함수를 제공합니다:
- 머신 러닝 모델 훈련
- 다양한 지표로 모델 성능 평가
- 다양한 전략의 교차 검증
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.model_selection import cross_validate, cross_val_score


def train_model(X_train: np.ndarray,
               y_train: np.ndarray,
               model_type: str = 'RandomForestClassifier',
               params: Optional[Dict[str, Any]] = None):
    """
    Train a machine learning model with specified parameters.

    지정된 매개변수를 사용하여 머신 러닝 모델을 훈련합니다.

    Args:
        X_train (np.ndarray): Training features. 훈련 특성들.
        y_train (np.ndarray): Training labels. 훈련 레이블들.
        model_type (str): Type of model to train.
                         Default: 'RandomForestClassifier'
                         훈련할 모델의 종류. 기본값: 'RandomForestClassifier'
                         Supported models: 'RandomForestClassifier', 'LogisticRegression',
                         'XGBClassifier', 'LGBMClassifier'
        params (dict, optional): Model parameters. If None, uses default parameters.
                                모델 매개변수. None이면 기본 매개변수 사용.

    Returns:
        model: Trained model instance. 훈련된 모델 인스턴스.

    Raises:
        ValueError: If model_type is not supported.
                   지원되지 않는 모델 종류인 경우 발생합니다.
        Exception: If training fails.
                  훈련이 실패한 경우 발생합니다.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = train_model(X_train, y_train, 'RandomForestClassifier',
        ...                     {'n_estimators': 100, 'max_depth': 5})
        >>> model.score(X_test, y_test)
    """
    # Import required libraries / 필요한 라이브러리 임포트
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Set default parameters if not provided / 매개변수가 없으면 기본값 설정
    if params is None:
        params = {}

    # Select model based on type / 종류에 따라 모델 선택
    if model_type == 'RandomForestClassifier':
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        }
        default_params.update(params)
        model = RandomForestClassifier(**default_params)

    elif model_type == 'LogisticRegression':
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        default_params.update(params)
        model = LogisticRegression(**default_params)

    elif model_type == 'XGBClassifier':
        try:
            from xgboost import XGBClassifier
            default_params = {
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            default_params.update(params)
            model = XGBClassifier(**default_params)
        except ImportError:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")

    elif model_type == 'LGBMClassifier':
        try:
            from lightgbm import LGBMClassifier
            default_params = {
                'random_state': 42,
                'verbose': -1
            }
            default_params.update(params)
            model = LGBMClassifier(**default_params)
        except ImportError:
            raise ImportError("LightGBM is not installed. Please install it with: pip install lightgbm")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model / 모델 훈련
    model.fit(X_train, y_train)

    return model


def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate model performance with various metrics.

    다양한 지표로 모델 성능을 평가합니다.

    Metrics calculated:
    - Accuracy: Overall correctness
    - Precision: True positives / (true positives + false positives)
    - Recall: True positives / (true positives + false negatives)
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under the ROC curve (if probabilities provided)

    계산되는 지표:
    - Accuracy: 전체 정확도
    - Precision: 참 양성 / (참 양성 + 거짓 양성)
    - Recall: 참 양성 / (참 양성 + 거짓 음성)
    - F1-Score: 정밀도와 재현율의 조화 평균
    - ROC-AUC: ROC 곡선 아래 영역 (확률이 제공된 경우)

    Args:
        y_true (np.ndarray): True labels. 참 레이블들.
        y_pred (np.ndarray): Predicted labels. 예측된 레이블들.
        y_pred_proba (np.ndarray, optional): Predicted probabilities for positive class.
                                             양성 클래스에 대한 예측 확률.

    Returns:
        dict: Dictionary with metric names and values.
             지표 이름과 값을 포함하는 딕셔너리.

    Examples:
        >>> metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"F1-Score: {metrics['f1']:.4f}")
    """
    # Calculate metrics / 지표 계산
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # Add ROC-AUC if probabilities are provided / 확률이 제공되면 ROC-AUC 추가
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def cross_validate_model(X: np.ndarray,
                        y: np.ndarray,
                        model,
                        cv: int = 5,
                        scoring: Optional[str] = 'f1') -> Dict[str, Any]:
    """
    Perform cross-validation on a model.

    모델에 대한 교차 검증을 수행합니다.

    Args:
        X (np.ndarray): Features. 특성들.
        y (np.ndarray): Labels. 레이블들.
        model: Model instance to validate. 검증할 모델 인스턴스.
        cv (int): Number of cross-validation folds. Default: 5
                 교차 검증 폴드 수. 기본값: 5
        scoring (str, optional): Scoring metric. Default: 'f1'
                                점수 메트릭. 기본값: 'f1'
                                Supported: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'

    Returns:
        dict: Dictionary with cross-validation results including:
             교차 검증 결과를 포함하는 딕셔너리:
             - 'test_scores': Array of test scores for each fold
             - 'mean_score': Mean test score
             - 'std_score': Standard deviation of test scores
             - 'train_scores': Array of train scores for each fold (if available)

    Examples:
        >>> cv_results = cross_validate_model(X, y, model, cv=5, scoring='f1')
        >>> print(f"Mean F1-Score: {cv_results['mean_score']:.4f}")
        >>> print(f"Std F1-Score: {cv_results['std_score']:.4f}")
    """
    # Validate scoring metric / 점수 메트릭 검증
    valid_scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    if scoring and scoring not in valid_scores:
        raise ValueError(f"Invalid scoring metric: {scoring}. Valid options: {valid_scores}")

    # Perform cross-validation / 교차 검증 수행
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Prepare results / 결과 준비
    results = {
        'test_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'num_folds': cv,
        'scoring_metric': scoring
    }

    return results
