"""
Utility helper functions for Titanic Survival Prediction project.
타이타닉 생존 예측 프로젝트를 위한 유틸리티 헬퍼 함수들입니다.

This module provides utility functions for:
- Setting random seeds for reproducibility
- Loading training and test data
- Creating submission files

이 모듈은 다음을 위한 유틸리티 함수를 제공합니다:
- 재현성을 위한 랜덤 시드 설정
- 훈련 및 테스트 데이터 로드
- 제출 파일 생성
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    모든 라이브러리에서 재현성을 위한 랜덤 시드를 설정합니다.

    Args:
        seed (int): Random seed value / 랜덤 시드 값

    Returns:
        None

    Examples:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    # Add other library seeds as needed (e.g., TensorFlow, PyTorch)
    # 필요에 따라 다른 라이브러리 시드 추가 (예: TensorFlow, PyTorch)


def load_data(train_path: str = None, test_path: str = None) -> tuple:
    """
    Load training and test data from CSV files.

    CSV 파일에서 훈련 및 테스트 데이터를 로드합니다.

    Args:
        train_path (str, optional): Path to training CSV file.
                                    Default: 'data/train.csv'
                                    훈련 CSV 파일의 경로. 기본값: 'data/train.csv'
        test_path (str, optional): Path to test CSV file.
                                   Default: 'data/test.csv'
                                   테스트 CSV 파일의 경로. 기본값: 'data/test.csv'

    Returns:
        tuple: (train_df, test_df) - Training and test DataFrames
               훈련 및 테스트 DataFrame 튜플

    Raises:
        FileNotFoundError: If the specified files are not found.
                          지정된 파일을 찾을 수 없는 경우 발생합니다.

    Examples:
        >>> train_df, test_df = load_data()
        >>> train_df.shape
        (891, 12)
    """
    # Set default paths / 기본 경로 설정
    if train_path is None:
        train_path = 'data/train.csv'
    if test_path is None:
        test_path = 'data/test.csv'

    # Check if files exist / 파일 존재 여부 확인
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    # Load data / 데이터 로드
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def create_submission(X_test: pd.DataFrame,
                     y_pred: np.ndarray,
                     filename: str = 'submission.csv') -> pd.DataFrame:
    """
    Create a submission file with predictions.

    예측 결과를 포함한 제출 파일을 생성합니다.

    Args:
        X_test (pd.DataFrame): Test features containing PassengerId column.
                              PassengerId 열을 포함하는 테스트 특성들.
        y_pred (np.ndarray): Predicted values for the test set.
                            테스트 세트에 대한 예측값들.
        filename (str, optional): Output filename. Default: 'submission.csv'
                                 출력 파일명. 기본값: 'submission.csv'

    Returns:
        pd.DataFrame: Submission DataFrame with PassengerId and Survived columns.
                     PassengerId와 Survived 열을 포함하는 제출 DataFrame.

    Raises:
        ValueError: If X_test does not contain 'PassengerId' column.
                   X_test가 'PassengerId' 열을 포함하지 않으면 발생합니다.
        ValueError: If length of y_pred does not match X_test.
                   y_pred의 길이가 X_test와 맞지 않으면 발생합니다.

    Examples:
        >>> submission_df = create_submission(X_test, predictions)
        >>> submission_df.to_csv('submission.csv', index=False)
    """
    # Validate inputs / 입력값 검증
    if 'PassengerId' not in X_test.columns:
        raise ValueError("X_test must contain 'PassengerId' column")

    if len(y_pred) != len(X_test):
        raise ValueError(
            f"Length mismatch: y_pred ({len(y_pred)}) vs X_test ({len(X_test)})"
        )

    # Create submission DataFrame / 제출 DataFrame 생성
    submission_df = pd.DataFrame({
        'PassengerId': X_test['PassengerId'].values,
        'Survived': y_pred
    })

    # Ensure output directory exists / 출력 디렉토리 확인
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to file / 파일로 저장
    submission_df.to_csv(filename, index=False)

    return submission_df
