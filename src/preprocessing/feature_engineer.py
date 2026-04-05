"""
Feature engineering functions for Titanic Survival Prediction project.
타이타닉 생존 예측 프로젝트를 위한 특성 엔지니어링 함수들입니다.

This module provides functions for:
- Creating family-related features (FamilySize, IsAlone, etc.)
- Creating age-related features (AgeGroup, IsChild, etc.)
- Creating fare-related features (FarePerPerson, FareGroup, etc.)
- Advanced imputation strategies for missing values

이 모듈은 다음을 위한 함수를 제공합니다:
- 가족 관련 특성 생성 (FamilySize, IsAlone 등)
- 나이 관련 특성 생성 (AgeGroup, IsChild 등)
- 요금 관련 특성 생성 (FarePerPerson, FareGroup 등)
- 누락된 값에 대한 고급 대체 전략
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create family-related features from SibSp and Parch columns.

    SibSp 및 Parch 열에서 가족 관련 특성을 생성합니다.

    Features created:
    - FamilySize: Total family size (SibSp + Parch + 1)
    - IsAlone: Whether passenger is alone (1 if FamilySize == 1, else 0)

    생성되는 특성:
    - FamilySize: 총 가족 크기 (SibSp + Parch + 1)
    - IsAlone: 승객이 혼자인지 여부 (FamilySize == 1이면 1, 아니면 0)

    Args:
        df (pd.DataFrame): Input dataframe containing 'SibSp' and 'Parch' columns.
                          'SibSp'와 'Parch' 열을 포함하는 입력 데이터프레임.

    Returns:
        pd.DataFrame: DataFrame with new family-related features added.
                     새로운 가족 관련 특성이 추가된 데이터프레임.

    Raises:
        KeyError: If 'SibSp' or 'Parch' columns are not found in dataframe.
                 'SibSp' 또는 'Parch' 열을 찾을 수 없으면 발생합니다.

    Examples:
        >>> df_new = create_family_features(df)
        >>> df_new[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head()
    """
    # Create a copy to avoid modifying original / 원본 수정을 피하기 위해 복사본 생성
    df_copy = df.copy()

    # Check required columns / 필수 열 확인
    required_cols = ['SibSp', 'Parch']
    if not all(col in df_copy.columns for col in required_cols):
        raise KeyError(f"Missing required columns: {required_cols}")

    # Create FamilySize / FamilySize 생성
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1

    # Create IsAlone / IsAlone 생성
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)

    return df_copy


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age-related features from Age column.

    Age 열에서 나이 관련 특성을 생성합니다.

    Features created:
    - AgeGroup: Categorical age groups (Child, Young Adult, Adult, Senior)
    - IsChild: Whether passenger is a child (Age < 13)
    - IsSenior: Whether passenger is senior (Age >= 60)

    생성되는 특성:
    - AgeGroup: 범주형 나이 그룹 (Child, Young Adult, Adult, Senior)
    - IsChild: 승객이 아이인지 여부 (Age < 13)
    - IsSenior: 승객이 노인인지 여부 (Age >= 60)

    Args:
        df (pd.DataFrame): Input dataframe containing 'Age' column.
                          'Age' 열을 포함하는 입력 데이터프레임.

    Returns:
        pd.DataFrame: DataFrame with new age-related features added.
                     새로운 나이 관련 특성이 추가된 데이터프레임.

    Raises:
        KeyError: If 'Age' column is not found in dataframe.
                 'Age' 열을 찾을 수 없으면 발생합니다.

    Examples:
        >>> df_new = create_age_features(df)
        >>> df_new[['Age', 'AgeGroup', 'IsChild', 'IsSenior']].head()
    """
    # Create a copy to avoid modifying original / 원본 수정을 피하기 위해 복사본 생성
    df_copy = df.copy()

    # Check required columns / 필수 열 확인
    if 'Age' not in df_copy.columns:
        raise KeyError("'Age' column not found in dataframe")

    # Create IsChild / IsChild 생성
    df_copy['IsChild'] = (df_copy['Age'] < 13).astype(int)

    # Create IsSenior / IsSenior 생성
    df_copy['IsSenior'] = (df_copy['Age'] >= 60).astype(int)

    # Create AgeGroup with bins / 구간을 사용하여 AgeGroup 생성
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=False)

    return df_copy


def create_fare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fare-related features from Fare column.

    Fare 열에서 요금 관련 특성을 생성합니다.

    Features created:
    - LogFare: Log-transformed fare (log(Fare + 1) to avoid log(0))
    - FarePerPerson: Fare divided by family size
    - FareGroup: Categorical fare groups (Low, Medium, High, VeryHigh)

    생성되는 특성:
    - LogFare: 로그 변환된 요금 (log(Fare + 1)로 log(0) 피함)
    - FarePerPerson: 요금을 가족 크기로 나눔
    - FareGroup: 범주형 요금 그룹 (Low, Medium, High, VeryHigh)

    Args:
        df (pd.DataFrame): Input dataframe containing 'Fare' and optionally 'FamilySize' columns.
                          'Fare' 및 선택적으로 'FamilySize' 열을 포함하는 입력 데이터프레임.

    Returns:
        pd.DataFrame: DataFrame with new fare-related features added.
                     새로운 요금 관련 특성이 추가된 데이터프레임.

    Raises:
        KeyError: If 'Fare' column is not found in dataframe.
                 'Fare' 열을 찾을 수 없으면 발생합니다.

    Examples:
        >>> df_new = create_fare_features(df)
        >>> df_new[['Fare', 'LogFare', 'FarePerPerson', 'FareGroup']].head()
    """
    # Create a copy to avoid modifying original / 원본 수정을 피하기 위해 복사본 생성
    df_copy = df.copy()

    # Check required columns / 필수 열 확인
    if 'Fare' not in df_copy.columns:
        raise KeyError("'Fare' column not found in dataframe")

    # Create LogFare / LogFare 생성
    df_copy['LogFare'] = np.log1p(df_copy['Fare'])

    # Create FarePerPerson / FarePerPerson 생성
    if 'FamilySize' in df_copy.columns:
        df_copy['FarePerPerson'] = df_copy['Fare'] / df_copy['FamilySize']
    else:
        # Use 1 if FamilySize not available / FamilySize를 사용할 수 없으면 1 사용
        df_copy['FarePerPerson'] = df_copy['Fare']

    # Create FareGroup with quantile-based binning / 분위수 기반 구간으로 FareGroup 생성
    df_copy['FareGroup'] = pd.qcut(df_copy['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'],
                                    duplicates='drop')

    return df_copy


def advanced_imputation(df: pd.DataFrame,
                       age_strategy: str = 'median',
                       fare_strategy: str = 'median',
                       embarked_strategy: str = 'mode') -> pd.DataFrame:
    """
    Apply advanced imputation strategies for missing values.

    누락된 값에 대해 고급 대체 전략을 적용합니다.

    Strategies:
    - Age: 'mean', 'median', 'mode', or group-based strategies
    - Fare: 'mean', 'median', 'mode'
    - Embarked: 'mode' (most common value)

    전략:
    - Age: 'mean', 'median', 'mode', 또는 그룹 기반 전략
    - Fare: 'mean', 'median', 'mode'
    - Embarked: 'mode' (가장 일반적인 값)

    Args:
        df (pd.DataFrame): Input dataframe with potential missing values.
                          잠재적으로 누락된 값을 가진 입력 데이터프레임.
        age_strategy (str): Strategy for imputing Age. Default: 'median'
                           Age 대체 전략. 기본값: 'median'
        fare_strategy (str): Strategy for imputing Fare. Default: 'median'
                            Fare 대체 전략. 기본값: 'median'
        embarked_strategy (str): Strategy for imputing Embarked. Default: 'mode'
                                Embarked 대체 전략. 기본값: 'mode'

    Returns:
        pd.DataFrame: DataFrame with imputed values.
                     대체된 값을 포함하는 데이터프레임.

    Examples:
        >>> df_imputed = advanced_imputation(df, age_strategy='median', fare_strategy='median')
        >>> df_imputed.isnull().sum()
    """
    # Create a copy to avoid modifying original / 원본 수정을 피하기 위해 복사본 생성
    df_copy = df.copy()

    # Impute Age / Age 대체
    if 'Age' in df_copy.columns and df_copy['Age'].isnull().sum() > 0:
        if age_strategy == 'mean':
            df_copy['Age'].fillna(df_copy['Age'].mean(), inplace=True)
        elif age_strategy == 'median':
            df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
        elif age_strategy == 'mode':
            df_copy['Age'].fillna(df_copy['Age'].mode()[0], inplace=True)
        else:
            # Default to median / 기본값으로 중앙값 사용
            df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)

    # Impute Fare / Fare 대체
    if 'Fare' in df_copy.columns and df_copy['Fare'].isnull().sum() > 0:
        if fare_strategy == 'mean':
            df_copy['Fare'].fillna(df_copy['Fare'].mean(), inplace=True)
        elif fare_strategy == 'median':
            df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)
        elif fare_strategy == 'mode':
            df_copy['Fare'].fillna(df_copy['Fare'].mode()[0], inplace=True)
        else:
            # Default to median / 기본값으로 중앙값 사용
            df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)

    # Impute Embarked / Embarked 대체
    if 'Embarked' in df_copy.columns and df_copy['Embarked'].isnull().sum() > 0:
        if embarked_strategy == 'mode':
            df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)
        else:
            # Default to mode / 기본값으로 최빈값 사용
            df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)

    return df_copy
