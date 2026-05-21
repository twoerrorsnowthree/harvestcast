"""
베이스라인 모델 모음.

각 모델은 fit(train_df) 으로 학습하고 predict(test_df) 으로 예측한다.
모든 모델은 동일한 인터페이스라서 서로 바꿔 끼워 비교 가능.

타겟 컬럼: 'price'
사용 가능한 feature: lag price, 기상, 달력 — preprocessing.py 가 이미 만들어둠.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


DEFAULT_FEATURES = [
    # 가락 가격 파생
    "price_lag1",
    "price_lag7",
    "price_ma7",
    "price_ma14",
    # 기상
    "temp_avg",
    "temp_max",
    "temp_min",
    "temp_range",
    "rainfall_sum",
    "rainfall_cum7",
    "wind_avg",
    "ground_temp_avg",
    "cloud_avg",
    # 달력
    "month",
    "dow",
]


# ---------- 1. Naive baselines ----------

@dataclass
class NaiveLag:
    """test 시점의 price = train+test 통합 시계열에서 K일 전 price."""

    k: int = 1
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Naive lag-{self.k}"

    def predict_on_full(self, full_df: pd.DataFrame) -> pd.Series:
        """전체 dataframe(train+test)에서 lag-k 예측을 만들어준다."""
        return full_df["price"].shift(self.k)


@dataclass
class SeasonalNaive:
    """1년 전 같은 날짜의 가격으로 예측."""

    period_days: int = 365
    name: str = "Seasonal Naive (year-1)"

    def predict_on_full(self, full_df: pd.DataFrame) -> pd.Series:
        return full_df["price"].shift(self.period_days)


# ---------- 2. ML baselines ----------

@dataclass
class FeatureModel:
    """sklearn 호환 회귀 모델을 wrap. 학습/예측 시 features만 떼어 사용."""

    model: object
    features: Sequence[str] = field(default_factory=lambda: DEFAULT_FEATURES)
    name: str = "FeatureModel"

    def _xy(self, df: pd.DataFrame):
        df = df.dropna(subset=list(self.features) + ["price"])
        X = df[list(self.features)].values
        y = df["price"].values
        return X, y, df

    def fit(self, train: pd.DataFrame):
        X, y, _ = self._xy(train)
        self.model.fit(X, y)
        return self

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        df = test.dropna(subset=list(self.features)).copy()
        X = df[list(self.features)].values
        df["pred"] = self.model.predict(X)
        return df[["date", "price", "pred"]]


def make_ridge() -> FeatureModel:
    return FeatureModel(model=Ridge(alpha=1.0), name="Ridge")


def make_random_forest(n_estimators: int = 200, random_state: int = 42) -> FeatureModel:
    return FeatureModel(
        model=RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        name="RandomForest",
    )


def make_xgboost(random_state: int = 42) -> FeatureModel:
    from xgboost import XGBRegressor

    return FeatureModel(
        model=XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        name="XGBoost",
    )


# ---------- 3. 통합 실행 ----------

def run_naive_predictions(
    full_df: pd.DataFrame, test_dates: pd.Series
) -> dict[str, pd.DataFrame]:
    """test_dates 에 해당하는 행에 대한 naive 예측들."""
    out = {}
    for model in [NaiveLag(k=1), NaiveLag(k=7), SeasonalNaive()]:
        preds = model.predict_on_full(full_df)
        sub = full_df.assign(pred=preds)
        sub = sub[sub["date"].isin(test_dates)][["date", "price", "pred"]]
        out[model.name] = sub
    return out


def run_feature_models(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """3가지 ML 모델을 학습하고 test 예측 dataframe 반환."""
    out = {}
    for factory in [make_ridge, make_random_forest, make_xgboost]:
        m = factory()
        m.fit(train)
        out[m.name] = m.predict(test)
    return out
