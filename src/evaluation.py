"""
모델 평가 유틸: MAE, RMSE, MAPE + train/test 분할 + walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# ---------- 평가 지표 ----------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """0 또는 매우 작은 값은 분모 안정성을 위해 제외."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-3
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
    }


# ---------- train/test 분할 ----------

@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def train_test_split_by_date(
    df: pd.DataFrame, date_col: str = "date", test_ratio: float = 0.2
) -> Split:
    """
    시간 순서를 유지한 분할. 마지막 test_ratio 만큼이 test.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df) * (1 - test_ratio))
    return Split(train=df.iloc[:cut].copy(), test=df.iloc[cut:].copy())


def train_test_split_by_cutoff(
    df: pd.DataFrame, cutoff: str | pd.Timestamp, date_col: str = "date"
) -> Split:
    """특정 날짜 기준으로 분할. cutoff 이전(<)이 train."""
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff = pd.Timestamp(cutoff)
    return Split(
        train=df[df[date_col] < cutoff].copy(),
        test=df[df[date_col] >= cutoff].copy(),
    )


# ---------- walk-forward validation ----------

def walk_forward(
    df: pd.DataFrame,
    initial_train_size: int,
    step: int = 1,
    horizon: int = 1,
) -> Iterable[Split]:
    """
    Walk-forward(slide-forward) 분할 생성기.

    initial_train_size: 첫 fold에서 train으로 쓸 행 개수
    step: 한 번에 몇 행씩 앞으로 미는지
    horizon: 한 fold의 test 길이 (1이면 1-step-ahead)

    각 fold마다 (train, test) Split을 yield.
    """
    n = len(df)
    start = initial_train_size
    while start + horizon <= n:
        train = df.iloc[:start].copy()
        test = df.iloc[start : start + horizon].copy()
        yield Split(train=train, test=test)
        start += step


# ---------- 결과 비교 표 ----------

def compare_results(named_results: dict[str, dict]) -> pd.DataFrame:
    """
    {모델이름: {MAE, RMSE, MAPE(%)}} → DataFrame
    """
    return pd.DataFrame(named_results).T.round(2)
