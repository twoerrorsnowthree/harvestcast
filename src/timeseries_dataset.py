"""
시계열 윈도우 dataset 빌더.

merged_daily.csv 같은 일별 dataframe에서 sliding window를 만든다.
가격이 없는 구간(비출하기)이 있으면 자연스럽게 끊고, 연속 구간 안에서만 윈도우를 만듦.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Mamba 입력 features.
# (주의) humidity_avg / sunshine_sum 은 일부 ASOS 일자료 파일에서 빠져있어 제외.
# 다른 도매시장 (busan/daegu/gwangju/daejeon) 추가 실험 → 가격 동조성으로 효과 X → 제외
DEFAULT_FEATURES = [
    "price_ffill",       # auto-regressive: 주말 ffill된 과거 가락 가격
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


@dataclass
class WindowSpec:
    window: int = 30        # 입력 시퀀스 길이
    horizon: int = 1        # 예측 길이 (1 = next-day)
    features: Sequence[str] = tuple(DEFAULT_FEATURES)
    target: str = "price"


def find_consecutive_segments(df: pd.DataFrame, date_col: str = "date") -> list[pd.DataFrame]:
    """
    날짜가 연속된(diff == 1일) 구간들로 dataframe을 잘라 list로 반환.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    diff = df[date_col].diff().dt.days.fillna(1)
    seg_id = (diff != 1).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(seg_id)]


def build_windows(
    df: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    윈도우 빌더.

    - 입력 features는 모두 non-NaN 이어야 (price_ffill 포함)
    - 타겟 'price'는 윈도우 끝 다음 날(들)이 NaN이 아니어야 함 (실제 trading day)

    이렇게 하면 입력 시퀀스에는 주말 ffill 값이 들어가도 OK이고,
    타겟은 실제 거래일에만 평가하게 된다.

    return:
        X: (N, window, F)
        y: (N, horizon)
        target_dates: (N,)  타겟 첫째날
    """
    feat_cols = list(spec.features)
    # 1) feature 모두 있는 행만 → segment
    # target 이 features 에 이미 있을 경우 중복 컬럼 방지
    needed = ["date"] + feat_cols + ([spec.target] if spec.target not in feat_cols else [])
    sub = df[needed].copy()
    sub_feat_valid = sub.dropna(subset=feat_cols).copy()
    segments = find_consecutive_segments(sub_feat_valid)

    Xs, ys, ts = [], [], []
    min_len = spec.window + spec.horizon
    for seg in segments:
        if len(seg) < min_len:
            continue
        feats = seg[feat_cols].values.astype(np.float32)
        targets = seg[spec.target].values.astype(np.float32)
        dates = seg["date"].values
        for i in range(len(seg) - min_len + 1):
            tgt_slice = targets[i + spec.window : i + spec.window + spec.horizon]
            # 타겟에 NaN 있으면 스킵 (주말 등)
            if np.isnan(tgt_slice).any():
                continue
            Xs.append(feats[i : i + spec.window])
            ys.append(tgt_slice)
            ts.append(dates[i + spec.window])
    if not Xs:
        raise RuntimeError("윈도우를 만들 수 있는 연속 구간이 부족합니다.")
    X = np.stack(Xs, axis=0)
    y = np.stack(ys, axis=0)
    t = np.array(ts)
    return X, y, t


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- 정규화 (train 통계 기반) ----------

class FeatureScaler:
    """학습용 표준화. fit으로 mean/std 계산, transform/inverse_transform 제공."""

    def __init__(self):
        self.feat_mean = None
        self.feat_std = None
        self.tgt_mean = None
        self.tgt_std = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X: (N, L, F)  y: (N, H)
        flat = X.reshape(-1, X.shape[-1])
        self.feat_mean = flat.mean(axis=0)
        self.feat_std = flat.std(axis=0) + 1e-8
        self.tgt_mean = y.mean()
        self.tgt_std = y.std() + 1e-8
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None):
        Xs = (X - self.feat_mean) / self.feat_std
        if y is None:
            return Xs
        ys = (y - self.tgt_mean) / self.tgt_std
        return Xs, ys

    def inverse_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.tgt_std + self.tgt_mean


def chronological_split(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray, test_ratio: float = 0.2
):
    """타겟 날짜 기준 마지막 test_ratio를 test로 분리."""
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    cut = int(len(X) * (1 - test_ratio))
    return (X[:cut], y[:cut], dates[:cut]), (X[cut:], y[cut:], dates[cut:])
