"""
Mock predictor — Streamlit UI 개발용.

실제 모델 학습이 끝나면 이 함수를 src/ 의 진짜 모델 호출로 바꾸면 됨.
인터페이스(predict 함수의 입력/출력)는 그대로 유지.
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd


def predict(
    start_date: date,
    horizon_days: int,
    last_price: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    가짜 가격 예측 결과를 반환한다.

    Args:
        start_date: 예측 시작일 (포함)
        horizon_days: 예측 기간 (일수)
        last_price: 직전 실제 가격 (시작점). None 이면 12000원으로 가정.
        seed: 재현성용 시드

    Returns:
        DataFrame with columns:
            date            (datetime)
            predicted_price (float, 원/1kg)
            lower           (float, 95% 신뢰구간 하단)
            upper           (float, 95% 신뢰구간 상단)
    """
    rng = np.random.default_rng(seed)
    base = last_price if last_price is not None else 12000.0

    # 랜덤워크 + 약한 평균회귀
    noise = rng.normal(0, 350, size=horizon_days)
    drift = -10 * np.arange(horizon_days)  # 시즌 후반 하락 추세 흉내
    prices = base + np.cumsum(noise) + drift
    prices = np.clip(prices, 2000, 30000)

    # 신뢰구간: 시간이 갈수록 넓어짐
    band = 400 * np.sqrt(np.arange(1, horizon_days + 1))

    dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "predicted_price": prices,
            "lower": prices - band,
            "upper": prices + band,
        }
    )


def load_historical(merged_csv_path: str) -> pd.DataFrame:
    """
    실제 과거 가격 데이터(merged_daily.csv) 로드. 없으면 mock.
    """
    try:
        df = pd.read_csv(merged_csv_path, parse_dates=["date"])
        df = df.dropna(subset=["price"])[["date", "price"]].sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        # 데이터 없을 때 mock 시계열
        rng = np.random.default_rng(0)
        n = 200
        dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n)
        prices = 10000 + np.cumsum(rng.normal(0, 200, n))
        return pd.DataFrame({"date": dates, "price": prices})
