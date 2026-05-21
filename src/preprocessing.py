"""
가격(KAMIS) + 기상(ASOS) 머지 및 파생변수 생성

최종 산출 컬럼:
    date, price, temp_avg, temp_max, temp_min, temp_range,
    rainfall_sum, wind_avg, humidity_avg, sunshine_sum, ground_temp_avg, cloud_avg,
    price_lag1, price_lag7, price_ma7, price_ma14,
    temp_ma7, rainfall_cum7, rainfall_cum14
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from weather_loader import load_daily_weather
from price_loader import load_all_prices


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """가격 기반 파생변수 (lag, moving average)."""
    df = df.copy()
    df["price_lag1"] = df["price"].shift(1)
    df["price_lag7"] = df["price"].shift(7)
    df["price_ma7"] = df["price"].rolling(window=7, min_periods=1).mean()
    df["price_ma14"] = df["price"].rolling(window=14, min_periods=1).mean()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """기상 기반 파생변수 (이동평균, 누적강수)."""
    df = df.copy()
    df["temp_ma7"] = df["temp_avg"].rolling(window=7, min_periods=1).mean()
    df["rainfall_cum7"] = df["rainfall_sum"].rolling(window=7, min_periods=1).sum()
    df["rainfall_cum14"] = df["rainfall_sum"].rolling(window=14, min_periods=1).sum()
    df["humidity_ma7"] = df["humidity_avg"].rolling(window=7, min_periods=1).mean()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """달력 기반 변수 (월, 요일, 출하기 플래그)."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek  # 0=월
    # 딸기 출하기: 11월 ~ 익년 5월
    df["is_season"] = df["month"].isin([11, 12, 1, 2, 3, 4, 5]).astype(int)
    return df


def add_filled_price(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    """
    KAMIS 가격은 주말/공휴일에 비어있음. 짧은 gap(최대 3일)만 forward-fill 한다.
    출하기/비출하기 사이의 긴 공백(수개월)은 그대로 NaN으로 남김.
    """
    df = df.copy()
    # 짧은 gap만 ffill: limit=max_gap_days
    df["price_ffill"] = df["price"].ffill(limit=max_gap_days)
    return df


def add_other_markets(df: pd.DataFrame, csv_path: str | Path, max_gap_days: int = 3) -> pd.DataFrame:
    """
    다른 도매시장 (부산/대구/광주/대전) 가격을 보조 feature 로 머지.
    가락 가격과 마찬가지로 주말 ffill 처리.

    Args:
        df: 기존 머지 결과
        csv_path: kamis_other_markets.csv 경로
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [info] 다른 시장 데이터 없음 ({csv_path.name}) — 스킵")
        return df

    other = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.copy().merge(other, on="date", how="left")

    # 각 시장 ffill (주말 등 짧은 공백 채움)
    for col in ["busan", "daegu", "gwangju", "daejeon"]:
        if col in df.columns:
            df[f"{col}_ffill"] = df[col].ffill(limit=max_gap_days)
    return df


def load_market_source_weather(market_weather_csv: str | Path, market: str = "가락") -> pd.DataFrame:
    """
    market_weather.csv (12_source_region_weather.py 산출) 에서 특정 시장의 산지 weather 가져오기.

    가락 산지 = 광주(담양 대용) + 산청 평균.
    """
    df = pd.read_csv(market_weather_csv, parse_dates=["date"])
    sub = df[df["market"] == market].drop(columns=["market"]).copy()
    # temp_range 추가
    if "temp_max" in sub.columns and "temp_min" in sub.columns:
        sub["temp_range"] = sub["temp_max"] - sub["temp_min"]
    return sub


def build_dataset(
    weather_dir: str | Path,
    price_dir: str | Path,
    weather_source: str = "geumsan",
    market_weather_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    기상 + 가격 → 최종 일별 dataset.

    Args:
        weather_source: 'geumsan' (기존, 금산 ASOS) 또는 'gala_source' (가락 산지 = 광주+산청 평균)
        market_weather_csv: 'gala_source' 일 때 필요. 기본 None → data/processed/market_weather.csv 자동 탐색
    """
    if weather_source == "gala_source":
        if market_weather_csv is None:
            project_root = Path(__file__).resolve().parent.parent
            market_weather_csv = project_root / "data" / "processed" / "market_weather.csv"
        weather = load_market_source_weather(market_weather_csv, market="가락")
        print(f"  [weather] 가락 산지(광주+산청) 사용: {len(weather)}일")
    else:
        weather = load_daily_weather(weather_dir)
        weather = weather.drop(columns=["station_id", "station_name"])
        print(f"  [weather] 금산 ASOS 사용: {len(weather)}일")

    price = load_all_prices(price_dir)

    # 기상 일별 date 컬럼은 datetime64이고, price도 datetime64 → 그대로 머지 가능
    # 가격은 결측이 있을 수 있으므로 left join: 기상 기준
    merged = weather.merge(price, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)

    # 주말 ffill 가격 컬럼 추가 (Mamba 윈도우 입력용)
    merged = add_filled_price(merged, max_gap_days=3)

    # 다른 도매시장 가격 보조 feature 로 머지 (있을 경우)
    merged = add_other_markets(merged, Path(price_dir) / "kamis_other_markets.csv")

    merged = add_price_features(merged)
    merged = add_weather_features(merged)
    merged = add_calendar_features(merged)
    return merged


if __name__ == "__main__":
    # 프로젝트 루트 기준으로 경로 잡기
    project_root = Path(__file__).resolve().parent.parent
    df = build_dataset(project_root / "data/raw/weather", project_root / "data/raw/price")
    print(f"shape: {df.shape}")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"가격 결측일: {df['price'].isna().sum()} / {len(df)}")
    print(f"\n컬럼: {list(df.columns)}")
    print(df.head(3))
