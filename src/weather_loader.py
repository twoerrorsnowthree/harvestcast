"""
기상청 ASOS 관측 데이터 로더 + 일별 집계.

지원 파일 종류 (모두 cp949 인코딩):
    OBS_ASOS_TIM_*.csv  : 시간별 관측자료 → 일별로 집계 필요
    OBS_ASOS_DD_*.csv   : 일자료 (이미 일별)

다지점 파일도 지원 (예: 서산/홍성/천안/보령/부여/금산 같이 들어있는 csv).
station_filter 인자로 특정 지점만 추출.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


# 시간자료 컬럼
WEATHER_COLUMN_MAP_TIM = {
    "지점": "station_id",
    "지점명": "station_name",
    "일시": "datetime",
    "기온(°C)": "temp_c",
    "강수량(mm)": "rainfall_mm",
    "풍속(m/s)": "wind_ms",
    "풍향(16방위)": "wind_dir",
    "습도(%)": "humidity_pct",
    "일조(hr)": "sunshine_hr",
    "적설(cm)": "snow_cm",
    "전운량(10분위)": "cloud",
    "지면상태(지면상태코드)": "ground_code",
    "지면온도(°C)": "ground_temp_c",
}

# 일자료 컬럼
WEATHER_COLUMN_MAP_DD = {
    "지점": "station_id",
    "지점명": "station_name",
    "일시": "date",
    "평균기온(°C)": "temp_avg",
    "최저기온(°C)": "temp_min",
    "최고기온(°C)": "temp_max",
    "일강수량(mm)": "rainfall_sum",
    "평균 풍속(m/s)": "wind_avg",
    "평균 상대습도(%)": "humidity_avg",
    "평균 현지기압(hPa)": "pressure_avg",
    "합계 일사량(MJ/m2)": "solar_sum",
    "평균 전운량(1/10)": "cloud_avg",
    "평균 지면온도(°C)": "ground_temp_avg",
}


DEFAULT_STATION = "금산"


# ---------- 시간자료(TIM) ----------

def load_hourly_files(weather_dir: str | Path, station_filter: str | None = DEFAULT_STATION) -> pd.DataFrame:
    """
    weather_dir 안의 OBS_ASOS_TIM_*.csv 파일들을 합쳐 시간별 dataframe.
    station_filter가 주어지면 해당 지점만 남김.
    """
    weather_dir = Path(weather_dir)
    files = sorted(weather_dir.glob("OBS_ASOS_TIM_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="cp949")
        df = df.rename(columns=WEATHER_COLUMN_MAP_TIM)
        if station_filter:
            df = df[df["station_name"] == station_filter]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["rainfall_mm"] = out["rainfall_mm"].fillna(0)
    out["snow_cm"] = out["snow_cm"].fillna(0)
    out["sunshine_hr"] = out["sunshine_hr"].fillna(0)
    out = out.drop_duplicates(subset=["station_id", "datetime"]).sort_values("datetime")
    return out.reset_index(drop=True)


# ---------- 일자료(DD) ----------

def load_daily_dd_files(weather_dir: str | Path, station_filter: str | None = DEFAULT_STATION) -> pd.DataFrame:
    """
    weather_dir 안의 OBS_ASOS_DD_*.csv 파일들을 합쳐 일별 dataframe.
    다지점 파일이면 station_filter 로 한 지점만 남김.
    """
    weather_dir = Path(weather_dir)
    files = sorted(weather_dir.glob("OBS_ASOS_DD_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="cp949")
        df = df.rename(columns=WEATHER_COLUMN_MAP_DD)
        if station_filter:
            df = df[df["station_name"] == station_filter]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["rainfall_sum"] = out["rainfall_sum"].fillna(0)
    # 일자료 표준 출력 컬럼
    keep = [
        "station_id", "station_name", "date",
        "temp_avg", "temp_max", "temp_min",
        "rainfall_sum", "wind_avg", "humidity_avg",
        "ground_temp_avg", "cloud_avg", "solar_sum",
    ]
    out = out[[c for c in keep if c in out.columns]].drop_duplicates(subset=["date"]).sort_values("date")
    out["temp_range"] = out["temp_max"] - out["temp_min"]
    # 일조시간(sunshine_sum)은 DD 파일엔 없고 일사량(solar_sum)만 있음 → solar_sum을 sunshine_sum으로도 노출
    # (단위는 MJ/m2 로 다름, 변수명은 모델 호환을 위해 유지)
    if "sunshine_sum" not in out.columns and "solar_sum" in out.columns:
        out["sunshine_sum"] = out["solar_sum"]
    return out.reset_index(drop=True)


def aggregate_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    시간별 → 일별 집계.
    기온: 평균/최고/최저
    강수량: 합계
    풍속: 평균
    습도: 평균
    일조: 합계
    """
    hourly = hourly.copy()
    hourly["date"] = hourly["datetime"].dt.normalize()

    agg = hourly.groupby(["station_id", "station_name", "date"]).agg(
        temp_avg=("temp_c", "mean"),
        temp_max=("temp_c", "max"),
        temp_min=("temp_c", "min"),
        rainfall_sum=("rainfall_mm", "sum"),
        wind_avg=("wind_ms", "mean"),
        humidity_avg=("humidity_pct", "mean"),
        sunshine_sum=("sunshine_hr", "sum"),
        ground_temp_avg=("ground_temp_c", "mean"),
        cloud_avg=("cloud", "mean"),
    ).reset_index()

    # 파생변수: 일교차
    agg["temp_range"] = agg["temp_max"] - agg["temp_min"]
    return agg


def load_daily_weather(
    weather_dir: str | Path, station_filter: str | None = DEFAULT_STATION
) -> pd.DataFrame:
    """
    weather_dir 안의 ASOS 파일을 자동으로 읽어 일별 dataframe.

    우선순위:
        1. 일자료(DD) 파일이 있으면 그쪽을 메인으로 사용
        2. 시간자료(TIM) 파일은 보조 — DD 가 커버하지 않는 기간만 추가
        3. DD/TIM 둘 다 있으면 같은 날짜는 DD 우선

    station_filter: 다지점 파일에서 추출할 지점명 (기본 '금산')
    """
    weather_dir = Path(weather_dir)
    daily_dd = load_daily_dd_files(weather_dir, station_filter)
    hourly = load_hourly_files(weather_dir, station_filter)

    if daily_dd.empty and hourly.empty:
        raise FileNotFoundError(f"기상 csv 파일이 없습니다: {weather_dir}")

    if not hourly.empty:
        daily_tim = aggregate_to_daily(hourly)
    else:
        daily_tim = pd.DataFrame()

    # DD 우선, TIM은 DD가 없는 날짜만
    if not daily_dd.empty and not daily_tim.empty:
        dd_dates = set(daily_dd["date"])
        tim_extra = daily_tim[~daily_tim["date"].isin(dd_dates)]
        # 컬럼 정합성: DD 컬럼 기준으로 맞춤
        common_cols = [c for c in daily_dd.columns if c in tim_extra.columns]
        merged = pd.concat([daily_dd[common_cols], tim_extra[common_cols]], ignore_index=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged
    elif not daily_dd.empty:
        return daily_dd
    else:
        return daily_tim


if __name__ == "__main__":
    import sys

    weather_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/weather"
    daily = load_daily_weather(weather_dir)
    print(f"shape: {daily.shape}")
    print(f"기간: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"컬럼: {list(daily.columns)}")
    print(daily.head())
