"""
12_source_region_weather.py
===========================

산지 ASOS 시간자료 (광주/산청/진주/밀양/합천) → 일별 집계 → 시장별 산지 weather 매핑.

매핑 (일일동향 분석 기반):
    가락 ← 광주(담양 대용) + 산청 평균 (전남 + 경남 산청)
    부산 ← 진주 + 밀양 평균 (경남)
    대구 ← 합천 (고령 대용)
    광주 ← 광주 (담양 대용)
    대전 ← 금산 (기존, 충남 — 이미 merged_daily 에 있음)

산출:
    data/raw/weather/source_region_daily.csv   각 station별 일별 집계
    data/processed/market_weather.csv          시장별 산지 weather (long format)
    figures/18_market_weather_compare.png      시장별 평균 기온 비교
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
OUT_DIR_RAW = WEATHER_DIR
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"

for fn in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    plt.rcParams["font.family"] = fn
    break
plt.rcParams["axes.unicode_minus"] = False


# 시장별 산지 ASOS 매핑 (일일동향 분석 기반)
MARKET_SOURCE_STATIONS = {
    "가락": ["광주", "산청"],      # 전남 담양 + 경남 산청
    "부산": ["진주", "밀양"],      # 경남 진주/밀양
    "대구": ["합천"],              # 경북 고령 → 합천 대용
    "광주": ["광주"],              # 전남 담양 → 광주 대용
    # 대전은 금산 기상 그대로 사용 (preprocessing.py)
}


def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df["일시"] = pd.to_datetime(df["일시"])
    return df


def aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """station(지점명) × date 단위 일별 집계."""
    hourly = hourly.copy()
    hourly["date"] = hourly["일시"].dt.normalize()
    # 결측은 자연스럽게 평균/합산에서 처리
    hourly["강수량(mm)"] = hourly["강수량(mm)"].fillna(0)
    # 일조 등 NaN 0 처리

    agg = hourly.groupby(["지점명", "date"]).agg(
        temp_avg=("기온(°C)", "mean"),
        temp_max=("기온(°C)", "max"),
        temp_min=("기온(°C)", "min"),
        rainfall_sum=("강수량(mm)", "sum"),
        wind_avg=("풍속(m/s)", "mean"),
        humidity_avg=("습도(%)", "mean"),
        ground_temp_avg=("지면온도(°C)", "mean"),
        cloud_avg=("전운량(10분위)", "mean"),
    ).reset_index()
    agg["temp_range"] = agg["temp_max"] - agg["temp_min"]
    agg = agg.rename(columns={"지점명": "station"})
    return agg


def main():
    print("[1/3] 시간자료 → 일별 집계")
    all_frames = []
    files = sorted(WEATHER_DIR.glob("*_gwangju.csv")) + sorted(WEATHER_DIR.glob("*_sjmh.csv"))
    for f in files:
        try:
            hourly = load_hourly(f)
            daily = aggregate_daily(hourly)
            print(f"  {f.name}: {len(daily)}행, 지점 {daily['station'].unique()}")
            all_frames.append(daily)
        except Exception as e:
            print(f"  err {f.name}: {e}")

    if not all_frames:
        print("⚠ 데이터 없음"); return
    daily_all = pd.concat(all_frames, ignore_index=True)
    daily_all = daily_all.drop_duplicates(subset=["station", "date"]).sort_values(["date", "station"]).reset_index(drop=True)
    out_path = OUT_DIR_RAW / "source_region_daily.csv"
    daily_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}  ({len(daily_all)}행, {daily_all['station'].nunique()}개 지점)")

    # 지점별 보유일/평균
    print("\n지점별 데이터:")
    summary = daily_all.groupby("station").agg(
        n_days=("date", "count"),
        start=("date", "min"),
        end=("date", "max"),
        temp_avg=("temp_avg", "mean"),
        rainfall_sum=("rainfall_sum", "mean"),
    ).round(1)
    print(summary.to_string())

    print("\n[2/3] 시장별 산지 weather 매핑 (long format)")
    pivot = daily_all.pivot_table(
        index="date",
        columns="station",
        values=["temp_avg", "temp_max", "temp_min", "rainfall_sum", "wind_avg", "humidity_avg", "ground_temp_avg", "cloud_avg"],
    )

    market_rows = []
    for market, stations in MARKET_SOURCE_STATIONS.items():
        # 시장별 산지 평균 (여러 station 이면 평균)
        for var in ["temp_avg", "temp_max", "temp_min", "rainfall_sum", "wind_avg", "humidity_avg", "ground_temp_avg", "cloud_avg"]:
            cols_exist = [s for s in stations if (var, s) in pivot.columns]
            if not cols_exist:
                continue
        market_df = pd.DataFrame({"date": pivot.index})
        for var in ["temp_avg", "temp_max", "temp_min", "rainfall_sum", "wind_avg", "humidity_avg", "ground_temp_avg", "cloud_avg"]:
            cols_exist = [s for s in stations if (var, s) in pivot.columns]
            if cols_exist:
                vals = pivot[var][cols_exist].mean(axis=1).values
                market_df[var] = vals
        market_df["market"] = market
        market_rows.append(market_df)

    market_weather = pd.concat(market_rows, ignore_index=True)
    market_weather["date"] = pd.to_datetime(market_weather["date"])
    market_weather = market_weather.sort_values(["market", "date"]).reset_index(drop=True)
    out2 = PROCESSED / "market_weather.csv"
    market_weather.to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"  저장: {out2}  ({len(market_weather)}행)")

    print("\n[3/3] 시각화 — 시장별 평균기온 비교 (월별)")
    market_weather["ym"] = market_weather["date"].dt.to_period("M")
    monthly = market_weather.groupby(["market", "ym"])["temp_avg"].mean().reset_index()
    monthly["ym"] = monthly["ym"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"가락": "#E63946", "부산": "#F4A261", "대구": "#9D4EDD", "광주": "#6A994E"}
    for market in monthly["market"].unique():
        sub = monthly[monthly["market"] == market]
        ax.plot(sub["ym"], sub["temp_avg"], marker="o", markersize=3, label=market,
                color=colors.get(market, "gray"), linewidth=1.5)
    ax.set_title("시장별 산지 평균기온 월별 추이 (대전=금산 별도)")
    ax.set_xlabel("년월")
    ax.set_ylabel("평균기온 (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "18_market_weather_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {FIGURES / '18_market_weather_compare.png'}")

    # 평균 기온 시장별
    print("\n시장별 산지 평균 기온 (전체 기간):")
    print(market_weather.groupby("market")["temp_avg"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
