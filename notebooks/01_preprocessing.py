"""
01_preprocessing.py
==================

전체 전처리 파이프라인 실행 스크립트.

진행:
1. 기상 시간별 csv → 일별 집계
2. KAMIS 가격 xls 파싱 + 1kg 상등급 필터
3. 가격 + 기상 머지 + 파생변수 생성
4. 데이터 진단 (결측/이상치/출하기 분포)
5. 시각화 결과를 figures/ 에 저장
6. 최종 dataframe을 data/processed/merged_daily.csv 로 저장

실행:
    cd /Users/archive/Documents/Claude/Projects/capstone
    python notebooks/01_preprocessing.py

또는 jupyter
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 (mac/linux 둘 다 시도)
for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import build_dataset  # noqa: E402

DATA_RAW_WEATHER = PROJECT_ROOT / "data" / "raw" / "weather"
DATA_RAW_PRICE = PROJECT_ROOT / "data" / "raw" / "price"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def diagnose(df: pd.DataFrame) -> None:
    """데이터 상태 콘솔 출력."""
    print("=" * 60)
    print("[데이터 진단]")
    print("=" * 60)
    print(f"전체 일수: {len(df)}")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"가격 결측일: {df['price'].isna().sum()} / {len(df)}")

    season = df[df["is_season"] == 1]
    print(f"\n출하기(11~5월) 일수: {len(season)}, 그 중 가격 있는 일: {season['price'].notna().sum()}")
    nonseason = df[df["is_season"] == 0]
    print(f"비출하기(6~10월) 일수: {len(nonseason)}, 그 중 가격 있는 일: {nonseason['price'].notna().sum()}")

    print("\n월별 가격 데이터 보유 현황:")
    by_month = df.assign(year_month=df["date"].dt.to_period("M")).groupby("year_month").agg(
        n_days=("date", "count"),
        n_price=("price", lambda s: s.notna().sum()),
    )
    print(by_month.to_string())


def plot_overview(df: pd.DataFrame, save_path: Path) -> None:
    """가격/기상 시계열 + 출하기 음영 표시."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 출하기 음영을 위한 plot 함수
    def shade_seasons(ax):
        years = sorted(df["date"].dt.year.unique())
        for y in years + [years[-1] + 1]:
            start = pd.Timestamp(f"{y-1}-11-01")
            end = pd.Timestamp(f"{y}-05-31")
            ax.axvspan(start, end, alpha=0.08, color="C2", label="_nolegend_")

    # 1. 가격
    ax = axes[0]
    ax.plot(df["date"], df["price"], color="C3", linewidth=0.8)
    ax.set_ylabel("가격 (원/1kg)")
    ax.set_title("가락시장 딸기(설향) 1kg 상등급 도매가")
    shade_seasons(ax)

    # 2. 평균기온
    ax = axes[1]
    ax.plot(df["date"], df["temp_avg"], color="C0", linewidth=0.8, label="평균")
    ax.fill_between(df["date"], df["temp_min"], df["temp_max"], alpha=0.15, color="C0", label="최저~최고")
    ax.set_ylabel("기온 (°C)")
    ax.set_title("금산 기온")
    ax.legend(loc="upper right")
    shade_seasons(ax)

    # 3. 강수량
    ax = axes[2]
    ax.bar(df["date"], df["rainfall_sum"], color="C0", width=1.0)
    ax.set_ylabel("강수량 (mm)")
    ax.set_title("일강수량")
    shade_seasons(ax)

    # 4. 일조 (있을 때만) 또는 평균기상온도
    ax = axes[3]
    if "sunshine_sum" in df.columns and df["sunshine_sum"].notna().any():
        ax.plot(df["date"], df["sunshine_sum"], color="C1", linewidth=0.6)
        ax.set_ylabel("일조 (hr)")
        ax.set_title("일조시간")
    elif "ground_temp_avg" in df.columns:
        ax.plot(df["date"], df["ground_temp_avg"], color="C1", linewidth=0.6)
        ax.set_ylabel("지면온도 (°C)")
        ax.set_title("지면온도")
    ax.set_xlabel("날짜")
    shade_seasons(ax)

    fig.suptitle("초록 음영 = 딸기 출하기(11월~5월)", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


def plot_seasonality(df: pd.DataFrame, save_path: Path) -> None:
    """월별 가격 분포 박스플롯."""
    fig, ax = plt.subplots(figsize=(10, 5))
    months = list(range(1, 13))
    data = [df[(df["date"].dt.month == m) & df["price"].notna()]["price"] for m in months]
    ax.boxplot(data, labels=[f"{m}월" for m in months])
    ax.set_ylabel("가격 (원/1kg)")
    ax.set_title("월별 도매가 분포 (boxplot)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


def plot_correlation(df: pd.DataFrame, save_path: Path) -> None:
    """가격 vs 주요 기상 변수 산점도."""
    sub = df.dropna(subset=["price"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 데이터에 있는 컬럼만 사용
    candidates = [
        ("temp_avg", "평균기온 (°C)"),
        ("rainfall_cum7", "최근 7일 누적 강수 (mm)"),
        ("humidity_avg", "평균습도 (%)"),
        ("sunshine_sum", "일조 (hr)"),
        ("ground_temp_avg", "지면온도 (°C)"),
        ("cloud_avg", "전운량 (1/10)"),
    ]
    pairs = [(c, l) for c, l in candidates if c in sub.columns and sub[c].notna().any()][:4]
    for ax, (col, label) in zip(axes.flat, pairs):
        ax.scatter(sub[col], sub["price"], s=8, alpha=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("가격 (원/1kg)")
        corr = sub[[col, "price"]].corr().iloc[0, 1]
        ax.set_title(f"{label} vs 가격  (r={corr:.2f})")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


def main():
    import os
    weather_source = os.environ.get("WEATHER_SOURCE", "geumsan")
    print(f"[1/4] 데이터 빌드 중 (weather_source={weather_source})...")
    df = build_dataset(DATA_RAW_WEATHER, DATA_RAW_PRICE, weather_source=weather_source)

    print("\n[2/4] 데이터 진단")
    diagnose(df)

    print("\n[3/4] 시각화")
    plot_overview(df, FIGURES / "01_overview.png")
    plot_seasonality(df, FIGURES / "02_monthly_boxplot.png")
    plot_correlation(df, FIGURES / "03_corr_scatter.png")

    print("\n[4/4] 저장")
    out_path = DATA_PROCESSED / "merged_daily.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  최종 dataframe → {out_path}  ({len(df)}행 × {len(df.columns)}컬럼)")

    # 출하기만 따로도 저장 (모델 학습용 후보)
    season_df = df[df["is_season"] == 1].dropna(subset=["price"]).copy()
    season_path = DATA_PROCESSED / "merged_season_only.csv"
    season_df.to_csv(season_path, index=False, encoding="utf-8-sig")
    print(f"  출하기+가격 있는 일만 → {season_path}  ({len(season_df)}행)")


if __name__ == "__main__":
    main()
