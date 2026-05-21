"""
10_variety_comparison.py
========================

가락도매 4개 딸기 품종 가격 비교 분석.

품종:
    설향   (메인 모델 타겟, 1kg 상등급)
    금실   (1kg 상등급)
    킹스베리 (1kg 상등급)
    죽향   (2kg 상등급만 거래 → ÷2 로 1kg 환산)

산출:
    data/processed/variety_summary.csv   품종별 기초 통계
    figures/14_variety_avg_price.png     월별 평균가 비교
    figures/15_variety_distribution.png  가격 분포 (boxplot)
    figures/16_variety_seasonal.png      시즌별 트렌드
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from price_loader import load_price_xls  # noqa: E402

PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "price"
RESULTS = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"

# 한글 폰트
for fn in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    plt.rcParams["font.family"] = fn
    break
plt.rcParams["axes.unicode_minus"] = False


# 품종별 파일 패턴 + KAMIS item 명 + 단위 (가격 환산 배수)
# 설향은 기존 가격 xls 들 (파일명에 품종 안 적혀있음) 에서 추출
VARIETIES = {
    # 설향은 옛날 파일(이름에 변종 표기 없음) + 새 파일(_seolhyang) 둘 다
    "설향":   {"file_pattern": "BOTH_SEOLHYANG",  "item": "딸기 설향",    "unit": "1키로상자", "multiplier": 1.0, "color": "#E63946"},
    "금실":   {"file_pattern": "*_keumsil*.xls",   "item": "딸기 금실",    "unit": "1키로상자", "multiplier": 1.0, "color": "#F4A261"},
    "죽향":   {"file_pattern": "*_jukhyang*.xls",  "item": "딸기 죽향",    "unit": "2키로상자", "multiplier": 0.5, "color": "#6A994E"},
    "킹스베리": {"file_pattern": "*_kingsberry*.xls", "item": "딸기 킹스베리", "unit": "1키로상자", "multiplier": 1.0, "color": "#9D4EDD"},
}

GRADE = "상"


def load_variety(name: str, conf: dict) -> pd.DataFrame:
    """한 품종의 1kg 상등급 일별 가격 시리즈 반환 (kg 환산)."""
    if conf["file_pattern"] == "BOTH_SEOLHYANG":
        # 설향: _seolhyang 붙은 새 파일 + 변종 표기 없는 옛 파일
        all_xls = sorted(PRICE_DIR.glob("*.xls"))
        files = [f for f in all_xls if (
            "_seolhyang" in f.name.lower()
            or not any(t in f.name.lower() for t in ["keumsil", "jukhyang", "kingsberry", "seolhyang"])
        )]
    else:
        files = sorted(PRICE_DIR.glob(conf["file_pattern"]))

    frames = []
    for f in files:
        try:
            df = load_price_xls(f)
            sub = df[
                (df["item"] == conf["item"]) &
                (df["unit"] == conf["unit"]) &
                (df["grade"] == GRADE)
            ].copy()
            if len(sub) > 0:
                sub["price_per_kg"] = sub["price"] * conf["multiplier"]
                frames.append(sub[["date", "price_per_kg"]])
        except Exception as e:
            print(f"  skip {f.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["date", "price_per_kg"])
    out = pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def main():
    print("[1/4] 품종별 데이터 로드")
    data = {}
    for name, conf in VARIETIES.items():
        df = load_variety(name, conf)
        data[name] = df
        if len(df):
            print(f"  {name:8s}: {len(df):4d}일,  {df['date'].min().date()} ~ {df['date'].max().date()},  평균 {df['price_per_kg'].mean():.0f}원/kg")
        else:
            print(f"  {name:8s}: 0일 (데이터 없음)")

    print("\n[2/4] 통계 요약")
    rows = []
    for name, df in data.items():
        if len(df) == 0:
            continue
        rows.append({
            "품종": name,
            "일수": len(df),
            "기간_시작": df["date"].min().date(),
            "기간_끝": df["date"].max().date(),
            "평균(원/kg)": round(df["price_per_kg"].mean()),
            "중앙값": round(df["price_per_kg"].median()),
            "최저": round(df["price_per_kg"].min()),
            "최고": round(df["price_per_kg"].max()),
            "표준편차": round(df["price_per_kg"].std()),
            "변동계수(%)": round(df["price_per_kg"].std() / df["price_per_kg"].mean() * 100, 1),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS / "variety_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n  저장: {RESULTS / 'variety_summary.csv'}")

    print("\n[3/4] 시각화")

    # ---- 그림 1: 월별 평균가 비교 라인 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, df in data.items():
        if len(df) == 0:
            continue
        monthly = df.set_index("date")["price_per_kg"].resample("MS").mean()
        ax.plot(monthly.index, monthly.values, marker="o", label=name,
                color=VARIETIES[name]["color"], linewidth=2, markersize=4)
    ax.set_title("품종별 월별 평균 도매가 (1kg 환산)")
    ax.set_ylabel("가격 (원/kg)")
    ax.set_xlabel("년월")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "14_variety_avg_price.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {FIGURES / '14_variety_avg_price.png'}")

    # ---- 그림 2: 가격 분포 boxplot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [n for n, df in data.items() if len(df) > 0]
    values = [data[n]["price_per_kg"].values for n in names]
    colors = [VARIETIES[n]["color"] for n in names]
    bp = ax.boxplot(values, tick_labels=names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title("품종별 도매가 분포 (1kg 환산)")
    ax.set_ylabel("가격 (원/kg)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "15_variety_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {FIGURES / '15_variety_distribution.png'}")

    # ---- 그림 3: 시즌별(월별) 평균 — 시즌성 비교 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in data.items():
        if len(df) == 0:
            continue
        df2 = df.copy()
        df2["month"] = df2["date"].dt.month
        by_month = df2.groupby("month")["price_per_kg"].mean()
        ax.plot(by_month.index, by_month.values, marker="o", label=name,
                color=VARIETIES[name]["color"], linewidth=2, markersize=6)
    ax.set_title("월별 평균가 — 시즌성 비교 (전체 기간 평균)")
    ax.set_xlabel("월")
    ax.set_ylabel("가격 (원/kg)")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "16_variety_seasonal.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {FIGURES / '16_variety_seasonal.png'}")

    print("\n[4/4] 분석 인사이트")
    if len(summary) > 0:
        cheapest = summary.loc[summary["평균(원/kg)"].idxmin(), "품종"]
        priciest = summary.loc[summary["평균(원/kg)"].idxmax(), "품종"]
        most_volatile = summary.loc[summary["변동계수(%)"].idxmax(), "품종"]
        most_data = summary.loc[summary["일수"].idxmax(), "품종"]
        print(f"  - 가장 저렴: {cheapest} ({summary.loc[summary['품종']==cheapest, '평균(원/kg)'].values[0]:,}원/kg)")
        print(f"  - 가장 비쌈: {priciest} ({summary.loc[summary['품종']==priciest, '평균(원/kg)'].values[0]:,}원/kg)")
        print(f"  - 가장 변동성 큼: {most_volatile} (CV {summary.loc[summary['품종']==most_volatile, '변동계수(%)'].values[0]}%)")
        print(f"  - 가장 데이터 많음: {most_data} ({summary.loc[summary['품종']==most_data, '일수'].values[0]:,}일)")


if __name__ == "__main__":
    main()
