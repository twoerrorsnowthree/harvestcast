"""
02_baselines.py
===============

베이스라인 모델 5종을 같은 train/test split에 학습/평가하고 비교한다.

모델:
    1. Naive lag-1 (어제 가격 그대로)
    2. Naive lag-7 (1주일 전 가격 그대로)
    3. Ridge (선형 회귀, L2)
    4. RandomForest
    5. XGBoost

데이터: data/processed/merged_daily.csv (전체 일별 머지 결과)
타겟: 'price' (가락시장 1kg 상등급 도매가)

사용:
    python notebooks/02_baselines.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for fn in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    plt.rcParams["font.family"] = fn
    break
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import (  # noqa: E402
    compare_results,
    metrics,
    train_test_split_by_date,
)
from baselines import run_feature_models, run_naive_predictions  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "data" / "processed"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    return df


def select_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델링 대상: 가격이 존재하는 일만.
    출하기/비출하기 둘 다 포함하지만, 가격 결측은 제외.
    """
    return df.dropna(subset=["price"]).reset_index(drop=True)


def main():
    print("[1/5] 데이터 로드")
    df_all = load_data()
    df = select_modeling_data(df_all)
    print(f"  전체 일수: {len(df_all)}, 가격 있는 일: {len(df)}")
    print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("\n[2/5] train/test 분할 (시간 순서, 마지막 20%가 test)")
    split = train_test_split_by_date(df, test_ratio=0.2)
    print(f"  train: {len(split.train)}일 ({split.train['date'].min().date()} ~ {split.train['date'].max().date()})")
    print(f"  test : {len(split.test)}일 ({split.test['date'].min().date()} ~ {split.test['date'].max().date()})")

    test_dates = split.test["date"]

    print("\n[3/5] Naive baseline 예측")
    # naive는 train/test 합친 전체 시계열에서 shift 해야 자연스러움
    full_for_naive = df.copy()
    naive_preds = run_naive_predictions(full_for_naive, test_dates)

    print("\n[4/5] ML baseline 학습 + 예측")
    ml_preds = run_feature_models(split.train, split.test)

    all_preds = {**naive_preds, **ml_preds}

    print("\n[5/5] 평가 + 시각화")
    results = {}
    for name, pred_df in all_preds.items():
        valid = pred_df.dropna(subset=["price", "pred"])
        if len(valid) == 0:
            print(f"  [skip] {name}: 유효 예측 0개")
            continue
        m = metrics(valid["price"].values, valid["pred"].values)
        m["n_test"] = len(valid)
        results[name] = m
        print(f"  {name:25s}  n={len(valid):3d}  MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  MAPE={m['MAPE(%)']:.1f}%")

    # 결과 표 저장
    table = compare_results(results).sort_values("MAE")
    print("\n=== 결과 비교 (MAE 오름차순) ===")
    print(table.to_string())
    out_csv = RESULTS / "baseline_metrics.csv"
    table.to_csv(out_csv, encoding="utf-8-sig")
    print(f"  표 저장: {out_csv}")

    # 평가 가능했던(results 있는) 모델만 시각화
    plotted = [(n, p) for n, p in all_preds.items() if n in results]
    fig, axes = plt.subplots(len(plotted), 1, figsize=(12, 2.2 * len(plotted)), sharex=True)
    if len(plotted) == 1:
        axes = [axes]
    for ax, (name, pred_df) in zip(axes, plotted):
        valid = pred_df.dropna(subset=["price", "pred"]).sort_values("date")
        ax.plot(valid["date"], valid["price"], label="actual", color="black", linewidth=1.2)
        ax.plot(valid["date"], valid["pred"], label="pred", color="C3", linewidth=1.0, alpha=0.85)
        ax.set_title(f"{name}  (MAE={results[name]['MAE']:.0f}, MAPE={results[name]['MAPE(%)']:.1f}%)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("date")
    fig.suptitle("Baseline: actual vs predicted (test 구간)", y=1.0)
    fig.tight_layout()
    fig_path = FIGURES / "04_baseline_predictions.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  그림 저장: {fig_path}")

    # 메트릭 막대 그래프
    fig, ax = plt.subplots(figsize=(8, 4))
    table_sorted = table.sort_values("MAE")
    x = np.arange(len(table_sorted))
    ax.bar(x, table_sorted["MAE"].values, color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(table_sorted.index, rotation=20, ha="right")
    ax.set_ylabel("MAE (원)")
    ax.set_title("Baseline 모델별 MAE 비교")
    for i, v in enumerate(table_sorted["MAE"].values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path2 = FIGURES / "05_baseline_mae_bar.png"
    fig.savefig(fig_path2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  그림 저장: {fig_path2}")


if __name__ == "__main__":
    main()
