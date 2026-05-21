"""
04_fair_comparison.py
=====================

Mamba와 베이스라인을 **같은 test 기간**으로 평가해 공정 비교.

이전 02_baselines.py는 merged_daily.csv 의 마지막 20% 를 test로 썼고,
03_mamba_train.py 는 윈도우 245개 중 마지막 20% (날짜로는 2022-12-31~2023-03-01) 을 test로 썼음.
서로 다른 기간이라 직접 비교가 불가. 이 스크립트는 동일한 test 날짜에 대해 모든 모델을 평가한다.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baselines import run_feature_models, run_naive_predictions  # noqa: E402
from evaluation import metrics, compare_results  # noqa: E402
from mamba_model import MambaConfig, MambaForecaster  # noqa: E402
from timeseries_dataset import (  # noqa: E402
    DEFAULT_FEATURES,
    FeatureScaler,
    WindowDataset,
    WindowSpec,
    build_windows,
    chronological_split,
)


PROCESSED = PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

WINDOW = 30
HORIZON = 1
TEST_RATIO = 0.2
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])

    # === Mamba용 윈도우 + split (이게 기준 test 기간) ===
    spec = WindowSpec(window=WINDOW, horizon=HORIZON, features=tuple(DEFAULT_FEATURES))
    X, y, dates = build_windows(df, spec)
    (Xtr, ytr, dtr), (Xte, yte, dte) = chronological_split(X, y, dates, TEST_RATIO)
    test_dates = pd.to_datetime(dte)
    train_cutoff = pd.Timestamp(test_dates.min())
    print(f"공통 test 기간: {test_dates.min().date()} ~ {test_dates.max().date()}  ({len(test_dates)}일)")
    print(f"공통 train 끝: {train_cutoff.date()} 이전")

    results = {}
    pred_dfs = {}

    # ============== 1. 베이스라인 (같은 cutoff 사용) ==============
    df_base = df.dropna(subset=["price"]).reset_index(drop=True)
    train_df = df_base[df_base["date"] < train_cutoff].copy()
    test_df = df_base[df_base["date"].isin(test_dates)].copy()
    print(f"\n[Baseline] train={len(train_df)}일, test={len(test_df)}일")

    naive = run_naive_predictions(df_base, test_df["date"])
    ml = run_feature_models(train_df, test_df)
    for name, pred_df in {**naive, **ml}.items():
        valid = pred_df.dropna(subset=["price", "pred"])
        if len(valid) == 0:
            continue
        m = metrics(valid["price"].values, valid["pred"].values)
        results[name] = {"MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE(%)": m["MAPE(%)"], "n": len(valid)}
        pred_dfs[name] = valid
        print(f"  {name:18s}  n={len(valid):3d}  MAE={m['MAE']:.0f}  MAPE={m['MAPE(%)']:.2f}%")

    # ============== 2. Mamba ==============
    print(f"\n[Mamba] 학습 시작...")
    scaler = FeatureScaler().fit(Xtr, ytr)
    Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
    Xte_s, yte_s = scaler.transform(Xte, yte)
    n_val = max(8, int(len(Xtr_s) * 0.15))
    train_ds = WindowDataset(Xtr_s[:-n_val], ytr_s[:-n_val])
    val_ds = WindowDataset(Xtr_s[-n_val:], ytr_s[-n_val:])
    test_ds = WindowDataset(Xte_s, yte_s)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    cfg = MambaConfig(d_model=64, d_state=16, d_conv=4, expand=2, n_layers=3, dropout=0.15)
    model = MambaForecaster(n_features=Xtr.shape[-1], horizon=HORIZON, cfg=cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    EPOCHS_FAIR = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS_FAIR)
    loss_fn = torch.nn.MSELoss()

    best = float("inf"); best_state = None
    bad_count = 0
    for ep in range(1, EPOCHS_FAIR + 1):
        model.train()
        for Xb, yb in train_loader:
            optim.zero_grad()
            l = loss_fn(model(Xb), yb)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = sum(loss_fn(model(Xb), yb).item() * len(Xb) for Xb, yb in val_loader) / len(val_ds)
        if v < best - 1e-6:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            bad_count = 0
        else:
            bad_count += 1
        if bad_count >= 20:
            print(f"  early stop @ epoch {ep}")
            break
    model.load_state_dict(best_state)

    model.eval()
    preds_s, ys_s = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds_s.append(model(Xb).numpy())
            ys_s.append(yb.numpy())
    preds_s = np.concatenate(preds_s).squeeze(-1)
    ys_s = np.concatenate(ys_s).squeeze(-1)
    pred = scaler.inverse_y(preds_s)
    true = scaler.inverse_y(ys_s)

    m = metrics(true, pred)
    results["Mamba"] = {"MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE(%)": m["MAPE(%)"], "n": len(true)}
    pred_dfs["Mamba"] = pd.DataFrame({"date": test_dates, "price": true, "pred": pred})
    print(f"  Mamba             n={len(true):3d}  MAE={m['MAE']:.0f}  MAPE={m['MAPE(%)']:.2f}%")

    # ============== 3. 비교 표 ==============
    table = pd.DataFrame(results).T.sort_values("MAE")
    print("\n=== 공정 비교 (같은 test 기간) ===")
    print(table.round(2).to_string())
    out = RESULTS / "fair_comparison.csv"
    table.round(2).to_csv(out, encoding="utf-8-sig")
    print(f"  저장: {out}")

    # ============== 4. 시각화 ==============
    fig, ax = plt.subplots(figsize=(8, 4))
    table_sorted = table.sort_values("MAE")
    x = np.arange(len(table_sorted))
    colors = ["C3" if name == "Mamba" else "C0" for name in table_sorted.index]
    ax.bar(x, table_sorted["MAE"].values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(table_sorted.index, rotation=20, ha="right")
    ax.set_ylabel("MAE (원)")
    ax.set_title(f"공정 비교 - test {test_dates.min().date()}~{test_dates.max().date()}")
    for i, v in enumerate(table_sorted["MAE"].values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_fair_mae.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 모델별 예측 라인 (top 3 + Mamba)
    show = list(table.head(3).index)
    if "Mamba" not in show:
        show.append("Mamba")
    fig, ax = plt.subplots(figsize=(12, 5))
    actual = pred_dfs[show[0]].sort_values("date")
    ax.plot(actual["date"], actual["price"], label="actual", color="black", linewidth=1.4)
    for name in show:
        d = pred_dfs[name].sort_values("date")
        ax.plot(d["date"], d["pred"], label=f"{name} (MAE={results[name]['MAE']:.0f})", linewidth=1.0, alpha=0.85)
    ax.set_title(f"공정 비교 - 예측 vs 실제 ({test_dates.min().date()} ~ {test_dates.max().date()})")
    ax.set_ylabel("가격 (원/1kg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_fair_predictions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  그림 저장: 08_fair_mae.png, 09_fair_predictions.png")


if __name__ == "__main__":
    main()
