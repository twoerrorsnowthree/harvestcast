"""
05_walk_forward.py
==================

Walk-forward validation — 단일 train/test split이 아니라
여러 시점(fold) 에서 모델을 반복 학습/평가해 안정성 검증.

방식:
    initial train = 데이터의 처음 60%
    각 fold: 그 시점까지로 학습 → 다음 H일 test → fold 결과 저장
    매 step마다 30일씩 슬라이드 (fold 간격)

비교 모델:
    Naive lag-1 (참조), Ridge, XGBoost, Mamba

산출:
    data/processed/walk_forward.csv  (fold별 MAE/MAPE)
    figures/10_walk_forward_mae.png
    figures/11_walk_forward_distribution.png  (모델별 MAE 분포)
"""
import sys
import time
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

from baselines import DEFAULT_FEATURES as ML_FEATURES, make_ridge, make_xgboost  # noqa: E402
from evaluation import metrics  # noqa: E402
from mamba_model import MambaConfig, MambaForecaster  # noqa: E402
from timeseries_dataset import (  # noqa: E402
    DEFAULT_FEATURES,
    FeatureScaler,
    WindowDataset,
    WindowSpec,
    build_windows,
)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"
RESULTS = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# 설정
WINDOW = 30
HORIZON = 1
INITIAL_TRAIN_RATIO = 0.6   # 첫 fold의 train 비율
STEP = 30                    # fold 간격 (일)
TEST_LEN = 30                # 각 fold의 test 길이
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Mamba 학습 설정 (가벼운 버전 — fold가 많아서)
MAMBA_EPOCHS = 30


def train_mamba_quick(Xtr, ytr, scaler, n_features):
    """Walk-forward 한 fold 안에서 빠르게 학습."""
    Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
    n_val = max(8, int(len(Xtr_s) * 0.15))
    if len(Xtr_s) > 30 + n_val:
        train_X, train_y = Xtr_s[:-n_val], ytr_s[:-n_val]
        val_X, val_y = Xtr_s[-n_val:], ytr_s[-n_val:]
    else:
        train_X, train_y = Xtr_s, ytr_s
        val_X, val_y = Xtr_s[-8:], ytr_s[-8:]

    train_ds = WindowDataset(train_X, train_y)
    val_ds = WindowDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    cfg = MambaConfig(d_model=32, d_state=8, n_layers=2, dropout=0.1)
    model = MambaForecaster(n_features=n_features, horizon=HORIZON, cfg=cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAMBA_EPOCHS)
    loss_fn = torch.nn.MSELoss()

    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAMBA_EPOCHS):
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
            bad = 0
        else:
            bad += 1
        if bad >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mamba(model, Xte, scaler):
    Xte_s = scaler.transform(Xte)
    test_ds = WindowDataset(Xte_s, np.zeros((len(Xte), HORIZON), dtype=np.float32))
    loader = DataLoader(test_ds, batch_size=32)
    model.eval()
    preds_s = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds_s.append(model(Xb).numpy())
    preds_s = np.concatenate(preds_s).squeeze(-1)
    return scaler.inverse_y(preds_s)


def predict_ml_at_dates(df_full, model_factory, train_cutoff, test_dates):
    """ML 모델: train_cutoff 이전으로 학습 → test_dates 예측."""
    train_df = df_full[df_full["date"] < train_cutoff].dropna(subset=ML_FEATURES + ["price"])
    test_df = df_full[df_full["date"].isin(test_dates)].dropna(subset=ML_FEATURES + ["price"])
    if len(train_df) < 30 or len(test_df) == 0:
        return None
    m = model_factory()
    m.fit(train_df)
    pred_df = m.predict(test_df)
    return pred_df  # date, price, pred


def predict_naive_lag1(df_full, test_dates):
    """전체 시계열에서 lag-1 예측."""
    full = df_full.sort_values("date").copy()
    full["pred"] = full["price"].shift(1)
    sub = full[full["date"].isin(test_dates)][["date", "price", "pred"]]
    return sub


def main():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    print(f"전체 일수: {len(df)}, 가격 있는 일: {df['price'].notna().sum()}")

    # ---- Mamba 윈도우 만들기 (전체) ----
    spec = WindowSpec(window=WINDOW, horizon=HORIZON, features=tuple(DEFAULT_FEATURES))
    X_all, y_all, dates_all = build_windows(df, spec)
    order = np.argsort(dates_all)
    X_all, y_all, dates_all = X_all[order], y_all[order], dates_all[order]
    print(f"Mamba 윈도우: {len(X_all)}")

    n_total = len(X_all)
    initial = int(n_total * INITIAL_TRAIN_RATIO)
    print(f"initial train: {initial}, step: {STEP}, test_len: {TEST_LEN}")

    fold_results = []
    fold_idx = 0
    cur = initial
    t0 = time.time()
    while cur + TEST_LEN <= n_total:
        fold_idx += 1
        Xtr, ytr = X_all[:cur], y_all[:cur]
        Xte, yte = X_all[cur : cur + TEST_LEN], y_all[cur : cur + TEST_LEN]
        test_dates = pd.to_datetime(dates_all[cur : cur + TEST_LEN])
        train_cutoff = pd.Timestamp(test_dates.min())

        # ---- Mamba ----
        scaler = FeatureScaler().fit(Xtr, ytr)
        try:
            mamba_model = train_mamba_quick(Xtr, ytr, scaler, n_features=X_all.shape[-1])
            mamba_pred = predict_mamba(mamba_model, Xte, scaler)
            mamba_true = yte.squeeze(-1)
            m_mamba = metrics(mamba_true, mamba_pred)
            fold_results.append({"fold": fold_idx, "model": "Mamba", "test_start": test_dates.min().date(), **m_mamba})
        except Exception as e:
            print(f"  fold {fold_idx} Mamba 실패: {e}")

        # ---- ML 모델 (Ridge, XGBoost) ----
        for name, factory in [("Ridge", make_ridge), ("XGBoost", make_xgboost)]:
            try:
                p = predict_ml_at_dates(df, factory, train_cutoff, test_dates)
                if p is None or len(p) == 0:
                    continue
                m_ = metrics(p["price"].values, p["pred"].values)
                fold_results.append({"fold": fold_idx, "model": name, "test_start": test_dates.min().date(), **m_})
            except Exception as e:
                print(f"  fold {fold_idx} {name} 실패: {e}")

        # ---- Naive lag-1 ----
        p = predict_naive_lag1(df, test_dates).dropna(subset=["price", "pred"])
        if len(p) > 0:
            m_ = metrics(p["price"].values, p["pred"].values)
            fold_results.append({"fold": fold_idx, "model": "Naive lag-1", "test_start": test_dates.min().date(), **m_})

        elapsed = time.time() - t0
        print(f"  fold {fold_idx:2d}  test={test_dates.min().date()}~{test_dates.max().date()}  ({elapsed:.0f}s 누적)")
        cur += STEP

    res = pd.DataFrame(fold_results)
    print(f"\n총 {fold_idx} fold, {len(res)} 결과 행")

    # 모델별 평균 MAE
    print("\n=== 모델별 평균 ===")
    summary = res.groupby("model").agg(
        n_folds=("fold", "count"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        MAPE_mean=("MAPE(%)", "mean"),
        MAPE_std=("MAPE(%)", "std"),
    ).round(2)
    print(summary.to_string())

    res.to_csv(RESULTS / "walk_forward.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "walk_forward_summary.csv", encoding="utf-8-sig")

    # 시각화 1: fold별 MAE 라인
    fig, ax = plt.subplots(figsize=(11, 5))
    for model in res["model"].unique():
        sub = res[res["model"] == model].sort_values("fold")
        ax.plot(sub["fold"], sub["MAE"], marker="o", label=model, alpha=0.85)
    ax.set_xlabel("fold")
    ax.set_ylabel("MAE (원)")
    ax.set_title("Walk-forward validation: fold별 MAE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "10_walk_forward_mae.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 시각화 2: 모델별 MAE 분포 (boxplot)
    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted(res["model"].unique())
    data = [res[res["model"] == m]["MAE"].values for m in models]
    ax.boxplot(data, tick_labels=models)
    ax.set_ylabel("MAE (원)")
    ax.set_title("Walk-forward validation: 모델별 MAE 분포")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_walk_forward_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n저장: {RESULTS / 'walk_forward.csv'}")
    print(f"      {RESULTS / 'walk_forward_summary.csv'}")
    print(f"      {FIGURES / '10_walk_forward_mae.png'}")
    print(f"      {FIGURES / '11_walk_forward_distribution.png'}")


if __name__ == "__main__":
    main()
