"""
08_ensemble.py
==============

Mamba + XGBoost 앙상블 — 두 모델의 예측을 결합해 단일 모델보다 안정적이고
정확한 예측을 만든다.

앙상블 종류:
    1. Simple Average: (Mamba + XGBoost) / 2
    2. Weighted Average: 검증 MAE 의 역수로 가중치 (잘하는 모델에 더 큰 비중)

기대 효과:
    - 두 모델은 서로 다른 정보(시퀀스 vs feature 간 관계)를 학습 → 다른 실수 → 평균에서 상쇄
    - 단일 Mamba 11.55% MAPE → 앙상블 8% 대 가능성

산출:
    data/processed/ensemble_metrics.csv
    figures/13_ensemble_comparison.png
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

from baselines import DEFAULT_FEATURES as ML_FEATURES, make_xgboost  # noqa: E402
from evaluation import metrics  # noqa: E402
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
RESULTS = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

WINDOW = 30
HORIZON = 1
TEST_RATIO = 0.2
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def train_mamba(Xtr, ytr, Xv, yv, n_features, scaler):
    Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
    Xv_s, yv_s = scaler.transform(Xv, yv)

    train_loader = DataLoader(WindowDataset(Xtr_s, ytr_s), batch_size=32, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xv_s, yv_s), batch_size=32)

    cfg = MambaConfig(d_model=64, d_state=16, n_layers=3, dropout=0.15)
    model = MambaForecaster(n_features=n_features, horizon=HORIZON, cfg=cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=80)
    loss_fn = torch.nn.MSELoss()

    best = float("inf"); best_state = None; bad = 0
    for ep in range(80):
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
            v = sum(loss_fn(model(Xb), yb).item() * len(Xb) for Xb, yb in val_loader) / len(val_loader.dataset)
        if v < best - 1e-6:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mamba(model, X, scaler):
    Xs = scaler.transform(X)
    test_ds = WindowDataset(Xs, np.zeros((len(X), HORIZON), dtype=np.float32))
    loader = DataLoader(test_ds, batch_size=32)
    model.eval()
    preds_s = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds_s.append(model(Xb).numpy())
    preds_s = np.concatenate(preds_s).squeeze(-1)
    return scaler.inverse_y(preds_s)


def main():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])

    # ============== 윈도우 (Mamba 입력 형태) ==============
    spec = WindowSpec(window=WINDOW, horizon=HORIZON, features=tuple(DEFAULT_FEATURES))
    X, y, dates = build_windows(df, spec)
    (Xtr_full, ytr_full, dtr_full), (Xte, yte, dte) = chronological_split(X, y, dates, TEST_RATIO)

    # train 안에서 val 분리 (앙상블 가중치 산출용)
    n_val = max(20, int(len(Xtr_full) * 0.20))
    Xtr, ytr, dtr = Xtr_full[:-n_val], ytr_full[:-n_val], dtr_full[:-n_val]
    Xv, yv, dv = Xtr_full[-n_val:], ytr_full[-n_val:], dtr_full[-n_val:]
    test_dates = pd.to_datetime(dte)
    val_dates = pd.to_datetime(dv)
    train_cutoff = pd.Timestamp(test_dates.min())
    val_cutoff = pd.Timestamp(val_dates.min())
    print(f"train: {len(Xtr)}, val: {len(Xv)}, test: {len(Xte)} 윈도우")
    print(f"test 기간: {test_dates.min().date()} ~ {test_dates.max().date()}")

    # ============== 1. Mamba 학습 ==============
    print("\n[1/4] Mamba 학습...")
    scaler = FeatureScaler().fit(Xtr, ytr)
    mamba = train_mamba(Xtr, ytr, Xv, yv, n_features=Xtr.shape[-1], scaler=scaler)

    # 검증/테스트 예측
    mamba_val = predict_mamba(mamba, Xv, scaler)
    mamba_test = predict_mamba(mamba, Xte, scaler)

    # ============== 2. XGBoost 학습 ==============
    print("\n[2/4] XGBoost 학습...")
    train_df = df[df["date"] < val_cutoff].dropna(subset=ML_FEATURES + ["price"]).copy()
    val_df = df[(df["date"] >= val_cutoff) & (df["date"] < train_cutoff)].dropna(subset=ML_FEATURES + ["price"]).copy()
    test_df = df[df["date"] >= train_cutoff].dropna(subset=ML_FEATURES + ["price"]).copy()
    xgb = make_xgboost()
    xgb.fit(train_df)
    xgb_val_pred = xgb.predict(val_df)  # date, price, pred
    xgb_test_pred = xgb.predict(test_df)

    # ============== 3. 공통 날짜로 정렬 ==============
    # Mamba 와 XGBoost 의 예측 날짜를 맞춰야 함
    print("\n[3/4] 두 모델 결과 align")
    mamba_val_df = pd.DataFrame({"date": val_dates, "price": yv.squeeze(-1), "pred_mamba": mamba_val})
    mamba_test_df = pd.DataFrame({"date": test_dates, "price": yte.squeeze(-1), "pred_mamba": mamba_test})

    val_aligned = mamba_val_df.merge(
        xgb_val_pred.rename(columns={"pred": "pred_xgb"})[["date", "pred_xgb"]],
        on="date", how="inner"
    )
    test_aligned = mamba_test_df.merge(
        xgb_test_pred.rename(columns={"pred": "pred_xgb"})[["date", "pred_xgb"]],
        on="date", how="inner"
    )
    print(f"  val aligned: {len(val_aligned)},  test aligned: {len(test_aligned)}")

    # ============== 4. 앙상블 ==============
    print("\n[4/4] 앙상블 계산")

    results = {}

    # 단일 모델 점수
    m_mamba = metrics(test_aligned["price"].values, test_aligned["pred_mamba"].values)
    m_xgb = metrics(test_aligned["price"].values, test_aligned["pred_xgb"].values)
    results["Mamba"] = {**m_mamba, "n": len(test_aligned)}
    results["XGBoost"] = {**m_xgb, "n": len(test_aligned)}

    # (a) Simple Average
    test_aligned["pred_avg"] = (test_aligned["pred_mamba"] + test_aligned["pred_xgb"]) / 2
    m_avg = metrics(test_aligned["price"].values, test_aligned["pred_avg"].values)
    results["Ensemble (Simple Avg)"] = {**m_avg, "n": len(test_aligned)}

    # (b) Weighted Average — 검증 MAE 역수로 가중치
    val_mae_mamba = np.mean(np.abs(val_aligned["price"] - val_aligned["pred_mamba"]))
    val_mae_xgb = np.mean(np.abs(val_aligned["price"] - val_aligned["pred_xgb"]))
    w_mamba = (1 / val_mae_mamba) / (1 / val_mae_mamba + 1 / val_mae_xgb)
    w_xgb = 1 - w_mamba
    print(f"  검증 MAE — Mamba: {val_mae_mamba:.0f}  XGBoost: {val_mae_xgb:.0f}")
    print(f"  가중치 — Mamba: {w_mamba:.3f}  XGBoost: {w_xgb:.3f}")

    test_aligned["pred_weighted"] = (
        w_mamba * test_aligned["pred_mamba"] + w_xgb * test_aligned["pred_xgb"]
    )
    m_w = metrics(test_aligned["price"].values, test_aligned["pred_weighted"].values)
    results[f"Ensemble (Weighted, w_mamba={w_mamba:.2f})"] = {**m_w, "n": len(test_aligned)}

    # ============== 표 ==============
    table = pd.DataFrame(results).T.sort_values("MAE")
    print("\n=== 앙상블 비교 (MAE 오름차순) ===")
    print(table.round(2).to_string())
    table.round(2).to_csv(RESULTS / "ensemble_metrics.csv", encoding="utf-8-sig")

    # ============== 시각화 ==============
    test_aligned = test_aligned.sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # (1) 4개 라인: 실제 + Mamba + XGB + Ensemble
    ax = axes[0]
    ax.plot(test_aligned["date"], test_aligned["price"], "k-", linewidth=1.6, label="actual")
    ax.plot(test_aligned["date"], test_aligned["pred_mamba"], color="C3", alpha=0.75, label=f"Mamba (MAE={m_mamba['MAE']:.0f})")
    ax.plot(test_aligned["date"], test_aligned["pred_xgb"], color="C0", alpha=0.75, label=f"XGBoost (MAE={m_xgb['MAE']:.0f})")
    ax.plot(test_aligned["date"], test_aligned["pred_avg"], color="C2", linewidth=1.6, label=f"Ensemble Simple (MAE={m_avg['MAE']:.0f})")
    ax.plot(test_aligned["date"], test_aligned["pred_weighted"], color="purple", linewidth=1.6, linestyle="--",
            label=f"Ensemble Weighted (MAE={m_w['MAE']:.0f})")
    ax.set_title("앙상블 예측 비교")
    ax.set_ylabel("가격 (원/1kg)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # (2) 잔차 (실제 - 예측) — 모델별 오차 패턴
    ax = axes[1]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(test_aligned["date"], test_aligned["price"] - test_aligned["pred_mamba"],
            color="C3", alpha=0.7, label="Mamba 잔차")
    ax.plot(test_aligned["date"], test_aligned["price"] - test_aligned["pred_xgb"],
            color="C0", alpha=0.7, label="XGBoost 잔차")
    ax.plot(test_aligned["date"], test_aligned["price"] - test_aligned["pred_avg"],
            color="C2", linewidth=1.6, label="Ensemble 잔차 (작을수록 좋음)")
    ax.set_xlabel("날짜")
    ax.set_ylabel("실제 - 예측 (원)")
    ax.set_title("잔차 비교 — Ensemble 잔차가 0에 가까워야 좋음")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "13_ensemble_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그림 저장: {FIGURES / '13_ensemble_comparison.png'}")
    print(f"표 저장: {RESULTS / 'ensemble_metrics.csv'}")

    # ============== 추가 분석: 두 모델이 얼마나 '다르게' 틀리나? ==============
    # 앙상블 효과를 결정하는 핵심 지표 = 두 모델 잔차의 상관계수
    # 상관 낮을수록 (다른 종류 실수) 앙상블 효과 큼
    res_m = test_aligned["price"] - test_aligned["pred_mamba"]
    res_x = test_aligned["price"] - test_aligned["pred_xgb"]
    corr = np.corrcoef(res_m, res_x)[0, 1]
    print(f"\n[잔차 상관] Mamba vs XGBoost = {corr:.3f}")
    print("  (1.0 = 똑같이 틀림 → 앙상블 효과 없음, 0 이하 = 반대로 틀림 → 앙상블 효과 큼)")


if __name__ == "__main__":
    main()
