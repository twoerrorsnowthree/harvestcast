"""
03_mamba_train.py
=================

Mamba (SSM) 시계열 모델 학습 및 평가.

데이터 흐름:
    merged_daily.csv → sliding window (W=30, H=1)
    → 학습/검증 분할 (chronological)
    → 정규화 (train 통계 기반)
    → MambaForecaster 학습
    → test 예측 → MAE/RMSE/MAPE
    → baseline_metrics.csv 와 비교

사용:
    python notebooks/03_mamba_train.py
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
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# 하이퍼파라미터 (튜닝 후)
# CPU에서도 5~10분 정도면 끝. GPU/MPS 있으면 더 빠름.
WINDOW = 30
HORIZON = 1
BATCH_SIZE = 32
EPOCHS = 80
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0.0
    preds, ys = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        total += loss.item() * len(X)
        preds.append(pred.cpu().numpy())
        ys.append(y.cpu().numpy())
    return total / len(loader.dataset), np.concatenate(preds), np.concatenate(ys)


def main():
    print(f"DEVICE: {DEVICE}")
    print("[1/6] 데이터 로드 + 윈도우 생성")
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    spec = WindowSpec(window=WINDOW, horizon=HORIZON, features=tuple(DEFAULT_FEATURES))
    X, y, dates = build_windows(df, spec)
    print(f"  windows: {len(X)},  X.shape={X.shape},  y.shape={y.shape}")
    print(f"  타겟 날짜 범위: {pd.Timestamp(dates.min()).date()} ~ {pd.Timestamp(dates.max()).date()}")

    print("\n[2/6] train/test 분할 (chronological, test=20%)")
    (Xtr, ytr, dtr), (Xte, yte, dte) = chronological_split(X, y, dates, test_ratio=0.2)
    print(f"  train: {len(Xtr)} windows ({pd.Timestamp(dtr.min()).date()} ~ {pd.Timestamp(dtr.max()).date()})")
    print(f"  test : {len(Xte)} windows ({pd.Timestamp(dte.min()).date()} ~ {pd.Timestamp(dte.max()).date()})")

    print("\n[3/6] 정규화 (train 통계로 fit)")
    scaler = FeatureScaler().fit(Xtr, ytr)
    Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
    Xte_s, yte_s = scaler.transform(Xte, yte)

    train_ds = WindowDataset(Xtr_s, ytr_s)
    test_ds = WindowDataset(Xte_s, yte_s)
    # 학습 안에서 일부를 validation 으로 (마지막 15%)
    n_val = max(8, int(len(train_ds) * 0.15))
    train_ds_real = WindowDataset(Xtr_s[:-n_val], ytr_s[:-n_val])
    val_ds = WindowDataset(Xtr_s[-n_val:], ytr_s[-n_val:])
    print(f"  학습 안 분할: train={len(train_ds_real)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds_real, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("\n[4/6] 모델 빌드")
    cfg = MambaConfig(d_model=64, d_state=16, d_conv=4, expand=2, n_layers=3, dropout=0.15)
    model = MambaForecaster(n_features=Xtr.shape[-1], horizon=HORIZON, cfg=cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_features={Xtr.shape[-1]}, params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    print("\n[5/6] 학습")
    best_val = float("inf")
    best_state = None
    bad = 0
    history = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn)
        sched.step()
        history.append((ep, tr_loss, val_loss))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            print(f"  early stop @ epoch {ep}")
            break
    print(f"  학습 시간: {time.time() - t0:.1f}s")
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[6/6] test 예측 + 평가")
    _, pred_s, true_s = evaluate(model, test_loader, loss_fn)
    pred = scaler.inverse_y(pred_s.squeeze(-1))
    true = scaler.inverse_y(true_s.squeeze(-1))

    m = metrics(true, pred)
    print(f"  Mamba   MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  MAPE={m['MAPE(%)']:.2f}%")

    # 베이스라인 비교
    base_csv = RESULTS / "baseline_metrics.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv, index_col=0)
        # n_test 컬럼 빼고 합치기
        base = base.drop(columns=["n_test"], errors="ignore")
        combined = pd.concat([base, pd.DataFrame({"MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE(%)": m["MAPE(%)"]}, index=["Mamba"])])
        combined = combined.sort_values("MAE")
        print("\n=== 전체 비교 (MAE 오름차순) ===")
        print(combined.round(2).to_string())
        out = RESULTS / "all_metrics.csv"
        combined.round(2).to_csv(out, encoding="utf-8-sig")
        print(f"  통합 표 저장: {out}")

    # 그림: 학습 곡선
    fig, ax = plt.subplots(figsize=(8, 4))
    h = pd.DataFrame(history, columns=["epoch", "train", "val"])
    ax.plot(h["epoch"], h["train"], label="train")
    ax.plot(h["epoch"], h["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (정규화 공간)")
    ax.set_title("Mamba 학습 곡선")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_mamba_loss.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  학습 곡선 저장: {FIGURES / '06_mamba_loss.png'}")

    # 그림: 예측 vs 실제
    test_dates = pd.to_datetime(dte)
    fig, ax = plt.subplots(figsize=(11, 4))
    order = np.argsort(test_dates)
    ax.plot(test_dates.values[order], true[order], label="actual", color="black", linewidth=1.3)
    ax.plot(test_dates.values[order], pred[order], label="Mamba pred", color="C3", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Mamba 1-day-ahead 예측  (MAE={m['MAE']:.0f}, MAPE={m['MAPE(%)']:.2f}%)")
    ax.set_ylabel("가격 (원/1kg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_mamba_predictions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  예측 그림 저장: {FIGURES / '07_mamba_predictions.png'}")

    # 모델 저장
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "scaler": vars(scaler)}, MODELS / "mamba_v1.pt")
    print(f"  모델 저장: {MODELS / 'mamba_v1.pt'}")


if __name__ == "__main__":
    main()
