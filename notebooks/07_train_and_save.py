"""
07_train_and_save.py
====================

Streamlit 데모용으로 학습한 Mamba 모델을 저장한다.

저장 내용 (models/mamba_deploy.pt):
    - state_dict
    - 모델 config (MambaConfig)
    - 입력 features 리스트
    - FeatureScaler 통계 (mean/std)
    - 타겟 정보 (target column, horizon)
    - 학습 메타데이터 (학습일, 마지막 데이터 일자, 검증 MAE 등)

사용:
    python notebooks/07_train_and_save.py

이후 app/real_predictor.py 가 이 파일을 로드해서 실시간 예측 수행.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
DEPLOY_PATH = MODELS / "mamba_deploy.pt"
META_PATH = MODELS / "mamba_deploy_meta.json"

# 배포용 하이퍼파라미터
WINDOW = 30
HORIZON = 14            # 2주일 예측 (입도선매 시나리오에 맞춤)
TARGET = "price_ffill"  # 주말 ffill 가격 — 일별 연속성 보장
BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    print(f"[1/5] 데이터 로드: {PROCESSED}")
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    print(f"  전체 일수: {len(df)}")

    print("\n[2/5] 윈도우 생성")
    spec = WindowSpec(
        window=WINDOW,
        horizon=HORIZON,
        features=tuple(DEFAULT_FEATURES),
        target=TARGET,
    )
    X, y, dates = build_windows(df, spec)
    print(f"  windows={len(X)}, X.shape={X.shape}, y.shape={y.shape}")
    print(f"  타겟 날짜: {pd.Timestamp(dates.min()).date()} ~ {pd.Timestamp(dates.max()).date()}")

    # train / val / test 분할 (시간 순서, 검증/테스트 각각 15%)
    n = len(X)
    test_start = int(n * 0.85)
    val_start = int(n * 0.70)
    Xtr, ytr, dtr = X[:val_start], y[:val_start], dates[:val_start]
    Xv, yv, dv = X[val_start:test_start], y[val_start:test_start], dates[val_start:test_start]
    Xte, yte, dte = X[test_start:], y[test_start:], dates[test_start:]
    print(f"  train: {len(Xtr)}, val: {len(Xv)}, test: {len(Xte)}")

    print("\n[3/5] 정규화 + 모델 빌드")
    scaler = FeatureScaler().fit(Xtr, ytr)
    Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
    Xv_s, yv_s = scaler.transform(Xv, yv)
    Xte_s, yte_s = scaler.transform(Xte, yte)

    train_loader = DataLoader(WindowDataset(Xtr_s, ytr_s), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xv_s, yv_s), batch_size=BATCH_SIZE)
    test_loader = DataLoader(WindowDataset(Xte_s, yte_s), batch_size=BATCH_SIZE)

    cfg = MambaConfig(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=3,
        dropout=0.15,
    )
    model = MambaForecaster(n_features=Xtr.shape[-1], horizon=HORIZON, cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    loss_fn = torch.nn.MSELoss()

    print("\n[4/5] 학습")
    best = float("inf"); best_state = None; bad = 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
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
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  val_loss={v:.4f}  best={best:.4f}")
        if bad >= PATIENCE:
            print(f"  early stop @ epoch {ep}")
            break
    print(f"  학습 시간: {time.time() - t0:.1f}s")
    model.load_state_dict(best_state)

    # ---- 검증 / 테스트 평가 ----
    model.eval()
    with torch.no_grad():
        val_pred_s = np.concatenate([model(Xb).numpy() for Xb, _ in val_loader])
        test_pred_s = np.concatenate([model(Xb).numpy() for Xb, _ in test_loader])
    val_pred = scaler.inverse_y(val_pred_s)
    val_true = scaler.inverse_y(yv_s)
    test_pred = scaler.inverse_y(test_pred_s)
    test_true = scaler.inverse_y(yte_s)

    val_m = metrics(val_true.flatten(), val_pred.flatten())
    test_m = metrics(test_true.flatten(), test_pred.flatten())
    print(f"\n  validation: MAE={val_m['MAE']:.0f}  MAPE={val_m['MAPE(%)']:.2f}%")
    print(f"  test      : MAE={test_m['MAE']:.0f}  MAPE={test_m['MAPE(%)']:.2f}%")

    # 학습된 모델의 horizon 별 잔차 std (불확실성 추정용)
    residuals = val_true - val_pred  # (n_val, horizon)
    horizon_std = residuals.std(axis=0).tolist()
    print(f"\n  horizon별 잔차 std (95% CI 폭): {[f'{1.96*s:.0f}' for s in horizon_std]}")

    print("\n[5/5] 저장")
    payload = {
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "features": list(DEFAULT_FEATURES),
        "target": TARGET,
        "window": WINDOW,
        "horizon": HORIZON,
        "scaler": {
            "feat_mean": scaler.feat_mean.tolist(),
            "feat_std": scaler.feat_std.tolist(),
            "tgt_mean": float(scaler.tgt_mean),
            "tgt_std": float(scaler.tgt_std),
        },
        "horizon_std": horizon_std,
    }
    torch.save(payload, DEPLOY_PATH)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_last_date": str(df["date"].max().date()),
        "n_train": len(Xtr),
        "n_val": len(Xv),
        "n_test": len(Xte),
        "validation_MAE": round(val_m["MAE"], 1),
        "validation_MAPE": round(val_m["MAPE(%)"], 2),
        "test_MAE": round(test_m["MAE"], 1),
        "test_MAPE": round(test_m["MAPE(%)"], 2),
        "horizon": HORIZON,
        "window": WINDOW,
        "n_params": n_params,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  모델: {DEPLOY_PATH}")
    print(f"  메타: {META_PATH}")
    print("\n완료. 이제 'streamlit run app/streamlit_app.py' 로 실제 모델을 사용한 데모 가능.")


if __name__ == "__main__":
    main()
