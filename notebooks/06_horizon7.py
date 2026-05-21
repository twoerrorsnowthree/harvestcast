"""
06_horizon7.py
==============

Horizon=7 (1주일 후 가격 예측) 비교.

입도선매 시나리오에서는 1일 예측보다 1주~1달 전 가격 예측이 더 의미있음.
Horizon=7 로 늘리면 시퀀스 모델(Mamba) 의 강점이 더 부각될 가능성.

비교 모델 (모두 7일 후 한 번에 예측):
    Naive lag-1 (1주일 후 = 오늘 가격, 가장 단순)
    Naive seasonal (작년 같은 주의 평균, 데이터 있으면)
    Ridge / XGBoost (feature 기반 direct multi-output)
    Mamba (horizon=7로 학습)

산출:
    data/processed/horizon7_metrics.csv
    figures/12_horizon7_predictions.png  (test 구간 예측 라인)
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

from baselines import DEFAULT_FEATURES as ML_FEATURES, FeatureModel  # noqa: E402
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

PROCESSED = PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"
RESULTS = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"

WINDOW = 30
HORIZON = 7
TEST_RATIO = 0.2
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def predict_naive_today(df, test_dates, h=7):
    """ '오늘 = h일 후 가격' 가정. test_dates 의 h일 전 가격을 예측값으로."""
    full = df.sort_values("date").set_index("date")
    rows = []
    for d in test_dates:
        d = pd.Timestamp(d)
        actual = full.at[d, "price"] if d in full.index else np.nan
        prev = d - pd.Timedelta(days=h)
        # 가장 가까운 prev 이전 거래일
        candidates = full.loc[: prev]
        if len(candidates) > 0:
            pred = candidates["price"].iloc[-1]
        else:
            pred = np.nan
        rows.append({"date": d, "price": actual, "pred": pred})
    return pd.DataFrame(rows).dropna(subset=["price", "pred"])


def main():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])

    # ---- 윈도우 (horizon=7) ----
    # 타겟을 price_ffill 로 두면 주말 NaN 때문에 윈도우 깨지는 일 없음.
    # 주말 가격은 금요일 가격으로 채워진 값을 사용 → 일별 예측에 적합.
    spec = WindowSpec(
        window=WINDOW,
        horizon=HORIZON,
        features=tuple(DEFAULT_FEATURES),
        target="price_ffill",
    )
    X, y, dates = build_windows(df, spec)
    print(f"Windows: {len(X)},  X.shape={X.shape},  y.shape={y.shape} (horizon=7)")

    (Xtr, ytr, dtr), (Xte, yte, dte) = chronological_split(X, y, dates, TEST_RATIO)
    test_start_dates = pd.to_datetime(dte)  # 각 윈도우의 첫째 예측일
    train_cutoff = pd.Timestamp(test_start_dates.min())
    print(f"train: {len(Xtr)}, test: {len(Xte)} 윈도우")
    print(f"공통 test 기간: {test_start_dates.min().date()} ~ {test_start_dates.max().date()}")

    results = {}

    # ============== Naive: 1주일 전 가격 ==============
    naive = predict_naive_today(df, test_start_dates, h=7)
    if len(naive) > 0:
        m_ = metrics(naive["price"].values, naive["pred"].values)
        results["Naive 1wk-ago"] = {**m_, "n": len(naive)}
        print(f"Naive 1wk-ago     n={len(naive):3d}  MAE={m_['MAE']:.0f}  MAPE={m_['MAPE(%)']:.2f}%")

    # ============== Ridge / XGBoost (multi-output direct) ==============
    # feature 기반 모델은 horizon=7 을 출력으로 내야 함 → MultiOutput 처리.
    # 간단히 각 horizon-step을 별도 모델로 학습해서 합치는 방식 사용.
    train_df = df[df["date"] < train_cutoff].dropna(subset=ML_FEATURES + ["price"]).copy()
    if len(train_df) > 60:
        # 각 horizon h 대해 target = price.shift(-h) 로 학습
        train_df = train_df.sort_values("date").reset_index(drop=True)

        # h=7 한 step만 평가 (나머지는 horizon=1~6은 sliding 안 하고 최종 7일째만)
        # multi-step direct 의 단순화: 가장 먼 horizon (7일 후) 만 평가
        train_df["target_h7"] = train_df["price"].shift(-7)
        sub_tr = train_df.dropna(subset=["target_h7"]).copy()
        Xml_tr = sub_tr[ML_FEATURES].values
        yml_tr = sub_tr["target_h7"].values

        # test: 각 윈도우 시작일 + 6 (= 마지막 예측일) 의 실제 가격을 타겟으로
        for name, model in [("Ridge", Ridge(alpha=1.0)),
                            ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]:
            try:
                model.fit(Xml_tr, yml_tr)

                # test: window 시작일에서 features 사용 → 7일 후 예측
                test_pairs = []
                for d in test_start_dates:
                    feat_row = df[(df["date"] == pd.Timestamp(d)) & df[ML_FEATURES].notna().all(axis=1)]
                    actual_d = pd.Timestamp(d) + pd.Timedelta(days=6)  # 마지막 horizon
                    actual_row = df[df["date"] == actual_d]
                    if len(feat_row) == 0 or len(actual_row) == 0:
                        continue
                    if pd.isna(actual_row["price"].iloc[0]):
                        continue
                    pred_v = model.predict(feat_row[ML_FEATURES].values)[0]
                    test_pairs.append({"date": actual_d, "price": float(actual_row["price"].iloc[0]), "pred": float(pred_v)})
                if test_pairs:
                    p = pd.DataFrame(test_pairs)
                    m_ = metrics(p["price"].values, p["pred"].values)
                    results[name] = {**m_, "n": len(p)}
                    print(f"{name:15s}  n={len(p):3d}  MAE={m_['MAE']:.0f}  MAPE={m_['MAPE(%)']:.2f}%")
            except Exception as e:
                print(f"  {name} 실패: {e}")

    # ============== Mamba (horizon=7) ==============
    print("\nMamba 학습 (horizon=7)...")
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
    EPOCHS_M = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS_M)
    loss_fn = torch.nn.MSELoss()

    best = float("inf"); best_state = None; bad = 0
    for ep in range(1, EPOCHS_M + 1):
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
        if bad >= 20:
            print(f"  early stop @ epoch {ep}")
            break
    model.load_state_dict(best_state)

    model.eval()
    preds_s, ys_s = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds_s.append(model(Xb).numpy())
            ys_s.append(yb.numpy())
    preds_s = np.concatenate(preds_s)  # (N, 7)
    ys_s = np.concatenate(ys_s)
    pred_all = scaler.inverse_y(preds_s)
    true_all = scaler.inverse_y(ys_s)

    # 7일째 (마지막 horizon) MAE
    m_last = metrics(true_all[:, -1], pred_all[:, -1])
    results["Mamba (h=7th day)"] = {**m_last, "n": len(true_all)}
    print(f"Mamba (h=7th day)  n={len(true_all):3d}  MAE={m_last['MAE']:.0f}  MAPE={m_last['MAPE(%)']:.2f}%")

    # 평균 (1~7일 모든 horizon 통합)
    m_avg = metrics(true_all.flatten(), pred_all.flatten())
    results["Mamba (h=1..7 avg)"] = {**m_avg, "n": len(true_all) * HORIZON}
    print(f"Mamba (h=1..7 avg) n={len(true_all)*HORIZON:3d}  MAE={m_avg['MAE']:.0f}  MAPE={m_avg['MAPE(%)']:.2f}%")

    # ============== 표 ==============
    table = pd.DataFrame(results).T.sort_values("MAE")
    print("\n=== Horizon=7 비교 (MAE 오름차순) ===")
    print(table.round(2).to_string())
    table.round(2).to_csv(RESULTS / "horizon7_metrics.csv", encoding="utf-8-sig")

    # ============== 시각화: Mamba 7-step 예측 ==============
    # 첫 5개 윈도우만 시각화
    n_show = min(5, len(true_all))
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.0 * n_show), sharex=False)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        sd = test_start_dates[i]
        days = pd.date_range(sd, periods=HORIZON)
        ax.plot(days, true_all[i], "o-", color="black", label="actual", markersize=4)
        ax.plot(days, pred_all[i], "s--", color="C3", label="Mamba pred", markersize=4)
        ax.set_title(f"7일 예측 — 시작일 {pd.Timestamp(sd).date()}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "12_horizon7_predictions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그림 저장: {FIGURES / '12_horizon7_predictions.png'}")
    print(f"표 저장: {RESULTS / 'horizon7_metrics.csv'}")


if __name__ == "__main__":
    main()
