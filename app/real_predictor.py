"""
실제 학습된 Mamba 모델을 사용한 예측 + Bootstrap 신뢰구간.

mock_predictor.py 와 인터페이스(predict 함수 시그니처) 가 동일해서
streamlit_app.py 에서 그대로 갈아끼울 수 있다.

핵심:
    - models/mamba_deploy.pt 로드
    - merged_daily.csv 의 가장 최근 30일 윈도우로 예측
    - 신뢰구간: horizon 별 검증 잔차 std + MC Dropout 결합
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mamba_model import MambaConfig, MambaForecaster  # noqa: E402

DEPLOY_PATH = PROJECT_ROOT / "models" / "mamba_deploy.pt"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"


# ---------- 로드 (singleton 패턴: 한 번만 로드) ----------

_LOADED = {"model": None, "payload": None}


def _ensure_loaded():
    if _LOADED["model"] is not None:
        return _LOADED["model"], _LOADED["payload"]
    if not DEPLOY_PATH.exists():
        raise FileNotFoundError(
            f"학습된 모델 파일이 없습니다: {DEPLOY_PATH}\n"
            f"먼저 'python notebooks/07_train_and_save.py' 를 실행해 모델을 만들어 주세요."
        )
    payload = torch.load(DEPLOY_PATH, map_location="cpu", weights_only=False)
    cfg = MambaConfig(**payload["cfg"])
    model = MambaForecaster(
        n_features=len(payload["features"]),
        horizon=payload["horizon"],
        cfg=cfg,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _LOADED["model"], _LOADED["payload"] = model, payload
    return model, payload


# ---------- 데이터 윈도우 ----------

def _get_recent_window(start_date: date, payload: dict) -> np.ndarray:
    """start_date 직전 window일치의 features 추출."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feats = list(payload["features"])
    window = payload["window"]

    cutoff = pd.Timestamp(start_date)
    sub = df[df["date"] < cutoff].dropna(subset=feats).sort_values("date").tail(window)
    if len(sub) < window:
        raise RuntimeError(
            f"start_date {start_date} 직전에 사용 가능한 feature 일수가 {len(sub)}개로 "
            f"필요한 {window}일에 못 미칩니다. 더 이른 날짜를 선택해 주세요."
        )
    X = sub[feats].values.astype(np.float32)
    return X  # (window, n_features)


def _scale_features(X: np.ndarray, payload: dict) -> np.ndarray:
    mean = np.array(payload["scaler"]["feat_mean"], dtype=np.float32)
    std = np.array(payload["scaler"]["feat_std"], dtype=np.float32)
    return (X - mean) / std


def _inverse_target(y_s: np.ndarray, payload: dict) -> np.ndarray:
    mean = float(payload["scaler"]["tgt_mean"])
    std = float(payload["scaler"]["tgt_std"])
    return y_s * std + mean


# ---------- 예측 ----------

def _mc_dropout_predict(model, X_t: torch.Tensor, n_samples: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout 으로 평균/표준편차 추정.
    학습용 dropout 을 inference 에 활성화.
    """
    model.train()  # dropout 활성화
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_t).numpy())
    model.eval()
    arr = np.stack(preds, axis=0)  # (n_samples, 1, horizon)
    mean = arr.mean(axis=0).squeeze(0)  # (horizon,)
    std = arr.std(axis=0).squeeze(0)
    return mean, std


def predict(
    start_date: date,
    horizon_days: int,
    last_price: float | None = None,   # 인터페이스 호환용 — 실제 모델은 사용 안 함
    seed: int | None = None,
) -> pd.DataFrame:
    """
    실제 학습된 Mamba 로 가격 예측.

    Args:
        start_date: 예측 시작일 (포함)
        horizon_days: 예측 기간 (일). 모델 학습 horizon(14) 보다 크면 제한됨.

    Returns:
        DataFrame with columns: date, predicted_price, lower, upper
    """
    if seed is not None:
        torch.manual_seed(seed)

    model, payload = _ensure_loaded()
    horizon_max = payload["horizon"]
    horizon_used = min(horizon_days, horizon_max)

    # 입력 윈도우
    X = _get_recent_window(start_date, payload)
    X_s = _scale_features(X, payload)
    X_t = torch.from_numpy(X_s[None, :, :]).float()  # (1, window, F)

    # MC Dropout 예측
    mean_s, mc_std_s = _mc_dropout_predict(model, X_t, n_samples=50)
    mean = _inverse_target(mean_s, payload)
    mc_std = mc_std_s * payload["scaler"]["tgt_std"]

    # horizon 별 검증 잔차 std (이게 우세, MC dropout 은 추가 보정)
    horizon_std = np.array(payload.get("horizon_std", [400] * horizon_max), dtype=np.float32)
    # 두 불확실성 결합 (제곱합 루트)
    total_std = np.sqrt(horizon_std[:horizon_used] ** 2 + mc_std[:horizon_used] ** 2)

    # 95% 신뢰구간
    lower = mean[:horizon_used] - 1.96 * total_std
    upper = mean[:horizon_used] + 1.96 * total_std

    dates = [start_date + timedelta(days=i) for i in range(horizon_used)]
    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "predicted_price": mean[:horizon_used],
        "lower": np.maximum(lower, 0),  # 가격은 음수 안 됨
        "upper": upper,
    })

    # 요청 horizon 이 더 길면 안내 행으로 채움 (선형 외삽)
    if horizon_days > horizon_max:
        extra_days = horizon_days - horizon_max
        last = out.iloc[-1]
        extra_dates = [out["date"].iloc[-1] + pd.Timedelta(days=i + 1) for i in range(extra_days)]
        # 예측은 마지막값 유지, 신뢰구간은 폭 1.5배로 점차 확대
        widen = np.linspace(1.5, 2.5, extra_days)
        ext = pd.DataFrame({
            "date": extra_dates,
            "predicted_price": [last["predicted_price"]] * extra_days,
            "lower": np.maximum(last["lower"] - widen * (last["predicted_price"] - last["lower"]), 0),
            "upper": last["upper"] + widen * (last["upper"] - last["predicted_price"]),
        })
        out = pd.concat([out, ext], ignore_index=True)

    return out


def load_historical(merged_csv_path: str) -> pd.DataFrame:
    """mock_predictor.load_historical 과 동일 시그니처."""
    df = pd.read_csv(merged_csv_path, parse_dates=["date"])
    df = df.dropna(subset=["price"])[["date", "price"]].sort_values("date").reset_index(drop=True)
    return df


def get_model_info() -> dict | None:
    """배포 메타 정보 (학습일, 검증 점수 등) — Streamlit 표시용."""
    meta_path = PROJECT_ROOT / "models" / "mamba_deploy_meta.json"
    if not meta_path.exists():
        return None
    import json
    return json.loads(meta_path.read_text())


if __name__ == "__main__":
    # 빠른 테스트
    from datetime import date as d
    today = d.today()
    df = predict(today, 14, seed=42)
    print(df.to_string())
    info = get_model_info()
    if info:
        print("\n메타:", info)
