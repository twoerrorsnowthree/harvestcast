"""
KAMIS 도매가격 xls 파서

KAMIS 웹에서 다운로드한 '가격정보 *.xls' 파일은 사실 HTML 테이블이다.
첫 번째 테이블에 일자별 도매가격(품목/시장/단위/등급/평균가)이 들어있다.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_price_xls(xls_path: str | Path) -> pd.DataFrame:
    """
    KAMIS 'xls'(실제로는 html) 파일 하나를 읽어 dataframe으로 반환.
    유효한 일자 행만 남기고, 평균가는 숫자형으로 변환.
    """
    tables = pd.read_html(xls_path, encoding="utf-8")
    df = tables[0]

    # 헤더/안내 행 제거: '일자'가 datetime으로 파싱되는 행만 유지
    parsed = pd.to_datetime(df["일자"], format="%Y.%m.%d", errors="coerce")
    df = df[parsed.notna()].copy()
    df["date"] = parsed[parsed.notna()].values
    df["price"] = pd.to_numeric(df["평균가"], errors="coerce")

    df = df.rename(columns={"시장": "market", "품목": "item", "단위": "unit", "등급": "grade"})
    return df[["date", "market", "item", "unit", "grade", "price"]].reset_index(drop=True)


def load_price_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    KAMIS API로 받아 저장된 csv 형식 (kamis_api*.csv).
    필수 컬럼: date, price.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])[["date", "price"]]
    return df.sort_values("date").reset_index(drop=True)


def load_all_prices(
    price_dir: str | Path,
    item: str = "딸기 설향",
    market: str = "가락시장",
    unit: str = "1키로상자",
    grade: str = "상",
) -> pd.DataFrame:
    """
    price_dir 안의 모든 가격 파일을 읽어 합친 일별 가격 시리즈를 반환.

    지원 포맷:
        - .xls / .xlsx : KAMIS 웹 다운로드 (HTML 테이블)
                          → 품목/시장/단위/등급으로 필터링됨
        - .csv         : KAMIS API 결과 (이미 단일 시리즈, 그대로 사용)
                          → kamis_api* 또는 date+price 컬럼이면 OK
    """
    price_dir = Path(price_dir)
    xls_files = sorted(price_dir.glob("*.xls")) + sorted(price_dir.glob("*.xlsx"))
    csv_files = sorted(price_dir.glob("*.csv"))
    if not xls_files and not csv_files:
        raise FileNotFoundError(f"가격 파일이 없습니다: {price_dir}")

    series_frames = []  # 이미 (date, price) 형태인 csv 결과들
    raw_frames = []     # 필터링 필요한 xls 결과들

    for f in xls_files:
        try:
            raw_frames.append(load_price_xls(f))
        except Exception as e:
            print(f"[skip xls] {f.name}: {e}")

    for f in csv_files:
        try:
            series_frames.append(load_price_csv(f))
            print(f"[csv] {f.name} → 그대로 사용")
        except Exception as e:
            print(f"[skip csv] {f.name}: {e}")

    # xls는 필터링
    filtered = pd.DataFrame(columns=["date", "price"])
    if raw_frames:
        all_xls = pd.concat(raw_frames, ignore_index=True)
        mask = (
            (all_xls["item"] == item)
            & (all_xls["market"] == market)
            & (all_xls["unit"] == unit)
            & (all_xls["grade"] == grade)
        )
        filtered = all_xls[mask][["date", "price"]].copy()

    # csv는 그대로
    csv_part = pd.concat(series_frames, ignore_index=True) if series_frames else pd.DataFrame(columns=["date", "price"])

    out = pd.concat([filtered, csv_part], ignore_index=True)
    if out.empty:
        raise RuntimeError("유효한 KAMIS 가격 데이터를 찾지 못했습니다.")

    # 같은 날짜 중복 시 평균 (xls와 csv가 겹치는 기간이 있으면 평균)
    out = out.groupby("date", as_index=False)["price"].mean()
    out = out.sort_values("date").reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys

    price_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/price"
    df = load_all_prices(price_dir)
    print(f"shape: {df.shape}")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}, 일수: {df['date'].nunique()}")
    print(df.head())
    print(df.tail())
