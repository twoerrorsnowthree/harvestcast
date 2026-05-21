"""
09_fetch_other_markets.py
=========================

가락(서울) 외 4개 도매시장 (부산/대구/광주/대전) 일별 도매가를 수집한다.
가설: "다른 도시 도매가는 가락 도매가의 선행/후행 지표가 될 수 있다."

이 데이터를 보조 feature 로 추가해서 모델 성능이 올라가는지 검증.

산출:
    data/raw/price/kamis_other_markets.csv
        columns: date, busan, daegu, gwangju, daejeon

(서울/가락은 기존 kamis_api_full.csv 에 이미 있음)

사전 준비:
    export KAMIS_CERT_KEY='...'
    export KAMIS_CERT_ID='...'

실행:
    python notebooks/09_fetch_other_markets.py
또는 기간 지정:
    python notebooks/09_fetch_other_markets.py 2020-01-01 2026-05-02

소요 시간: 4개 시장 × 78 청크(30일씩) × ~0.5s ≈ 약 3분
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kamis_api import KamisCredentials, KamisQuery, fetch_range  # noqa: E402

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-05-02"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "price"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "kamis_other_markets.csv"


# 가락 외 4개 도매시장 (KAMIS 도매가 지원 지역)
MARKETS = [
    ("busan",   "2100", "부산"),
    ("daegu",   "2200", "대구"),
    ("gwangju", "2401", "광주"),
    ("daejeon", "2501", "대전"),
]


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_END
    print(f"수집 기간: {start} ~ {end}")
    print(f"품목: 딸기 / 단위: 1kg / 등급: 상품")
    print(f"시장: {', '.join(m[2] for m in MARKETS)}")
    print()

    creds = KamisCredentials.from_env()
    series = {}  # market_name -> DataFrame(date, price)

    for col_name, country_code, korean in MARKETS:
        print(f"━━━ {korean}({country_code}) ━━━")
        query = KamisQuery(
            itemcategorycode="200",   # 채소류
            itemcode="226",           # 딸기
            kindcode="00",            # 통합
            productrankcode="04",     # 상품
            countrycode=country_code,
            convert_kg_yn="Y",
        )
        # 시장 필터 = None : 해당 지역 모든 도매시장 평균 사용
        df = fetch_range(start, end, query=query, creds=creds, chunk_days=30, sleep_s=0.3, market_filter=None)
        print(f"  → {len(df)}일 수집됨\n")
        if not df.empty:
            series[col_name] = df.rename(columns={"price": col_name})

    if not series:
        print("⚠ 모든 시장에서 데이터 0건. 코드/필터/인증 확인 필요.")
        return

    # 4개 시리즈 outer join (날짜 기준)
    out = None
    for col, df in series.items():
        out = df if out is None else out.merge(df, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)

    print(f"통합 결과: {len(out)}일, {len(series)}개 시장")
    print(f"기간: {out['date'].min().date()} ~ {out['date'].max().date()}")
    print(f"\n각 시장 가격 보유일:")
    for col in series:
        n = out[col].notna().sum()
        print(f"  {col}: {n}일")

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
