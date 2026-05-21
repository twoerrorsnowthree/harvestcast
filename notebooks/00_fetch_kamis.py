"""
00_fetch_kamis.py
=================

KAMIS API로 5~6년치 가락도매시장 딸기 1kg 상등급 도매가 일괄 수집.

(주의) 16번 도매 API는 품종(kindcode) 구분 없이 "딸기" 통합 데이터.
       기존 웹다운로드의 '딸기 설향' 과 약간 다를 수 있음 — 가락 도매는
       대부분 설향이라 실질적 차이는 거의 없음.

사전 준비:
    export KAMIS_CERT_KEY='발급받은_키'
    export KAMIS_CERT_ID='발급받은_아이디'

(또는 .env 파일에 적어둔 뒤 `source .env` 실행)

실행:
    python notebooks/00_fetch_kamis.py
또는 기간 지정:
    python notebooks/00_fetch_kamis.py 2020-01-01 2026-04-30

결과:
    data/raw/price/kamis_api_full.csv  (날짜·가격 일별 시계열)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kamis_api import KamisCredentials, KamisQuery, fetch_range  # noqa: E402


# 기본 기간: 2020-01-01 ~ 2026-05-02
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-05-02"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "price"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "kamis_api_full.csv"


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_END
    print(f"수집 기간: {start} ~ {end}")
    print(f"품목: 딸기 / 시장: 가락도매 / 단위: 1kg / 등급: 상품")

    creds = KamisCredentials.from_env()
    query = KamisQuery(
        itemcategorycode="200",     # 채소류 (KAMIS 분류상 딸기는 채소)
        itemcode="226",             # 딸기
        kindcode="00",              # 딸기 통합 (16번 API 품종 구분 없음)
        productrankcode="04",       # 상품
        countrycode="1101",         # 서울 — 가락도매는 응답에서 marketname 필터로 추출
        convert_kg_yn="Y",
    )

    # chunk_days=30 — KAMIS API의 행 수 제한 회피 (자세한 내용은 fetch_range docstring)
    df = fetch_range(
        start, end,
        query=query, creds=creds,
        chunk_days=30, sleep_s=0.3,
        market_filter="가락",
    )
    print(f"\n총 {len(df)}일 수집됨")
    if not df.empty:
        print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
        df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"저장: {OUT_PATH}")
    else:
        print("⚠ 데이터가 0건. 응답이 비어있거나 코드/필터가 맞지 않음.")
        print("  market_filter=None 으로 다시 호출하면 어떤 시장이 잡히는지 확인 가능.")


if __name__ == "__main__":
    main()
