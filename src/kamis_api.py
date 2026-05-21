"""
KAMIS Open API 클라이언트.

문서: https://www.kamis.or.kr/customer/reference/openapi_list.do

사용 엔드포인트:
    periodWholesaleProductList — 신) 일별 품목별 도매 가격자료
    (기간 설정 가능, 최대 1년)

응답 필드:
    regday  : 날짜
    price   : 가격
    itemname, kindname, countyname, marketname

인증:
    cert_key, cert_id 두 개. 환경변수로:
        KAMIS_CERT_KEY, KAMIS_CERT_ID

지역 코드 (도매가격):
    1101 = 서울 (가락도매시장 포함)
    2100 = 부산, 2200 = 대구, 2401 = 광주, 2501 = 대전

품목 코드 (참고 — 인증 안내문서 코드표로 재확인 필수):
    400 = 과일류 (itemcategorycode)
    323 = 딸기 (itemcode)         ← 가능성 높음. 확인 필요
    01  = 설향 (kindcode)         ← 가능성 높음. 확인 필요
    04  = 상품 (productrankcode)
    05  = 중품
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import Iterator
from urllib.parse import urlencode

import requests
import pandas as pd


KAMIS_BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"


@dataclass
class KamisCredentials:
    cert_key: str
    cert_id: str

    @classmethod
    def from_env(cls) -> "KamisCredentials":
        key = os.environ.get("KAMIS_CERT_KEY")
        cid = os.environ.get("KAMIS_CERT_ID")
        if not key or not cid:
            raise RuntimeError(
                "KAMIS_CERT_KEY / KAMIS_CERT_ID 환경변수가 필요합니다.\n"
                "  export KAMIS_CERT_KEY='...'  ; export KAMIS_CERT_ID='...'"
            )
        return cls(cert_key=key, cert_id=cid)


@dataclass
class KamisQuery:
    """
    기본값: 서울 / 딸기 / 상품 / 도매 / kg 환산.

    KAMIS 분류상 딸기는 "채소류(200)" 안에 들어감 (과일류 아님).
    품종(kindcode)은 16번 도매 API에서 00 = 딸기 통합 (설향 따로 없음).
    가락도매만 보려면 응답의 marketname 으로 필터링 필요.
    """

    itemcategorycode: str = "200"     # 200 = 채소류
    itemcode: str = "226"             # 226 = 딸기
    kindcode: str = "00"              # 00 = 딸기(전체 품종)
    productrankcode: str = "04"       # 04 = 상품 (05 = 중품)
    countrycode: str = "1101"         # 1101 = 서울 (가락도매 포함)
    convert_kg_yn: str = "Y"          # Y = 1kg 환산
    returntype: str = "json"


def fetch_period(
    start: str,
    end: str,
    query: KamisQuery,
    creds: KamisCredentials,
    timeout: int = 30,
    base_url: str = KAMIS_BASE_URL,
) -> dict:
    """
    KAMIS periodWholesaleProductList 호출 (API 16번, 일별 품목별 도매).
    start/end: 'YYYY-MM-DD'  (한 번에 최대 1년)
    """
    params = {
        "action": "periodWholesaleProductList",
        "p_startday": start,
        "p_endday": end,
        "p_itemcategorycode": query.itemcategorycode,
        "p_itemcode": query.itemcode,
        "p_kindcode": query.kindcode,
        "p_productrankcode": query.productrankcode,
        "p_countrycode": query.countrycode,
        "p_convert_kg_yn": query.convert_kg_yn,
        "p_cert_key": creds.cert_key,
        "p_cert_id": creds.cert_id,
        "p_returntype": query.returntype,
    }
    url = f"{base_url}?{urlencode(params)}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if query.returntype == "json":
        return resp.json()
    return {"raw": resp.text}


def parse_period_response(payload: dict) -> pd.DataFrame:
    """
    periodWholesaleProductList 응답 → DataFrame.
    필드: regday, price, itemname, kindname, countyname, marketname

    Returns:
        columns = [date, price, marketname, kindname]
    """
    # 일반적으로 KAMIS는 condition + data 구조로 반환
    candidates = []
    if isinstance(payload.get("data"), dict):
        items = payload["data"].get("item")
        if isinstance(items, list):
            candidates = items
        elif isinstance(items, dict):
            candidates = [items]
    if not candidates and isinstance(payload.get("data"), list):
        candidates = payload["data"]

    if not candidates:
        # 빈 결과 또는 에러 메시지 출력 (디버깅용)
        cond = payload.get("condition") or payload.get("data") or payload
        raise RuntimeError(
            f"KAMIS 응답에서 데이터 행을 찾지 못함.\n응답: {str(cond)[:400]}"
        )

    rows = []
    for c in candidates:
        date_str = c.get("regday") or c.get("yyyy")
        price_str = c.get("price")
        if not date_str or price_str in (None, "", "-"):
            continue

        # regday 가 'MM-DD' 짧은 형태일 수도 → yyyy 와 결합
        s = str(date_str).strip().replace("/", "-").replace(".", "-")
        if len(s) <= 5 and c.get("yyyy"):
            s = f"{c['yyyy']}-{s.lstrip('-')}"

        try:
            p = float(str(price_str).replace(",", ""))
        except ValueError:
            continue

        rows.append({
            "date_str": s,
            "price": p,
            "marketname": c.get("marketname", ""),
            "kindname": c.get("kindname", ""),
            "countyname": c.get("countyname", ""),
        })

    if not rows:
        return pd.DataFrame(columns=["date", "price", "marketname", "kindname"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date_str"], errors="coerce")
    df = df.dropna(subset=["date"]).drop(columns=["date_str"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def iter_chunks(start: str, end: str, days: int = 180) -> Iterator[tuple[str, str]]:
    """
    start ~ end 기간을 days 단위로 잘라서 (chunk_start, chunk_end) yield.
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    cur = s
    while cur <= e:
        nxt = min(cur + pd.Timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


def fetch_range(
    start: str,
    end: str,
    query: KamisQuery | None = None,
    creds: KamisCredentials | None = None,
    chunk_days: int = 30,
    sleep_s: float = 0.3,
    market_filter: str | None = "가락",
) -> pd.DataFrame:
    """
    start ~ end 기간을 chunk_days 단위로 나눠 받아 합친 DataFrame을 반환.

    16번 API 명세상 "최대 1년" 이라고 적혀있지만, 실제로는 응답에 행 수 제한
    (~50행 추정) 이 걸려있어서 1년치를 한 번에 받으면 첫 ~2개월만 잡힘.
    chunk_days=30 으로 충분히 작게 자르면 모든 거래일이 잡힘.
    7년치 ≈ 84번 호출, sleep 0.3s 면 ~30초 소요.

    market_filter: 응답의 marketname 컬럼을 부분 일치(contains) 로 필터.
                   기본 '가락' → 가락도매시장 데이터만 남김.
                   None 이면 모든 시장 결과를 사용 (날짜별 평균이 됨).
    """
    query = query or KamisQuery()
    creds = creds or KamisCredentials.from_env()

    frames = []
    for cs, ce in iter_chunks(start, end, chunk_days):
        print(f"  fetching {cs} ~ {ce} ...", flush=True)
        try:
            payload = fetch_period(cs, ce, query, creds)
            df = parse_period_response(payload)

            if market_filter and not df.empty and "marketname" in df.columns:
                before = len(df)
                df = df[df["marketname"].str.contains(market_filter, na=False)].copy()
                print(f"    → {before}행 → 가락 필터 후 {len(df)}행")
            else:
                print(f"    → {len(df)}행")

            if not df.empty:
                frames.append(df[["date", "price"]])
        except Exception as e:
            print(f"    [error] {e}")
        time.sleep(sleep_s)

    if not frames:
        return pd.DataFrame(columns=["date", "price"])
    out = pd.concat(frames, ignore_index=True)
    # 같은 날짜에 여러 시장이 있으면(필터 안 했을 때) 평균
    out = out.groupby("date", as_index=False)["price"].mean()
    out = out.sort_values("date").reset_index(drop=True)
    return out


if __name__ == "__main__":
    import sys

    # 빠른 테스트: 최근 2주
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-01-15"
    creds = KamisCredentials.from_env()
    df = fetch_range(start, end, creds=creds)
    print(df)
