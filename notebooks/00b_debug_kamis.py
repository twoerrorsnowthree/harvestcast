"""
00b_debug_kamis.py
==================

KAMIS API 응답이 어떻게 생겼는지 한눈에 보기.
1주일치만 받아서:
  - marketname 별 행 수
  - kindname 별 행 수
  - 가락이 아니라 다른 표기일 수도 있어서 모두 나열
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kamis_api import KamisCredentials, KamisQuery, fetch_period, parse_period_response  # noqa: E402

creds = KamisCredentials.from_env()
query = KamisQuery()  # 기본값: 200/226/00/04/1101

# 출하기 중, 데이터 잘 나올 만한 날짜
print("호출: 2024-02-01 ~ 2024-02-07 (출하 성수기)")
payload = fetch_period("2024-02-01", "2024-02-07", query, creds)

# raw payload 일부 출력
print("\n=== payload 키 ===")
print(list(payload.keys())[:10])

print("\n=== payload['condition'] ===")
print(str(payload.get("condition", ""))[:500])

# 파싱된 결과
df = parse_period_response(payload)
print(f"\n=== 파싱 후 행 수: {len(df)} ===")
if not df.empty:
    print(df.head(20).to_string())
    print()
    print("marketname 별 행 수:")
    print(df["marketname"].value_counts())
    print()
    print("kindname 별 행 수:")
    print(df["kindname"].value_counts())
    print()
    print("countyname 별 행 수:")
    print(df["countyname"].value_counts() if "countyname" in df.columns else "없음")
