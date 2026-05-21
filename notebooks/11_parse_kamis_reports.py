"""
11_parse_kamis_reports.py
=========================

KAMIS 일일동향 RTF 파일들 자동 파싱 → 통합 CSV.

산출:
    data/raw/kamis_reports/kamis_daily_reports_all.csv
    data/processed/market_region_mapping.csv  (시장별 산지 빈도)
    figures/17_market_region_heatmap.png       (시장 × 산지 매트릭스)
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from striprtf.striprtf import rtf_to_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "kamis_reports" / "rtf"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "kamis_reports"
FIGURES = PROJECT_ROOT / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# 한글 폰트
for fn in ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"]:
    plt.rcParams["font.family"] = fn
    break
plt.rcParams["axes.unicode_minus"] = False


def parse_rtf_text(text: str) -> list[dict]:
    """RTF→텍스트 변환 후 보고서 단위로 분리, 구조화 dict 리스트 반환."""
    blocks = re.split(r"\n(?=(?:소매|중도매)\n)", text.strip())
    records = []
    for blk in blocks:
        lines = [l.strip() for l in blk.strip().splitlines() if l.strip()]
        if len(lines) < 4:
            continue
        type_ = lines[0]
        header_match = re.match(r".*?딸기,\s*(\d{4}-\d{2}-\d{2}),\s*(\S+)", lines[1])
        if not header_match:
            continue
        date_str = header_match.group(1)
        market = header_match.group(2).replace("지역", "")
        bullets = [l.lstrip("●").strip() for l in lines if l.startswith("●")]
        source_line = bullets[0] if len(bullets) > 0 else ""
        trend_line = bullets[1] if len(bullets) > 1 else ""
        price_line = bullets[2] if len(bullets) > 2 else ""
        forecast_line = bullets[3] if len(bullets) > 3 else ""

        # 가격 추출
        price_상 = None; price_중 = None
        joined = source_line + trend_line + price_line
        m1 = re.search(r"상품\s*([\d,]+)\s*원", joined)
        m2 = re.search(r"중품\s*([\d,]+)\s*원", joined)
        if m1: price_상 = int(m1.group(1).replace(",", ""))
        if m2: price_중 = int(m2.group(1).replace(",", ""))

        records.append({
            "date": date_str,
            "market": market,
            "type": type_,
            "source_regions": source_line,
            "price_grade_상": price_상,
            "price_grade_중": price_중,
            "trend_note": trend_line,
            "forecast": forecast_line,
        })
    return records


def extract_regions(text: str) -> list[str]:
    """산지 첫 bullet 에서 시군 키워드 추출."""
    if not text or pd.isna(text):
        return []
    # 'X에서/X지역에서/주로' 등 cut
    text = re.sub(r"(에서|등에서|등지에서|등|주로).*$", "", str(text))
    text = text.replace(" 및 ", ", ").replace("·", ",").replace("⋅", ",")
    parts = re.split(r"[,，]", text)
    out = []
    for p in parts:
        p = p.strip()
        # "충남 논산" → "논산"; "충남 논산시" → "논산"
        m = re.search(r"(\S+?)(?:시|군|구|광역시|특별시)?$", p)
        if m:
            name = m.group(1)
            # 광역단위 자체는 제외
            if not re.fullmatch(r"(충남|충북|전남|전북|경남|경북|강원|제주|경기|광주|대전|대구|부산|서울)", name) and len(name) >= 2:
                out.append(name)
    return out


def main():
    # rtf 폴더 정리: 사용자가 uploads 에 둔 파일 자동 복사
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    uploads = Path("/sessions/blissful-adoring-albattani/mnt/uploads")
    # 직접 rtf 파일들 + 압축 안의 rtf 모두 수집
    rtfs = []
    for f in uploads.glob("*.rtf"):
        rtfs.append(f)
    # rtfd 압축 처리
    import zipfile, tempfile
    for f in uploads.iterdir():
        if f.is_file() and f.suffix == "" and zipfile.is_zipfile(f):
            tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(f) as z:
                z.extractall(tmp)
            rtfs += list(tmp.rglob("*.rtf"))

    print(f"발견 RTF: {len(rtfs)}개")
    for r in rtfs:
        print(f"  {r}")

    all_records = []
    for rtf_path in rtfs:
        with open(rtf_path, "r", encoding="utf-8") as f:
            text = rtf_to_text(f.read())
        recs = parse_rtf_text(text)
        all_records.extend(recs)
        print(f"  {rtf_path.name}: {len(recs)}건")

    if not all_records:
        print("⚠ 보고서 파싱 결과 0건")
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date", "market", "type"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["regions_list"] = df["source_regions"].apply(extract_regions)

    out_csv = OUT_DIR / "kamis_daily_reports_all.csv"
    df.drop(columns=["regions_list"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n총 보고서: {len(df)}건")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"저장: {out_csv}")

    # === 시장 × 산지 빈도 행렬 ===
    print("\n[시장별 주요 산지 TOP 5]")
    market_region_data = {}
    target_markets = ["서울", "부산", "대구", "광주", "대전"]
    for market in target_markets:
        sub = df[df["market"] == market]
        all_regions = []
        for regs in sub["regions_list"]:
            all_regions.extend(regs)
        top = Counter(all_regions).most_common(10)
        market_region_data[market] = dict(top)
        print(f"\n[{market}] (n={len(sub)})")
        for r, c in top[:5]:
            print(f"  {r:15s}: {c:3d}회")

    # 매트릭스 저장
    all_regions_set = set()
    for d in market_region_data.values():
        all_regions_set.update(d.keys())
    matrix_df = pd.DataFrame(0, index=sorted(all_regions_set), columns=target_markets)
    for market, regions in market_region_data.items():
        for region, count in regions.items():
            matrix_df.at[region, market] = count
    # 빈도 합으로 정렬
    matrix_df["_total"] = matrix_df.sum(axis=1)
    matrix_df = matrix_df.sort_values("_total", ascending=False).drop(columns=["_total"])
    # 상위 20개 산지만
    matrix_df = matrix_df.head(20)
    matrix_df.to_csv(PROCESSED / "market_region_mapping.csv", encoding="utf-8-sig")
    print(f"\n저장: {PROCESSED / 'market_region_mapping.csv'}")

    # === 시각화 ===
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(matrix_df.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns)
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index)
    ax.set_title("도매시장별 주요 산지 출현 빈도 (KAMIS 일일동향)")
    plt.colorbar(im, ax=ax, label="등장 횟수")
    # 셀에 숫자 표시
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            v = matrix_df.iat[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > matrix_df.values.max() * 0.5 else "black",
                        fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "17_market_region_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {FIGURES / '17_market_region_heatmap.png'}")


if __name__ == "__main__":
    main()
