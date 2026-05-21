"""
HarvestCast — 딸기 도매가 예측 시스템 (Streamlit)

상단 네비게이션 + 3페이지 구조:
    🏠 메인  — 한 줄 소개 + 핵심 지표 + 7일 예측 미리보기 + CTA
    🔮 예측  — 날짜/기간 입력 + Mamba 예측 그래프 + 자연어 설명
    📊 분석  — 과거 가격 시계열 + 계절성 + 트렌드 + 데이터 출처

실행:
    streamlit run app/streamlit_app.py
"""
from datetime import date, timedelta
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# Altair 라이트 테마 전역 설정
def _light_theme():
    return {
        "config": {
            "background": "#ffffff",
            "view": {"stroke": "transparent", "fill": "#ffffff"},
            "axis": {
                "labelColor": "#1a1a1a",
                "titleColor": "#1a1a1a",
                "gridColor": "#f0f0f0",
                "domainColor": "#cccccc",
                "tickColor": "#cccccc",
                "labelFontSize": 12,
                "titleFontSize": 13,
            },
            "legend": {
                "labelColor": "#1a1a1a",
                "titleColor": "#1a1a1a",
            },
            "title": {"color": "#1a1a1a"},
        }
    }
alt.themes.register("light_pink", _light_theme)
alt.themes.enable("light_pink")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# 실제 모델 우선, 없으면 mock
try:
    from real_predictor import predict, load_historical, get_model_info  # noqa: E402
    USING_REAL_MODEL = True
    MODEL_INFO = get_model_info()
except Exception:
    from mock_predictor import load_historical, predict  # noqa: E402
    USING_REAL_MODEL = False
    MODEL_INFO = None


# ============================================================
# 페이지 설정 + CSS
# ============================================================
st.set_page_config(
    page_title="HarvestCast — 딸기 가격 예측",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* 스트림릿 기본 chrome 완전히 제거 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { display: none !important; visibility: hidden !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1rem; padding-bottom: 4rem; max-width: 1200px; }
section[data-testid="stSidebar"] { display: none; }

/* 입력 위젯 라이트 모드 강제 */
.stTextInput input, .stNumberInput input, .stDateInput input {
    background-color: white !important;
    color: #1a1a1a !important;
    border: 1px solid #ffe5ec !important;
    border-radius: 12px !important;
}
.stDateInput > div > div {
    background-color: white !important;
    border-radius: 12px !important;
}
.stNumberInput > div > div {
    background-color: white !important;
    border-radius: 12px !important;
}
.stTextInput label, .stNumberInput label, .stDateInput label, .stSlider label {
    color: #555 !important;
    font-weight: 500;
}
/* 슬라이더 트랙/숫자 */
.stSlider [data-baseweb="slider"] { color: #1a1a1a; }
.stSlider [role="slider"] { background-color: #E63946 !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #888 !important; }

/* Altair 차트 라이트 모드 */
[data-testid="stAltairChart"] {
    background: white;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    border: 1px solid #fff0f4;
}

/* 배경 — 아래쪽만 살짝 핑크 그라데이션 (딸기우유) */
.stApp {
    background: linear-gradient(180deg,
        #ffffff 0%,
        #ffffff 50%,
        #fffafc 75%,
        #ffeef3 100%);
    background-attachment: fixed;
}

/* 상단 네비게이션 바 */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 12px;
    border-bottom: 1px solid #f5e5ea;
    margin-bottom: 24px;
}
.brand-logo {
    font-size: 1.5em;
    font-weight: 800;
    color: #1a1a1a;
    letter-spacing: -0.5px;
}
.brand-logo .accent { color: #E63946; }

/* 네비 버튼 (커스텀) */
.stButton>button {
    background: transparent;
    color: #555;
    border: none;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
    box-shadow: none;
    font-size: 0.95em;
}
.stButton>button:hover {
    background: #fff5f7;
    color: #E63946;
}

/* "Primary" 버튼은 그라데이션 */
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #E63946 0%, #ff6b8b 100%);
    color: white;
    box-shadow: 0 4px 14px rgba(230,57,70,0.2);
    padding: 12px 28px;
    font-size: 1em;
}
.stButton>button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(230,57,70,0.3);
    color: white;
}

/* HERO */
.hero {
    padding: 60px 40px 30px 40px;
    text-align: left;
    margin-bottom: 30px;
}
.hero h1 {
    color: #1a1a1a !important;
    font-size: 3.4em !important;
    margin: 0 0 18px 0 !important;
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.05;
}
.hero .tagline {
    color: #444;
    font-size: 1.15em;
    margin: 0 0 10px 0;
    font-weight: 500;
    line-height: 1.5;
    word-break: keep-all;          /* 한국어 단어 단위 줄바꿈 */
    overflow-wrap: break-word;
}
.hero .sub {
    color: #666;
    font-size: 0.95em;
    line-height: 1.6;
}
.hero .badge {
    display: inline-block;
    background: #fff5f7;
    color: #E63946;
    padding: 5px 12px;
    border-radius: 18px;
    font-size: 0.8em;
    margin-bottom: 18px;
    font-weight: 600;
    border: 1px solid #ffd6e0;
}

/* 섹션 헤더 */
.section-title {
    color: #1a1a1a !important;
    font-size: 1.6em !important;
    margin: 40px 0 6px 0 !important;
    font-weight: 700;
    letter-spacing: -0.4px;
}
.section-sub {
    color: #888;
    font-size: 0.92em;
    margin-bottom: 24px;
}

/* 통계 카드 */
.stat-card {
    background: white;
    padding: 28px 20px;
    border-radius: 16px;
    text-align: left;
    box-shadow: 0 2px 12px rgba(230,57,70,0.04);
    border: 1px solid #fff0f4;
    height: 100%;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(230,57,70,0.08);
}
.stat-number {
    font-size: 2.2em;
    font-weight: 800;
    background: linear-gradient(135deg, #E63946 0%, #ff6b8b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.85em;
    color: #666;
    margin-top: 6px;
    font-weight: 500;
}
.stat-delta {
    font-size: 0.8em;
    color: #888;
    margin-top: 4px;
}

/* 피처/일반 카드 */
.feature-card {
    background: white;
    padding: 30px 24px;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    height: 100%;
}
.feature-icon { font-size: 1.8em; margin-bottom: 12px; }
.feature-card h3 {
    color: #1a1a1a !important;
    font-size: 1.1em !important;
    margin: 4px 0 10px 0 !important;
    font-weight: 700;
}
.feature-card p {
    color: #666;
    font-size: 0.9em;
    line-height: 1.55;
    margin: 0;
}

/* CTA 배너 */
.cta-banner {
    background: linear-gradient(135deg, #fff5f7 0%, #ffe5ec 100%);
    border-left: 4px solid #E63946;
    padding: 22px 28px;
    border-radius: 14px;
    margin: 28px 0;
    color: #333;
}

/* AI 해설 박스 */
.ai-explain {
    background: linear-gradient(135deg, #fffafc 0%, #fff5f7 100%);
    border: 1px solid #ffe5ec;
    padding: 24px 28px;
    border-radius: 16px;
    margin: 24px 0;
    line-height: 1.7;
    color: #333;
}
.ai-explain .ai-label {
    display: inline-block;
    font-size: 0.8em;
    color: #E63946;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}

/* 메트릭 카드 */
.stMetric {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #fff0f4;
    box-shadow: 0 2px 10px rgba(230,57,70,0.04);
}
.stMetric label { color: #777 !important; font-weight: 500; }
.stMetric [data-testid="stMetricValue"] {
    color: #1a1a1a !important;
    font-weight: 700;
}

/* selectbox */
div[data-baseweb="select"] > div {
    border-radius: 12px;
    border: 1px solid #ffe5ec !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 0 3px #ffe5ec !important;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
    color: #888;
}
.stTabs [aria-selected="true"] {
    background: #fff5f7;
    color: #E63946;
}

/* 푸터 */
.footer {
    text-align: center;
    color: #999;
    font-size: 0.82em;
    margin-top: 70px;
    padding: 28px;
    border-top: 1px solid #ffe5ec;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 상단 네비게이션 + 세션 상태
# ============================================================
PAGES = ["🏠 메인", "🔮 예측", "📊 분석"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# 상단 바
top_left, top_n1, top_n2, top_n3 = st.columns([4, 1, 1, 1])
with top_left:
    st.markdown(
        "<div class='brand-logo'>🍓 Harvest<span class='accent'>Cast</span></div>",
        unsafe_allow_html=True,
    )
for col, page_name in zip([top_n1, top_n2, top_n3], PAGES):
    with col:
        if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name

st.markdown("<hr style='border:none; border-top:1px solid #f5e5ea; margin:8px 0 16px 0;'>", unsafe_allow_html=True)


# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def get_historical():
    return load_historical(str(PROJECT_ROOT / "data" / "processed" / "merged_daily.csv"))


@st.cache_data
def get_full_df():
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "merged_daily.csv", parse_dates=["date"])


hist = get_historical()
last_price = float(hist["price"].iloc[-1]) if len(hist) else 12000.0
last_date = hist["date"].iloc[-1] if len(hist) else pd.Timestamp.today()


@st.cache_data(show_spinner="🍓 예측 중...")
def run_prediction(start_date, horizon_days, last_p):
    return predict(start_date=start_date, horizon_days=horizon_days, last_price=last_p, seed=42)


def ai_explanation(pred_df: pd.DataFrame) -> str:
    """간단한 자연어 해설 (mock — 향후 GPT-4o-mini API 연동)."""
    avg = pred_df["predicted_price"].mean()
    first = pred_df["predicted_price"].iloc[0]
    last = pred_df["predicted_price"].iloc[-1]
    change_pct = (last - first) / first * 100
    band = (pred_df["upper"] - pred_df["lower"]).mean() / 2

    if change_pct > 5:
        trend = f"전반적으로 **상승 추세**({change_pct:+.1f}%)"
        why = "출하기 후반으로 갈수록 공급 감소와 시장 수요 누적이 영향을 미친 것으로 보입니다."
    elif change_pct < -5:
        trend = f"전반적으로 **하락 추세**({change_pct:+.1f}%)"
        why = "출하량 증가 및 봄철 다른 과채류 소비 분산이 영향을 미친 것으로 보입니다."
    else:
        trend = f"**안정적인 흐름**({change_pct:+.1f}%)"
        why = "현재 시장 수급 균형이 유지되며 가격 변동성이 크지 않을 것으로 예상됩니다."

    return (
        f"예측 기간 평균 도매가는 **{int(avg):,}원/kg** 수준이며, {trend}을 보일 것으로 예상됩니다. "
        f"신뢰구간은 평균 ±{int(band):,}원으로, 입도선매 계약 시 가격 변동 위험 범위로 참고하시기 바랍니다. "
        f"{why}"
    )


# ============================================================
# 페이지 1: 🏠 메인
# ============================================================
def render_home():
    status_text = "🟢 Live" if USING_REAL_MODEL else "🟡 Demo Mode"

    # Hero 풀폭 — tagline 한 줄 들어가게
    st.markdown(f"""
    <div class='hero'>
        <div class='badge'>{status_text} · SSM(Mamba) 기반 · 6시즌 데이터</div>
        <h1>오늘의 딸기 도매가,<br>내일을 미리 봅니다.</h1>
        <p class='tagline'>KAMIS 가락도매시장 + 산지 기상 데이터를 결합해 일별 가격을 예측합니다.</p>
        <p class='sub'>입도선매 계약 시 데이터 기반 의사결정을 지원하는 농가·유통업체 도구.</p>
    </div>
    """, unsafe_allow_html=True)

    # CTA 버튼 (적당한 폭만 차지)
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        if st.button("🔮 바로 예측해보기", key="cta_to_predict", type="primary"):
            st.session_state.page = "🔮 예측"
            st.rerun()

    # 핵심 지표 3개
    st.markdown("<h2 class='section-title'>핵심 지표</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>최근 데이터 기반 요약</p>", unsafe_allow_html=True)

    today_pred = run_prediction(date.today(), 7, last_price)
    week_avg = today_pred["predicted_price"].mean()
    last_actual_week = hist["price"].iloc[-7:].mean() if len(hist) >= 7 else last_price
    delta_pct = (week_avg - last_actual_week) / last_actual_week * 100

    mape = MODEL_INFO.get("validation_MAPE", "8.73") if MODEL_INFO else "8.73"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number'>{int(week_avg):,}원</div>
            <div class='stat-label'>이번 주 평균 예측가 (kg)</div>
            <div class='stat-delta'>가락도매 · 설향 1kg 상등급</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        delta_color = "#E63946" if delta_pct > 0 else "#3a86ff"
        delta_sign = "▲" if delta_pct > 0 else "▼"
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color:{delta_color}; -webkit-text-fill-color:{delta_color};'>
                {delta_sign} {abs(delta_pct):.1f}%
            </div>
            <div class='stat-label'>지난주 대비 등락</div>
            <div class='stat-delta'>지난주 평균 {int(last_actual_week):,}원</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number'>{mape}%</div>
            <div class='stat-label'>모델 정확도 (MAPE)</div>
            <div class='stat-delta'>walk-forward 검증 기준</div>
        </div>""", unsafe_allow_html=True)

    # 7일 예측 미리보기
    st.markdown("<h2 class='section-title'>이번 주 예측 추이</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>최근 30일 + 향후 7일</p>", unsafe_allow_html=True)

    hist_recent = hist.tail(30).copy().rename(columns={"price": "value"})
    hist_recent["type"] = "실제"
    pred_chart = today_pred.rename(columns={"predicted_price": "value"}).copy()
    pred_chart["type"] = "예측"
    chart_df = pd.concat([hist_recent[["date", "value", "type"]], pred_chart[["date", "value", "type"]]])

    line = alt.Chart(chart_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y("value:Q", title="가격 (원/kg)"),
        color=alt.Color("type:N",
                        scale=alt.Scale(domain=["실제", "예측"], range=["#1a1a1a", "#E63946"]),
                        legend=alt.Legend(title="")),
    )
    band = alt.Chart(today_pred).mark_area(opacity=0.18, color="#E63946").encode(
        x="date:T", y="lower:Q", y2="upper:Q",
    )
    st.altair_chart((band + line).properties(height=320), use_container_width=True)

    # 푸터 안내
    st.markdown("""
    <div class='cta-banner'>
        💡 <b>입도선매 계약 의사결정 보조</b> — 예측가 + 95% 신뢰구간으로 가격 변동 위험을 정량적으로 파악하세요.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 페이지 2: 🔮 예측
# ============================================================
def render_predict():
    st.markdown("<h1 style='font-size:2.4em; margin:30px 0 8px 0; color:#1a1a1a; font-weight:800; letter-spacing:-1px;'>🔮 가격 예측</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:30px;'>날짜·기간을 설정하면 Mamba(SSM) 모델이 예측을 수행합니다.</p>", unsafe_allow_html=True)

    # 입력 영역 (상단 카드)
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.5])
        with c1:
            today = pd.Timestamp.today().normalize().date()
            start = st.date_input("예측 시작일", value=today,
                                  min_value=date(2020, 1, 1),
                                  max_value=today + timedelta(days=180))
        with c2:
            horizon = st.slider("예측 기간 (일)", min_value=7, max_value=60, value=14)
        with c3:
            contract_kg = st.number_input("계약 수량 (kg)", min_value=10, max_value=10000, value=500, step=10)
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("예측 실행", type="primary", use_container_width=True)

    try:
        pred = run_prediction(start, int(horizon), float(last_price))
    except Exception as e:
        st.error(f"예측 오류: {e}")
        st.stop()

    # 메트릭
    st.markdown("<h2 class='section-title'>예측 요약</h2>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("평균 예측가", f"{pred['predicted_price'].mean():,.0f} 원")
    m2.metric("최고가", f"{pred['predicted_price'].max():,.0f} 원")
    m3.metric("최저가", f"{pred['predicted_price'].min():,.0f} 원")
    m4.metric(f"예상 매출 ({contract_kg}kg)",
              f"{pred['predicted_price'].mean() * contract_kg:,.0f} 원")

    # 차트
    st.markdown("<h2 class='section-title'>예측 그래프</h2>", unsafe_allow_html=True)
    hist_recent = hist.tail(60).copy().rename(columns={"price": "value"})
    hist_recent["type"] = "실제"
    pred_chart = pred.rename(columns={"predicted_price": "value"}).copy()
    pred_chart["type"] = "예측"
    chart_df = pd.concat([hist_recent[["date", "value", "type"]], pred_chart[["date", "value", "type"]]])

    line = alt.Chart(chart_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("value:Q", title="가격 (원/kg)"),
        color=alt.Color("type:N",
                        scale=alt.Scale(domain=["실제", "예측"], range=["#1a1a1a", "#E63946"]),
                        legend=alt.Legend(title="")),
    )
    band = alt.Chart(pred).mark_area(opacity=0.18, color="#E63946").encode(
        x="date:T", y="lower:Q", y2="upper:Q",
    )
    st.altair_chart((band + line).properties(height=420), use_container_width=True)
    st.caption("음영 = 95% 신뢰구간  ·  검정 = 과거 실제가  ·  핑크 = 예측가")

    # AI 해설
    st.markdown("<h2 class='section-title'>AI 해설</h2>", unsafe_allow_html=True)
    explain = ai_explanation(pred)
    st.markdown(f"""
    <div class='ai-explain'>
        <div class='ai-label'>🤖 GPT 기반 자연어 해설 (시범)</div>
        {explain}
    </div>
    """, unsafe_allow_html=True)
    st.caption("⚠ 현재 규칙 기반 시범 버전 — 향후 GPT-4o-mini API 연동 예정")

    # 결과 표
    with st.expander("📋 일별 예측 표"):
        st.dataframe(
            pred.assign(
                predicted_price=pred["predicted_price"].round(0).astype(int),
                lower=pred["lower"].round(0).astype(int),
                upper=pred["upper"].round(0).astype(int),
            ).rename(columns={
                "date": "날짜", "predicted_price": "예측가",
                "lower": "하한 (95% CI)", "upper": "상한 (95% CI)"
            }),
            use_container_width=True, hide_index=True,
        )


# ============================================================
# 페이지 3: 📊 데이터 분석
# ============================================================
def render_analyze():
    st.markdown("<h1 style='font-size:2.4em; margin:30px 0 8px 0; color:#1a1a1a; font-weight:800; letter-spacing:-1px;'>📊 데이터 분석</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:30px;'>과거 가격 시계열·계절성·트렌드 시각화 + 데이터 출처</p>", unsafe_allow_html=True)

    try:
        df = get_full_df()
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

    # 전체 시계열 (큰 그래프)
    st.markdown("<h2 class='section-title'>📈 가락도매 딸기(설향) 일별 도매가 — 전체 기간</h2>", unsafe_allow_html=True)
    df_p = df.dropna(subset=["price"]).copy()

    base = alt.Chart(df_p).mark_line(strokeWidth=1.5, color="#E63946", opacity=0.8).encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("price:Q", title="가격 (원/kg)"),
        tooltip=["date:T", alt.Tooltip("price:Q", format=",")],
    )
    ma30 = alt.Chart(df_p.assign(ma=df_p["price"].rolling(30, min_periods=1).mean())).mark_line(
        strokeWidth=2.5, color="#1a1a1a"
    ).encode(x="date:T", y="ma:Q")
    st.altair_chart((base + ma30).properties(height=420), use_container_width=True)
    st.caption("핑크 = 일별 도매가  ·  검정 = 30일 이동평균")

    # 계절성 (월별 박스플롯)
    st.markdown("<h2 class='section-title'>🗓 월별 가격 분포 — 계절성</h2>", unsafe_allow_html=True)
    df_p["month"] = df_p["date"].dt.month
    df_p["year"] = df_p["date"].dt.year
    box = alt.Chart(df_p).mark_boxplot(color="#E63946", opacity=0.7, size=30).encode(
        x=alt.X("month:O", title="월"),
        y=alt.Y("price:Q", title="가격 (원/kg)"),
    )
    st.altair_chart(box.properties(height=360), use_container_width=True)
    st.caption("출하기(11~5월) 가격이 높고, 비출하기(6~10월)는 거래 자체가 거의 없음")

    # 연도별 트렌드
    st.markdown("<h2 class='section-title'>📅 연도별 평균가 추이</h2>", unsafe_allow_html=True)
    yearly = df_p.groupby("year")["price"].agg(["mean", "median", "std"]).reset_index()
    yearly = yearly.rename(columns={"mean": "평균", "median": "중앙값", "std": "표준편차"})
    yearly_long = yearly.melt(id_vars="year", value_vars=["평균", "중앙값"], var_name="지표", value_name="가격")
    bar = alt.Chart(yearly_long).mark_bar().encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("가격:Q", title="가격 (원/kg)"),
        color=alt.Color("지표:N", scale=alt.Scale(range=["#E63946", "#ff8fa3"])),
        column=alt.Column("지표:N", title=""),
    )
    st.altair_chart(bar.properties(height=300), use_container_width=False)

    # 데이터 출처
    st.markdown("<h2 class='section-title'>📚 데이터 출처</h2>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>💰</div>
            <h3>KAMIS</h3>
            <p>가락도매시장 딸기(설향) 1kg 상등급 일별 도매가<br>(2020-01 ~ 2026-04, 1,035일)</p>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>🌤️</div>
            <h3>기상청 ASOS</h3>
            <p>가락 산지(전남 광주 + 경남 산청) 평균 기상<br>일자료 평균기온/강수/풍속/지면온도/전운량</p>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""<div class='feature-card'>
            <div class='feature-icon'>📝</div>
            <h3>KAMIS 일일동향</h3>
            <p>일일 시장동향 보고서 582건 텍스트 분석<br>시장-산지 매핑 정량 도출 근거</p>
        </div>""", unsafe_allow_html=True)


# ============================================================
# 페이지 라우팅
# ============================================================
if st.session_state.page == "🏠 메인":
    render_home()
elif st.session_state.page == "🔮 예측":
    render_predict()
elif st.session_state.page == "📊 분석":
    render_analyze()


# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class='footer'>
    🍓 <b>HarvestCast</b> — SSM 기반 딸기 도매가 예측 시스템<br>
    © 고려대학교 컴퓨터소프트웨어학과 Capstone Design 2026 · 우채연 · 김소령 · 지도교수 조민호 · 참여기업 ㈜바르카<br>
    <span style='opacity:0.7;'>Data: KAMIS · 기상청 기상자료개방포털</span>
</div>
""", unsafe_allow_html=True)
