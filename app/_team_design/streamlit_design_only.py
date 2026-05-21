"""
Streamlit 디자인 전용 버전 (standalone)
====================================

실제 모델 / 데이터 파일 없이도 바로 돌아가는 디자인 시안.
mock(랜덤) 데이터로 채워져있고, CSS·레이아웃·컴포넌트 구조만 남겨둠.

실행:
    pip install streamlit altair pandas numpy
    streamlit run app/_team_design/streamlit_design_only.py

디자인 손볼 때 이 파일에서 작업 → 나중에 메인 streamlit_app.py 에 반영.
"""
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 페이지 설정 + 딸기 핑크 톤 CSS
# ============================================================
st.set_page_config(
    page_title="딸기 가격 예측",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #fff5f7; }

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

h1, h2, h3 { color: #E63946; }

.stButton>button {
    background-color: #E63946;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #ff8fa3;
    color: white;
}

section[data-testid="stSidebar"] { background-color: #ffe5ec; }

/* selectbox 클릭 테두리 정리 */
div[data-baseweb="select"] > div {
    border-radius: 12px;
    border: none !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within {
    border: none !important;
    box-shadow: 0 0 0 2px #ffccd5 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🍓 딸기 도매가 예측 — 입도선매 의사결정 보조")
st.success("✅ (디자인 시안) — 실제 모델 연결은 메인 streamlit_app.py 에서 처리")


# ============================================================
# 사이드바: 입력
# ============================================================
with st.sidebar:
    st.header("⚙️ 예측 설정")

    market = st.selectbox(
        "도매시장",
        ["가락시장", "부산엄궁시장", "대구북부시장", "광주서부시장"],
        index=0,
    )
    item = st.selectbox("품목", ["설향", "금실", "죽향", "킹스베리"], index=0)
    grade = st.selectbox("등급", ["상품(상)", "특품(특)", "중품(중)"], index=0)
    unit = st.selectbox("단위", ["1kg 상자", "2kg 상자"], index=0)

    st.divider()

    today = pd.Timestamp.today().normalize().date()
    start = st.date_input(
        "예측 시작일",
        value=today,
        min_value=date(2020, 1, 1),
        max_value=today + timedelta(days=180),
    )
    horizon = st.slider("예측 기간 (일)", min_value=7, max_value=60, value=14, step=1)

    st.divider()
    contract_kg = st.number_input("입도선매 계약 수량 (kg)", min_value=10, max_value=10000, value=500, step=10)

    run = st.button("예측 실행", type="primary", use_container_width=True)
    st.caption(f"{market} / {item} / {grade} / {unit} / 일별 도매가 예측 · SSM(Mamba) 기반")


# ============================================================
# Mock 데이터 (디자인 시안용)
# ============================================================
rng = np.random.default_rng(42)

# 과거 60일 가격
hist_dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
hist_price = 12000 + np.cumsum(rng.normal(0, 200, 60))
hist = pd.DataFrame({"date": hist_dates, "price": hist_price})

# 예측
future_dates = pd.date_range(start=pd.Timestamp(start), periods=horizon)
pred_price = hist_price[-1] + np.cumsum(rng.normal(0, 250, horizon))
band = 400 * np.sqrt(np.arange(1, horizon + 1))
pred = pd.DataFrame({
    "date": future_dates,
    "predicted_price": pred_price,
    "lower": pred_price - band,
    "upper": pred_price + band,
})


# ============================================================
# 상단 안내 + 메트릭
# ============================================================
st.info(
    f"📌 {start.strftime('%Y년 %m월 %d일')}부터 **{horizon}일간** {market} {item} 평균 예측가는 "
    f"약 **{int(pred['predicted_price'].mean()):,}원** 입니다.  \n"
    f"95% 신뢰구간 기준 ±{int((pred['upper'] - pred['lower']).mean() / 2):,}원 변동 가능."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("평균 예측가", f"{pred['predicted_price'].mean():,.0f} 원")
c2.metric("최고가", f"{pred['predicted_price'].max():,.0f} 원")
c3.metric("최저가", f"{pred['predicted_price'].min():,.0f} 원")
revenue = pred["predicted_price"].mean() * contract_kg
c4.metric(f"예상 매출 ({contract_kg}kg)", f"{revenue:,.0f} 원")


tab1, tab2, tab3 = st.tabs(["📈 가격 예측", "🌤 기상 변수", "ℹ️ 모델 정보"])


# ------------ Tab 1: 가격 예측 ------------
with tab1:
    st.subheader("과거 가격 + 예측")

    if item == "킹스베리":
        st.success("🍓 킹스베리는 프리미엄 품종으로 높은 가격대를 형성합니다.")
    elif item == "설향":
        st.info("🍓 설향은 국내에서 가장 대중적인 딸기 품종입니다.")
    elif item == "금실":
        st.warning("🍓 금실은 높은 당도와 선명한 색상이 특징입니다.")
    elif item == "죽향":
        st.success("🍓 죽향은 향이 진하고 고급 디저트용으로 많이 사용됩니다.")

    hist_recent = hist.copy()
    hist_recent["type"] = "실제"
    hist_recent = hist_recent.rename(columns={"price": "value"})[["date", "value", "type"]]

    pred_for_chart = pred.rename(columns={"predicted_price": "value"}).copy()
    pred_for_chart["type"] = "예측"

    chart_df = pd.concat([
        hist_recent[["date", "value", "type"]],
        pred_for_chart[["date", "value", "type"]],
    ])

    line = alt.Chart(chart_df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("value:Q", title="가격 (원/1kg)"),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["실제", "예측"], range=["#222", "#E63946"]),
            legend=alt.Legend(title=""),
        ),
    )

    band_layer = alt.Chart(pred).mark_area(opacity=0.18, color="#E63946").encode(
        x="date:T",
        y="lower:Q",
        y2="upper:Q",
    )

    st.altair_chart((band_layer + line).properties(height=380), use_container_width=True)
    st.caption("음영: 95% 신뢰구간  ·  검정선: 과거 실제가  ·  핑크선: 모델 예측가")

    with st.expander("📋 예측 결과 표"):
        st.dataframe(
            pred.assign(
                predicted_price=pred["predicted_price"].round(0).astype(int),
                lower=pred["lower"].round(0).astype(int),
                upper=pred["upper"].round(0).astype(int),
            ).rename(columns={
                "date": "날짜",
                "predicted_price": "예측가",
                "lower": "하한 (95% CI)",
                "upper": "상한 (95% CI)",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ------------ Tab 2: 기상 변수 ------------
with tab2:
    st.subheader("최근 기상 (논산 인근 — 금산 관측소)")
    st.info("기온/강수량/풍속/전운량 데이터를 기반으로 딸기 가격 변동 요인을 분석합니다.")

    weather_dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    weather = pd.DataFrame({
        "date": weather_dates,
        "평균기온": rng.integers(5, 30, 60),
        "최고기온": rng.integers(15, 35, 60),
        "최저기온": rng.integers(-5, 20, 60),
        "일강수량(mm)": rng.integers(0, 25, 60),
        "평균풍속(m/s)": rng.uniform(0.5, 5, 60),
        "평균전운량(1/10)": rng.uniform(0, 10, 60),
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌡 기온 변화")
        st.line_chart(weather.set_index("date")[["평균기온", "최고기온", "최저기온"]])
    with col2:
        st.subheader("🌧 강수량")
        st.bar_chart(weather.set_index("date")[["일강수량(mm)"]])

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💨 풍속")
        st.line_chart(weather.set_index("date")[["평균풍속(m/s)"]])
    with col4:
        st.subheader("☁️ 전운량")
        st.line_chart(weather.set_index("date")[["평균전운량(1/10)"]])


# ------------ Tab 3: 모델 정보 ------------
with tab3:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            """
            ### 모델 구조
            - **구조**: Mamba (State Space Model) 기반 시계열 예측
            - **입력 윈도우**: 30일
            - **예측 horizon**: 14일 (직접 multi-step)
            - **불확실성**: 검증 잔차 std + Monte Carlo Dropout 결합 95% 신뢰구간

            ### 데이터
            - **가격**: KAMIS 가락도매 일별 도매가 (kindcode=00, 1kg 환산, 상등급)
            - **기상**: 기상청 ASOS 일자료 (금산 관측소)
            - **기간**: 2020-01 ~ 2026-04

            ### 주요 입력 변수
            - 과거 가격 (price_ffill, 7일 이동평균)
            - 평균/최고/최저 기온, 일교차
            - 일강수량, 7일 누적 강수
            - 평균 풍속, 평균 전운량, 지면온도
            - 월/요일 (계절성)

            ### 평가
            - Walk-forward validation
            - 비교: Naive baselines / Ridge / RandomForest / XGBoost / **Mamba+XGBoost 앙상블**
            """
        )

    with col_b:
        st.subheader("📊 모델 성능 (디자인 시안 — 실제값 아님)")
        m = st.container()
        m.metric("Validation MAE", "737 원")
        m.metric("Validation MAPE", "7.55 %")
        m.metric("Test MAE", "1,036 원")
        m.metric("Test MAPE", "11.55 %")
        st.caption("학습 윈도우: 196 / 검증: 49 / 테스트: 52")

        st.divider()
        st.markdown("**🏆 베스트 (별도 비교 실험)**")
        st.metric("Mamba + XGBoost 가중 앙상블 MAPE", "7.55%")
        st.caption("(단일 split 기준, 04_fair_comparison + 08_ensemble 결과)")


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "🍓 SSM 기반 딸기 도매가 예측 시스템  |  "
    "© Capstone Design 2026 — 우채연 · 김소령 · 지도교수 조민호 · 참여기업 ㈜바르카  |  "
    "Data: KAMIS · 기상청 기상자료개방포털"
)
