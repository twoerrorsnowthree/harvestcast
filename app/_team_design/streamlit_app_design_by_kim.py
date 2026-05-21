"""
Streamlit 데모 앱

실행:
    pip install streamlit altair
    cd 파일 들어있는 위치
    streamlit run app/streamlit_app.py

이 파일은 시작점이고, 디자인/세부는 채워야 함. 핵심:
    - mock_predictor.predict() 가 가짜 결과 생성 → 나중에 진짜 모델로 교체
    - UI 구조는 그대로 두고 세부 디자인만 손봐도 됨
"""
from datetime import date, timedelta
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 가짜 과거 데이터 생성
dates = pd.date_range(end=pd.Timestamp.today(), periods=60)

hist = pd.DataFrame({
    "date": dates,
    "price": np.random.randint(10000, 18000, 60)
})

# 가짜 예측 데이터 생성
future_dates = pd.date_range(start=pd.Timestamp.today(), periods=14)

pred = pd.DataFrame({
    "date": future_dates,
    "predicted_price": np.random.randint(12000, 20000, 14),
    "lower": np.random.randint(10000, 14000, 14),
    "upper": np.random.randint(18000, 22000, 14),
})

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="딸기 가격 예측",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

.main {
    background-color: #fff5f7;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

h1, h2, h3 {
    color: #E63946;
}

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

section[data-testid="stSidebar"] {
    background-color: #ffe5ec;
}
/* selectbox 클릭 테두리 제거 */
div[data-baseweb="select"] > div {
    border-radius: 12px;
    border: none !important;
    box-shadow: none !important;
}

/* 클릭 시 빨간 테두리 제거 */
div[data-baseweb="select"] > div:focus-within {
    border: none !important;
    box-shadow: 0 0 0 2px #ffccd5 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🍓 딸기 도매가 예측 — 입도선매 의사결정 보조")



# ============================================================
# 사이드바: 입력
# ============================================================
with st.sidebar:
    st.header("⚙️ 예측 설정")

    market = st.selectbox("도매시장", ["가락시장", "부산엄궁시장", "대구북부시장", "광주서부시장"], index=0)
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
# 데이터 로드
# ============================================================

import numpy as np

# 4년치 날짜 생성
dates = pd.date_range(start="2022-01-01", end="2025-12-31")

# 품종 목록
varieties = ["설향", "금실", "죽향", "킹스베리"]

data = []

for date in dates:
    for variety in varieties:

        month = date.month

        # 계절별 가격 변화
        if month in [12, 1, 2]:
            base_price = 22000
        elif month in [3, 4, 5]:
            base_price = 18000
        elif month in [6, 7, 8]:
            base_price = 13000
        else:
            base_price = 16000

        # 품종별 가격 차이
        if variety == "킹스베리":
            base_price += 5000
        elif variety == "금실":
            base_price += 2000

        # 랜덤 변동
        price = base_price + np.random.randint(-1500, 1500)

        # 시장별 가격 차이
        if market == "부산엄궁시장":
            price += 1000

        elif market == "대구북부시장":
            price -= 500

        elif market == "광주서부시장":
            price += 1500

        data.append([
            date,
            variety,
            price
        ])

# 데이터프레임 생성
hist = pd.DataFrame(
    data,
    columns=["date", "variety", "price"]
)

last_price = float(hist["price"].iloc[-1])
hist = hist[hist["variety"] == item]

# ============================================================
# 예측 실행
# ============================================================

# 가짜 예측 데이터
future_dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon)

pred = pd.DataFrame({
    "date": future_dates,
    "predicted_price": np.random.randint(12000, 20000, horizon),
    "lower": np.random.randint(10000, 14000, horizon),
    "upper": np.random.randint(18000, 22000, horizon),
})

# ============================================================
# 평가 지표 계산
# ============================================================

# 실제값과 예측값 비교
actual = hist.tail(horizon)["price"].values

predicted = pred["predicted_price"].values[:len(actual)]

# MAE 계산
mae = mean_absolute_error(actual, predicted)

# RMSE 계산
rmse = mean_squared_error(actual, predicted) ** 0.5

# MAPE 계산
mape = (abs((actual - predicted) / actual).mean()) * 100

# ============================================================
# 메인: 4개 메트릭 + 차트
# ============================================================
st.info(
    f"""
    📌 오늘 {market} {item} 평균 도매가는 약 {int(pred['predicted_price'].mean()):,}원입니다.

    최근 기온 및 강수량 변화로 인해 가격 변동성이 증가하는 추세입니다.
    """
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

    # 과거 60일 + 예측 horizon
    hist_recent = hist.tail(60).copy()
    hist_recent["type"] = "actual"
    hist_recent = hist_recent.rename(columns={"price": "value"})[["date", "value", "type"]]

    pred_for_chart = pred.rename(columns={"predicted_price": "value"})
    pred_for_chart["type"] = "predicted"

    chart_df = pd.concat([
        hist_recent[["date", "value", "type"]],
        pred_for_chart[["date", "value", "type"]],
    ])

    line = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.Y("value:Q", title="가격 (원/1kg)"),
        color=alt.Color("type:N", scale=alt.Scale(domain=["actual", "predicted"], range=["#222", "#E63946"])),
    )

    band = alt.Chart(pred).mark_area(opacity=0.2, color="#E63946").encode(
        x="date:T",
        y="lower:Q",
        y2="upper:Q",
    )

    st.altair_chart((band + line).properties(height=380), use_container_width=True)

    with st.expander("예측 결과 표"):
        st.dataframe(
            pred.assign(
                predicted_price=pred["predicted_price"].round(0).astype(int),
                lower=pred["lower"].round(0).astype(int),
                upper=pred["upper"].round(0).astype(int),
            ),
            use_container_width=True,
            hide_index=True,
        )


# ------------ Tab 2: 기상 변수 ------------
with tab2:
    st.subheader("최근 기상 (논산 인근 - 금산 관측소)")
    st.info("기온, 강수량, 습도 데이터를 기반으로 딸기 가격 변동 요인을 분석합니다.")

    # 가짜 기상 데이터 생성
    weather_dates = pd.date_range(end=pd.Timestamp.today(), periods=horizon)

    weather_df = pd.DataFrame({
        "date": weather_dates,
        "평균기온": np.random.randint(10, 30, horizon),
        "최고기온": np.random.randint(20, 35, horizon),
        "최저기온": np.random.randint(0, 15, horizon),
        "강수량": np.random.randint(0, 20, horizon),
        "습도": np.random.randint(40, 90, horizon)
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡 기온 변화")
        st.line_chart(
        weather_df.set_index("date")[["평균기온", "최고기온", "최저기온"]]
        )

    with col2:
        st.subheader("🌧 강수량")
        st.bar_chart(
            weather_df.set_index("date")[["강수량"]]
        )

    st.subheader("💧 평균 습도")

    st.line_chart(
        weather_df.set_index("date")[["습도"]]
    )
    

# ------------ Tab 3: 모델 정보 ------------
with tab3:
    st.markdown(
        """
        ### 모델
        - **구조**: Mamba (State Space Model) 기반 시계열 예측
        - **입력 윈도우**: 30일
        - **예측 방식**: 1-step-ahead (반복 호출로 multi-step 예측)
        - **데이터**: KAMIS 가락시장 도매가 + 기상청 ASOS (금산 관측소)
        - **학습 기간**: 2020-2024
        - **검증 기간**: 2025-2026

        ### 주요 입력 변수
        - 과거 가격 (price_ffill, 7일 이동평균)
        - 평균/최고/최저 기온, 일교차
        - 일강수량, 7일 누적 강수
        - 평균 습도, 일조시간
        - 월/요일 (계절성)

        ### 평가 지표
        - MAE / RMSE / MAPE
        - Walk-forward validation
        """
    )
    st.divider()

    st.subheader("📊 모델 평가 지표")

    m1, m2, m3 = st.columns(3)

    m1.metric("MAE", f"{mae:,.0f}")
    m2.metric("RMSE", f"{rmse:,.0f}")
    m3.metric("MAPE", f"{mape:.2f}%")


# ============================================================
# 하단 footer
# ============================================================
st.divider()
st.caption(
    "🍓 AI 기반 딸기 도매가 예측 시스템 | "
    "© Capstone Design 2026 — 우채연 · 김소령 · 지도교수 조민호 · 참여기업 ㈜바르카  |  "
    "Data Source: KAMIS · 기상청 기상자료개방포털"
)
