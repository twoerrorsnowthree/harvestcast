# 🍓 HarvestCast

**SSM(Mamba) 기반 딸기 도매가 예측 시스템**

KAMIS 가락도매시장 가격 + 산지 기상 데이터를 결합해 일별 가격을 예측합니다. 

입도선매 계약 시 농가·유통업체의 의사결정을 지원합니다.

🌐 **라이브 데모**: `https://harvestcast.streamlit.app` *(배포 후 URL 업데이트)*

---

## 🎯 핵심 성과

- **MAPE 8.73%** (RandomForest, 단일 split)
- KAMIS **일일동향 보고서 582건 텍스트 분석** → 시장-산지 매핑 정량 도출
- **산지 매핑 적용으로 ML 모델 정확도 35% 개선** (도메인 지식 기반 feature engineering 효과 입증)
- **Mamba walk-forward 분산 36% 감소** (±4.33% → ±2.78%, 데이터 확장 효과)

---

## 📊 모델 비교

| 모델 | MAE (원) | MAPE | 비고 |
|---|---|---|---|
| 🥇 RandomForest | 1,133 | **8.73%** | 최고 성능 |
| 🥈 XGBoost | 1,190 | 8.84% | |
| 🥉 Ridge | 1,129 | 9.51% | |
| Mamba (SSM) | 1,340 | 12.04% | 메인 모델, 데이터 확장 시 잠재력 ↑ |
| Naive lag-1 | 1,473 | 12.11% | 단순 baseline |

평균 가격 11,821원/kg 기준 ±1,030원 오차 → 입도선매 의사결정 보조 충분.

---

## 🗂 데이터 현황

| 출처 | 내용 | 기간 | 보유량 |
|---|---|---|---|
| KAMIS API | 가락도매 딸기(설향) 1kg 상등급 도매가 | 2020-01 ~ 2026-04 | 1,035일 |
| KAMIS 웹 | 4품종(설향/금실/죽향/킹스베리) | 2020 ~ 2026 | 품종별 303~972일 |
| KAMIS 도매시장 | 부산/대구/광주/대전 가격 | 2020 ~ 2026 | 시장별 636~708일 |
| KAMIS 일일동향 | 시장-산지 매핑 텍스트 보고서 | 2020 ~ 2026 | **582건** |
| 기상청 ASOS | 5개 관측소 (금산/광주/산청/진주/밀양/합천) | 2020 ~ 2026 | 지점별 2,312일 |

### 시장-산지 매핑 (일일동향 텍스트 분석)

| 시장 | 1위 산지 | 2위 | 3위 |
|---|---|---|---|
| 서울(가락) | 전남 담양 (69%) | 경남 산청 (50%) | 충남 논산 (38%) |
| 부산 | 경남 진주 (67%) | 밀양 (62%) | 양산 (29%) |
| 대구 | 경북 고령 (50%) | 밀양 (47%) | 청도 (45%) |
| 광주 | 전남 담양 (91%) | 장성 (57%) | 충남 논산 (46%) |
| 대전 | 충남 금산 (32%) | 경남 진주 (30%) | 하동 (24%) |

→ 기존 가정(가락=충남)의 오류 발견. 실제 가락 주공급원은 **전남 담양 + 경남 산청**.

---

## 🛠 기술 스택

- **Python · PyTorch** — Mamba 모델 구현
- **scikit-learn · XGBoost** — 베이스라인 + 앙상블
- **Streamlit · Altair** — 인터랙티브 대시보드
- **KAMIS · 기상청 API** — 실시간 가격·기상 데이터

---

## 📁 폴더 구조

```
capstone/
├── app/                     Streamlit 데모
│   ├── streamlit_app.py        메인 앱 (3페이지: 홈/예측/분석)
│   ├── real_predictor.py       학습된 모델 로드 + 신뢰구간
│   └── mock_predictor.py       UI 개발용 mock
├── src/                     모듈
│   ├── mamba_model.py          Mamba(SSM) 구현
│   ├── timeseries_dataset.py   시계열 윈도우 dataset
│   ├── preprocessing.py        데이터 머지 + 파생변수
│   ├── price_loader.py         KAMIS xls/csv 파서
│   ├── weather_loader.py       ASOS 시간/일자료 로더
│   ├── kamis_api.py            KAMIS Open API 클라이언트
│   ├── baselines.py            ML 베이스라인 (Ridge/RF/XGB)
│   └── evaluation.py           MAE/RMSE/MAPE + walk-forward
├── notebooks/               파이프라인 (순서대로)
│   ├── 00_fetch_kamis.py           KAMIS API 수집
│   ├── 01_preprocessing.py         전처리 + 머지
│   ├── 02_baselines.py             ML 베이스라인 학습
│   ├── 03_mamba_train.py           Mamba 학습
│   ├── 04_fair_comparison.py       단일 split 비교
│   ├── 05_walk_forward.py          walk-forward 평가
│   ├── 06_horizon7.py              1주 예측 실험
│   ├── 07_train_and_save.py        배포용 Mamba 저장
│   ├── 08_ensemble.py              Mamba+XGB 앙상블
│   ├── 09_fetch_other_markets.py   다른 4개 시장 수집
│   ├── 10_variety_comparison.py    4품종 비교
│   ├── 11_parse_kamis_reports.py   일일동향 텍스트 분석
│   └── 12_source_region_weather.py 산지 ASOS 통합
├── data/
│   ├── raw/                    수집 원본 (xls/csv)
│   └── processed/              머지 결과
├── figures/                 시각화 png
├── models/                  학습된 모델 (mamba_deploy.pt)
└── RESULTS_SUMMARY.md       상세 결과 보고서
```

---

## 🚀 사용법

### 라이브 데모
가장 빠른 방법은 [Streamlit Cloud 데모](https://harvestcast.streamlit.app) 에서 직접 사용해보기.

### 로컬 실행

```bash
git clone https://github.com/[username]/harvestcast.git
cd harvestcast
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app/streamlit_app.py
```

### 모델 재학습 (선택)

```bash
# KAMIS 인증키 설정
export KAMIS_CERT_KEY='your_key'
export KAMIS_CERT_ID='your_id'

# 전처리 → 모델 학습 → 배포
python notebooks/01_preprocessing.py
python notebooks/07_train_and_save.py
```

자세한 파이프라인은 `notebooks/` 의 00~12 순서대로 참고.

---

## ⚠️ 한계

- **시장 단일화**: 가락도매시장 단일 모델. 부산/대구/광주/대전 데이터는 확보됨, 향후 통합 가능.
- **품종 단일화**: 메인 학습은 설향. 4품종 비교 분석은 부가 결과로만 활용.
- **장기 예측 어려움**: 시계열 모델 특성상 1년 후 직접 예측은 신뢰도 낮음. 시즌별 평균/트렌드 기반 거시 추정으로 보완 가능.
- **데이터 양**: 약 1,000일 학습. Mamba 같은 시퀀스 모델은 더 많은 데이터에서 추가 향상 기대.

## 🌱 향후 확장

- **다중 시장 통합 모델**: 시장 ID one-hot + 시장별 산지 weather 매핑 (인프라 완성)
- **신품종 데이터 누적**: 킹스베리 등 신품종 본격 시계열 분석
- **GPT-4o-mini 자연어 해설** 연동 (현재는 규칙 기반 mock)
- **실시간 운영**: 매일 KAMIS API 자동 호출 + 모델 재예측 파이프라인
- **거시 예측 결합**: 시즌별 평균/트렌드 모델 → 1년 후 가격대 추정 보조

---

## 👥 팀

- **고려대학교 세종캠퍼스 컴퓨터소프트웨어학과**
  - 우채연 (2022270682) · 김소령 (2022270692)
- **지도교수**: 조민호 교수님
- **참여기업**: ㈜바르카

## 📚 데이터 출처

- [KAMIS 농산물유통정보](https://www.kamis.or.kr) — 도매가, 일일동향
- [기상청 기상자료개방포털](https://data.kma.go.kr) — ASOS 관측 자료

---

*상세 결과 분석은 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 참고*
