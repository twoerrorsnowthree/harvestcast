# 딸기 가격 예측 (Mamba/SSM 기반)

논산 지역 기상 데이터와 KAMIS 가락시장 도매가격을 활용한 SSM 기반 일별 딸기 가격 예측 모델.

## 폴더 구조

```
capstone/
├── data/
│   ├── raw/
│   │   ├── weather/   # 기상청 ASOS 시간별 csv (OBS_ASOS_TIM_*.csv)
│   │   └── price/     # KAMIS 도매가 xls (HTML 형식)
│   └── processed/
│       ├── merged_daily.csv         # 기상+가격 일별 병합 (전체 기간)
│       └── merged_season_only.csv   # 출하기(11~5월) + 가격 있는 일만
├── src/
│   ├── weather_loader.py   # 시간별 ASOS → 일별 집계
│   ├── price_loader.py     # KAMIS xls 파서
│   └── preprocessing.py    # 머지 + 파생변수 생성
├── notebooks/
│   └── 01_preprocessing.py # 전처리 전체 실행 스크립트
├── figures/                # 시각화 결과
└── README.md
```

## 데이터 출처

| 구분 | 출처 | URL | 비고 |
|---|---|---|---|
| 가격 | KAMIS | kamis.or.kr | 가락시장 / 딸기 설향 / 1kg 상자 / 상등급 |
| 기상 | 기상청 ASOS | data.kma.go.kr | 금산 관측소(238), 시간별 |
| (예정) 생산량 | 통계청 KOSIS | kosis.kr | 시도/시군구 단위, 연 단위 |

## 사용법

```bash
# 1. 의존성
pip install pandas numpy matplotlib lxml html5lib openpyxl

# 2. 전처리 실행
python notebooks/01_preprocessing.py
```

새 raw 데이터(KAMIS xls, 기상 csv)를 `data/raw/` 아래 해당 폴더에 추가만 하면
스크립트가 자동으로 모두 읽어서 합친다.

## 데이터 현황 (2026-05-03 기준)

### 기상 (금산 ASOS, 시간별 → 일별 집계)
- 기간: 2021-04-01 ~ 2023-03-01 (700일)
- 변수: 평균/최고/최저기온, 일교차, 강수합계, 평균풍속, 평균습도,
  일조합계, 평균지면온도, 평균전운량

### 가격 (가락시장, 1kg 상등급, 일별)
- 보유 일수: 254일
  - 2021-04 ~ 2021-07 (출하기 말미 + 일부)
  - 2021-11 ~ 2022-06 (출하기 풀 시즌)
- **부족분**: 2022 후반 ~ 2023, 2020 이전 — KAMIS 인증 받으면 API로 일괄 수집 예정

## 모델링 메모

- 딸기 출하기는 11월 ~ 익년 5월. 비출하기(7~10월)는 가락시장에 데이터 자체가 거의 없음.
- 모델 학습 시 `merged_season_only.csv` 사용 권장 (출하기 + 가격 있는 일만 224일).
- 학습/테스트 분할은 시간 순서를 유지 (계획서: train 2021~2022, test 2023).
- 평가: MAE, RMSE, MAPE + Walk-forward validation.

## TODO

- [ ] KAMIS API 인증 받기 → 2020~2025 전체 가격 일괄 수집
- [ ] 기상 데이터도 같은 기간으로 확장 (시간별 또는 일자료)
- [ ] 부여(236) 관측소도 같이 받아서 비교
- [ ] KOSIS에서 논산 시군구 단위 재배면적/생산량 수집
- [ ] Mamba 모델 구현 + ARIMA/LSTM 베이스라인
