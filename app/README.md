# Streamlit 데모 앱

입도선매 가격 예측 데모용 Streamlit 앱.

## 실행

```bash
pip install streamlit altair
cd /Users/archive/Documents/Claude/Projects/capstone
streamlit run app/streamlit_app.py
```

브라우저가 자동으로 열림. 안 열리면 콘솔에 뜨는 `http://localhost:8501` 접속.

## 구조

- `streamlit_app.py` — 메인 앱
- `mock_predictor.py` — 가짜 예측 함수. **나중에 진짜 모델로 교체 대상**.

## 모델 인터페이스 (계약)

`predict()` 함수는 항상 다음 형태의 DataFrame 반환:

```python
predict(start_date, horizon_days, last_price=None, seed=None) -> pd.DataFrame
# columns: date, predicted_price, lower, upper
```

이 형식만 유지하면 UI 쪽 수정 없이 모델만 갈아끼울 수 있다.

## 디자인 TODO (팀원 작업)

- [ ] 색상 팔레트 (현재는 기본값) — 딸기 주제에 맞게 핑크/레드 톤
- [ ] 폰트 (한글 잘 보이게)
- [ ] 메트릭 카드 디자인
- [ ] 기상 탭에 시각화 더 풍성하게
- [ ] 모바일 반응형 확인
- [ ] 로딩 스피너 / 에러 처리
- [ ] (선택) 시장 비교 (가락시장 vs 다른 도매시장)
- [ ] (선택) "이 가격이 작년 동일 시점 대비 비싼지/싼지" 표시

## 진짜 모델 연결 (나중에)

`mock_predictor.predict` 안의 랜덤 워크 부분을 `src/mamba_model.py` 의 학습된 모델 호출로 교체.

```python
def predict(start_date, horizon_days, last_price=None, seed=None):
    model = load_trained_mamba("models/mamba_v2.pt")
    # 최근 30일 윈도우를 만들어서 model에 넣고 예측
    # ...
    return df  # 같은 형태
```
