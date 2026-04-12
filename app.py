import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.frequencies import to_offset

# 페이지 설정
st.set_page_config(page_title="Universal Time Series Predictor", layout="wide")

st.title("📊 범용 시계열 분석 및 예측 시스템")
st.markdown("어떤 단변량 시계열 CSV라도 업로드하여 분석하고 미래를 예측할 수 있습니다.")

# --- 사이드바 영역 ---
st.sidebar.header("📂 1. 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일 선택", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    st.sidebar.divider()
    st.sidebar.header("🔍 2. 분석 컬럼 설정")
    
    # [오류 방지 로직] 날짜 및 숫자 컬럼 후보군 추출
    all_cols = list(df.columns)
    date_candidates = [col for col in all_cols if 'date' in col.lower() or 'time' in col.lower()]
    num_candidates = [col for col in all_cols if df[col].dtype in ['int64', 'float64']]

    # 날짜 컬럼 인덱스 안전하게 결정
    d_default_idx = 0
    if date_candidates:
        d_default_idx = all_cols.index(date_candidates[0])

    date_col = st.sidebar.selectbox("날짜 컬럼 선택", options=all_cols, index=d_default_idx)

    # 수치 컬럼 인덱스 안전하게 결정 (수치 컬럼이 없을 경우 대비)
    t_options = num_candidates if num_candidates else all_cols
    target_col = st.sidebar.selectbox("예측 수치 컬럼 선택", options=t_options, index=0)

    st.sidebar.divider()
    st.sidebar.header("🧹 3. 데이터 전처리")
    window_size = st.sidebar.slider("노이즈 제거 (이동평균)", 1, 30, 1)
    
    # 분석 기간 설정
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    min_date, max_date = df[date_col].min().to_pydatetime(), df[date_col].max().to_pydatetime()
    date_range = st.sidebar.date_input("분석 기간", [min_date, max_date])

    st.sidebar.divider()
    st.sidebar.header("🔮 4. 예측 설정")
    horizon = st.sidebar.number_input("미래 예측 단계 (Horizon)", 1, 500, 30)
    seasonality = st.sidebar.radio("계절성 패턴", ["None", "add", "mul"], horizontal=True, index=1)

    # --- 메인 분석 로직 ---
    if len(date_range) == 2:
        start_d, end_d = date_range
        mask = (df[date_col].dt.date >= start_d) & (df[date_col].dt.date <= end_d)
        filtered_df = df.loc[mask].copy()

        if st.sidebar.button("✨ 예측 실행", use_container_width=True):
            try:
                # 데이터 정제 및 리샘플링
                data = filtered_df[[date_col, target_col]].groupby(date_col).mean()
                if window_size > 1:
                    data[target_col] = data[target_col].rolling(window_size, center=True).mean().interpolate(method='linear')
                
                # 빈도 추정
                freq = pd.infer_freq(data.index) or 'D'
                data = data.asfreq(freq).interpolate(method='linear')

                # 모델 학습
                sp = 7 if len(data) > 14 else 1
                use_seasonal = seasonality if seasonality != "None" and len(data) > sp*2 else None
                
                model = ExponentialSmoothing(
                    data[target_col], trend='add', seasonal=use_seasonal, seasonal_periods=sp if use_seasonal else None
                ).fit()

                # 미래 예측값 생성 (실제값의 마지막 점과 겹치도록 설정)
                forecast_steps = model.forecast(horizon)
                last_date = data.index[-1]
                last_val = data[target_col].iloc[-1]

                # 겹치는 구간을 위한 인덱스와 값 합치기
                forecast_idx = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)
                forecast_vals = np.concatenate(([last_val], forecast_steps.values))

                # 시각화 대시보드
                st.subheader("📈 시계열 예측 분석 차트")
                fig = go.Figure()
                
                # 실제값 실선
                fig.add_trace(go.Scatter(x=data.index, y=data[target_col], name='실제 데이터 (Actual)', line=dict(color='#007BFF', width=2.5)))
                
                # 예측값 점선 (실제값 끝점에서 시작)
                fig.add_trace(go.Scatter(x=forecast_idx, y=forecast_vals, name='미래 예측 (Forecast)', line=dict(color='#FF8C00', width=2.5, dash='dash')))
                
                # 접점 마커
                fig.add_trace(go.Scatter(x=[last_date], y=[last_val], name='예측 시작점', mode='markers', marker=dict(color='#FF8C00', size=10)))

                fig.update_layout(template="plotly_white", hovermode="x unified", height=550, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # 성과 지표 출력
                mae = mean_absolute_error(data[target_col], model.fittedvalues)
                c1, c2 = st.columns(2)
                c1.metric("모델 학습 오차 (MAE)", f"{mae:.2f}")
                c2.metric("예측 빈도 (Frequency)", freq)

            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
    else:
        st.warning("사이드바에서 시작일과 종료일을 모두 선택해주세요.")
else:
    st.info("👈 왼쪽 사이드바에 분석할 시계열 CSV 파일을 업로드해주세요.")
