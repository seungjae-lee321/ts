import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pmdarima as pm

# 1. 페이지 설정
st.set_page_config(page_title="Ultimate TS Analyzer", layout="wide")
st.title("📊 지능형 시계열 분석 및 통합 예측 시스템")

# --- 메인 가이드 ---
with st.expander("📌 시스템 구성 및 분석 철학 확인", expanded=False):
    st.markdown("""
    - **데이터 정제**: 선형 보간(결측치) 및 3-Sigma(이상치)를 통한 데이터 무결성 확보
    - **통계 모델**: 지수평활법(ETS) 계열과 자기회귀(Auto-ARIMA) 모델의 교차 검증
    - **최적화**: AIC 지표를 최소화하는 방향으로 Auto-ARIMA 파라미터 자동 탐색
    - **평가**: RMSE, MAE, MAPE, R²를 통한 다각도 모델 성능 평가
    """)

# --- 사이드바: 설정 영역 ---
st.sidebar.header("📂 1. 데이터 설정")
uploaded_file = st.sidebar.file_uploader("시계열 CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_cols = list(df.columns)
    date_col = st.sidebar.selectbox("날짜 컬럼 선택", options=all_cols)
    target_col = st.sidebar.selectbox("예측 수치 컬럼 선택", options=[c for c in all_cols if c != date_col])
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df[target_col] = df[target_col].astype(float)

    st.sidebar.divider()
    st.sidebar.header("🧹 2. 데이터 전처리 설정")
    do_cleaning = st.sidebar.checkbox("자동 정제 활성화", value=True)

    if do_cleaning:
        df[target_col] = df[target_col].interpolate(method='linear')
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        outliers = (df[target_col] < mean_val - 3 * std_val) | (df[target_col] > mean_val + 3 * std_val)
        df.loc[outliers, target_col] = mean_val

    st.sidebar.divider()
    st.sidebar.header("⚙️ 3. 모델링 및 파라미터")
    
    # ★ 수동 조절 로직 복구
    manual_mode = st.sidebar.checkbox("파라미터 수동 조절 모드", value=False)
    
    if manual_mode:
        st.sidebar.info("💡 직접 파라미터를 설정합니다.")
        alpha = st.sidebar.slider("Level (α) - 최근치 반영", 0.01, 1.0, 0.3)
        beta = st.sidebar.slider("Trend (β) - 추세 반영", 0.0, 1.0, 0.1)
        gamma = st.sidebar.slider("Seasonal (γ) - 계절성 반영", 0.0, 1.0, 0.1)
    else:
        st.sidebar.success("🤖 알고리즘이 최적값을 자동 탐색합니다.")
        alpha = beta = gamma = None

    test_size = st.sidebar.slider("테스트 데이터 비중 (%)", 5, 50, 20)
    horizon = st.sidebar.number_input("미래 예측 단계", 1, 365, 30)
    season_period = st.sidebar.number_input("계절 주기 (S)", 2, 365, 7)

    if st.sidebar.button("🚀 전체 모델 분석 실행", use_container_width=True):
        split_idx = int(len(df) * (1 - test_size/100))
        train_y = df.iloc[:split_idx][target_col]
        test_y = df.iloc[split_idx:][target_col]
        test_dates = df.iloc[split_idx:][date_col]
        
        results = {}
        
        try:
            # 모델들 실행 (optimized 옵션에 manual_mode 반영)
            results['Simple ES'] = SimpleExpSmoothing(train_y, initialization_method="estimated").fit(
                smoothing_level=alpha, optimized=not manual_mode
            ).forecast(len(test_y))
            
            results['Holt'] = Holt(train_y, initialization_method="estimated").fit(
                smoothing_level=alpha, smoothing_trend=beta, optimized=not manual_mode
            ).forecast(len(test_y))
            
            results['Holt-Winters'] = ExponentialSmoothing(
                train_y, trend='add', seasonal='add', seasonal_periods=season_period, initialization_method="estimated"
            ).fit(
                smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=not manual_mode
            ).forecast(len(test_y))

            with st.spinner('Auto-ARIMA 최적 파라미터 탐색 중...'):
                arima_model = pm.auto_arima(train_y, seasonal=True, m=season_period, max_p=2, max_q=2, d=1, suppress_warnings=True, stepwise=True, approximation=True)
                results[f'Auto-ARIMA{arima_model.order}'] = arima_model.predict(n_periods=len(test_y))

            # --- 결과 출력 ---
            st.subheader("🏁 모델 성능 비교 분석")
            metrics_list = []
            for name, pred in results.items():
                rmse = np.sqrt(mean_squared_error(test_y, pred))
                mae = mean_absolute_error(test_y, pred)
                mape = np.mean(np.abs((test_y - pred) / (test_y + 1e-9))) * 100
                r2 = r2_score(test_y, pred)
                metrics_list.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE(%)": f"{mape:.1f}%", "R² Score": round(r2, 3)})
            
            st.table(pd.DataFrame(metrics_list).sort_values("RMSE").set_index("Model"))

            # 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col][:split_idx], y=train_y, name='Train', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_dates, y=test_y, name='Actual', line=dict(color='black', width=2)))
            for name, pred in results.items():
                fig.add_trace(go.Scatter(x=test_dates, y=pred, name=name, line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

            # --- 미래 예측 ---
            st.subheader(f"🔮 향후 {horizon}단계 미래 예측 (Holt-Winters)")
            # 전체 데이터로 재학습
            future_model = ExponentialSmoothing(df[target_col], trend='add', seasonal='add', seasonal_periods=season_period, initialization_method="estimated").fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=not manual_mode)
            future_preds = future_model.forecast(horizon)
            future_dates = pd.date_range(start=df[date_col].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
            
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df[date_col], y=df[target_col], name='과거 데이터', line=dict(color='black')))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, name='홀트-윈터스 예측', line=dict(color='red', width=3)))
            fig_future.update_layout(template="plotly_white")
            st.plotly_chart(fig_future, use_container_width=True)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
else:
    st.info("👈 왼쪽 사이드바에서 시계열 CSV 파일을 업로드하여 분석을 시작하세요.")




