import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="Pro TS Analyzer", layout="wide")
st.title("📊 고도화된 시계열 분석 및 데이터 정제 시스템")

# --- 메인 가이드 (UX 개선) ---
with st.expander("📌 사용 방법 및 분석 프로세스 확인", expanded=False):
    st.markdown("""
    1. **데이터 업로드**: 사이드바에서 시계열 CSV 파일을 선택하세요.
    2. **데이터 정제**: '자동 정제' 활성 시 **결측치는 선형 보간**, **이상치는 3-Sigma** 기준으로 자동 처리됩니다.
    3. **파라미터 설정**: '수동 조절 모드'를 통해 알파($\\alpha$), 베타($\\beta$), 감마($\\gamma$) 값을 직접 튜닝할 수 있습니다.
    4. **결과 해석**: **RMSE**는 큰 오차에 민감하며, **MAPE**는 실무적인 오차율(%)을 나타냅니다.
    """)

st.markdown("---")

# --- 사이드바: 설정 영역 ---
st.sidebar.header("📂 1. 데이터 설정")
uploaded_file = st.sidebar.file_uploader("시계열 CSV 파일 업로드", type="csv")

if uploaded_file:
    # 데이터 로드
    df = pd.read_csv(uploaded_file)
    all_cols = list(df.columns)
    
    date_col = st.sidebar.selectbox("날짜 컬럼 선택", options=all_cols)
    target_col = st.sidebar.selectbox("예측 수치 컬럼 선택", options=[c for c in all_cols if c != date_col])
    
    # [데이터 전처리 시작]
    # 1. 날짜 변환 및 정렬
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    
    # 2. ★ 중요: 데이터 타입 에러 방지 (int64 -> float64 강제 변환)
    df[target_col] = df[target_col].astype(float)

    st.sidebar.divider()
    st.sidebar.header("🧹 2. 데이터 전처리 설정")
    do_cleaning = st.sidebar.checkbox("자동 결측치/이상치 정제 활성화", value=True)

    if do_cleaning:
        # 결측치 보간
        null_count = df[target_col].isnull().sum()
        if null_count > 0:
            df[target_col] = df[target_col].interpolate(method='linear')
            st.sidebar.info(f"💡 결측치 {null_count}개 보간 완료")

        # 이상치 정제 (3-Sigma 원칙)
        mean_val = df[target_col].mean()
        std_val = df[target_col].std()
        outliers = (df[target_col] < mean_val - 3 * std_val) | (df[target_col] > mean_val + 3 * std_val)
        if outliers.any():
            outlier_count = outliers.sum()
            # 평균값으로 대체 (dtype이 float이므로 에러 발생 안 함)
            df.loc[outliers, target_col] = mean_val
            st.sidebar.warning(f"🚨 이상치 {outlier_count}개 평균값 대체 완료")

    st.sidebar.divider()
    st.sidebar.header("⚙️ 3. 모델링 설정")
    manual_mode = st.sidebar.checkbox("파라미터 수동 조절 (α, β, γ)", value=False)
    
    if manual_mode:
        alpha = st.sidebar.slider("Level (α)", 0.01, 1.0, 0.3)
        beta = st.sidebar.slider("Trend (β)", 0.0, 1.0, 0.1)
        gamma = st.sidebar.slider("Seasonal (γ)", 0.0, 1.0, 0.1)
    else:
        alpha = beta = gamma = None

    season_type = st.sidebar.selectbox("계절성 진폭 타입", ["mul", "add", None], index=0)
    season_period = st.sidebar.number_input("계절 주기 (S)", 2, 365, 7)
    test_size = st.sidebar.slider("테스트 데이터 비중 (%)", 5, 50, 20)
    horizon = st.sidebar.number_input("미래 예측 단계 (Horizon)", 1, 365, 30)

    if st.sidebar.button("🚀 분석 실행", use_container_width=True):
        # 데이터 분할 (시간 순서 준수)
        split_idx = int(len(df) * (1 - test_size/100))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_y = train_df[target_col]
        test_y = test_df[target_col]
        test_dates = test_df[date_col]
        
        results = {}
        
        try:
            # 모델 학습 및 예측
            # (1) Moving Average
            results['Moving Average'] = np.full(len(test_y), train_y.iloc[-7:].mean())
            
            # (2) Simple ES
            results['Simple ES'] = SimpleExpSmoothing(train_y, initialization_method="estimated").fit(
                smoothing_level=alpha, optimized=not manual_mode
            ).forecast(len(test_y))
            
            # (3) Holt
            results['Holt'] = Holt(train_y, initialization_method="estimated").fit(
                smoothing_level=alpha, smoothing_trend=beta, optimized=not manual_mode
            ).forecast(len(test_y))
            
            # (4) Holt-Winters
            results['Holt-Winters'] = ExponentialSmoothing(
                train_y, trend='add', seasonal=season_type, seasonal_periods=season_period, initialization_method="estimated"
            ).fit(
                smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=not manual_mode
            ).forecast(len(test_y))

            # --- 결과 대시보드 ---
            st.subheader("🏁 모델 성능 비교")
            metrics_list = []
            for name, pred in results.items():
                rmse = np.sqrt(mean_squared_error(test_y, pred))
                mape = np.mean(np.abs((test_y - pred) / test_y)) * 100
                metrics_list.append({"Model": name, "RMSE": round(rmse, 2), "MAPE(%)": f"{mape:.1f}%"})
            
            st.table(pd.DataFrame(metrics_list).sort_values("RMSE").set_index("Model"))

            # --- 시각화 ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col][:split_idx], y=train_y, name='Train', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=test_dates, y=test_y, name='Actual', line=dict(color='black', width=2)))
            
            colors = ['#FFA07A', '#20B2AA', '#9370DB', '#FF4500']
            for (name, pred), color in zip(results.items(), colors):
                fig.add_trace(go.Scatter(x=test_dates, y=pred, name=name, line=dict(color=color, dash='dash')))

            fig.update_layout(template="plotly_white", hovermode="x unified", title="데이터 정제가 반영된 모델 비교")
            st.plotly_chart(fig, use_container_width=True)

            # --- 미래 예측 ---
            st.subheader(f"🔮 향후 {horizon}단계 미래 예측 (Holt-Winters)")
            future_model = ExponentialSmoothing(
                df[target_col], trend='add', seasonal=season_type, seasonal_periods=season_period, initialization_method="estimated"
            ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=not manual_mode)
            
            future_preds = future_model.forecast(horizon)
            future_dates = pd.date_range(start=df[date_col].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
            
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df[date_col], y=df[target_col], name='현재 데이터', line=dict(color='black')))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, name='미래 예측', line=dict(color='green', width=3)))
            st.plotly_chart(fig_future, use_container_width=True)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
else:
    st.info("👈 왼쪽 사이드바에서 시계열 CSV 파일을 업로드해주세요.")
