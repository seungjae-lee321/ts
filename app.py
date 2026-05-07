import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="TS Analyzer Pro", layout="wide")
st.title("📊 시계열 분석 및 4종 모델 성능 비교 시스템")
st.markdown("""
이 시스템은 **데이터 누수(Data Leakage)**를 방지하기 위해 학습과 검증 데이터를 엄격히 분리합니다. 
시험에서 강조된 **RMSE**와 **MAPE** 지표를 통해 모델의 안정성과 실무적 정확도를 동시에 평가합니다.
""")

# --- 사이드바: 설정 영역 ---
st.sidebar.header("📂 1. 데이터 설정")
uploaded_file = st.sidebar.file_uploader("시계열 CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_cols = list(df.columns)
    
    date_col = st.sidebar.selectbox("날짜 컬럼 선택", options=all_cols)
    target_col = st.sidebar.selectbox("예측 수치 컬럼 선택", options=[c for c in all_cols if c != date_col])
    
    # 데이터 기본 전처리
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)
    
    st.sidebar.divider()
    st.sidebar.header("🔮 2. 모델링 파라미터")
    test_size = st.sidebar.slider("테스트(검증) 데이터 비중 (%)", 5, 30, 20)
    horizon = st.sidebar.number_input("미래 예측 단계 (Horizon)", 1, 365, 30)
    
    st.sidebar.subheader("계절성 설정 (Holt-Winters)")
    season_type = st.sidebar.selectbox("진폭 변화 타입", ["mul", "add", None], index=0, help="진폭이 커지는 데이터는 'mul' 권장")
    season_period = st.sidebar.number_input("계절 주기 (S)", 2, 365, 7)

    if st.sidebar.button("🚀 전체 모델 비교 실행"):
        # 1. 데이터 분할 (시간 순서 준수 - 데이터 누수 방지)
        split_idx = int(len(df) * (1 - test_size/100))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_y = train_df[target_col]
        test_y = test_df[target_col]
        
        results = {}
        
        try:
            # 2. 모델 4종 학습 및 예측
            # (1) 단순 이동 평균 (Moving Average) - 벤치마크용
            ma_window = 7
            last_ma = train_y.iloc[-ma_window:].mean()
            results['Moving Average'] = np.full(len(test_df), last_ma)
            
            # (2) 단순 지수 평활 (Simple ES) - 수준 반영
            results['Simple ES'] = SimpleExpSmoothing(train_y, initialization_method="estimated").fit().forecast(len(test_df))
            
            # (3) 홀트 모델 (Holt) - 추세 반영
            results['Holt'] = Holt(train_y, initialization_method="estimated").fit().forecast(len(test_df))
            
            # (4) 홀트-윈터스 (Holt-Winters) - 추세 + 계절성(진폭) 반영
            results['Holt-Winters'] = ExponentialSmoothing(
                train_y, trend='add', seasonal=season_type, seasonal_periods=season_period, initialization_method="estimated"
            ).fit().forecast(len(test_df))

            # 3. 지표 산출 및 결과 대시보드
            st.subheader("🏁 모델별 성능 비교표")
            metrics_list = []
            
            for name, pred in results.items():
                mae = mean_absolute_error(test_y, pred)
                rmse = np.sqrt(mean_squared_error(test_y, pred)) # 큰 오차에 민감
                mape = np.mean(np.abs((test_y - pred) / test_y)) * 100 # 실무적 오차율
                metrics_list.append({"Model": name, "RMSE": round(rmse, 2), "MAPE(%)": f"{mape:.1f}%", "MAE": round(mae, 2)})
            
            # 성능 순으로 정렬 (RMSE 기준)
            metrics_df = pd.DataFrame(metrics_list).sort_values("RMSE")
            st.table(metrics_df.set_index("Model"))

            # 4. 인터랙티브 시각화 (Plotly)
            st.subheader("📈 예측 결과 시각적 비교")
            fig = go.Figure()
            
            # 실제값 (학습 + 테스트)
            fig.add_trace(go.Scatter(x=train_df[date_col], y=train_y, name='Train (과거)', line=dict(color='#d3d3d3')))
            fig.add_trace(go.Scatter(x=test_df[date_col], y=test_y, name='Test (실제)', line=dict(color='black', width=2)))
            
            # 모델별 예측값
            colors = ['#FFA07A', '#20B2AA', '#9370DB', '#FF4500']
            for (name, pred), color in zip(results.items(), colors):
                fig.add_trace(go.Scatter(x=test_df[date_col], y=pred, name=f'{name} 예측', line=dict(color=color, dash='dash')))

            fig.update_layout(template="plotly_white", hovermode="x unified", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. 최종 미래 예측 (가장 우수한 모델인 Holt-Winters 기준)
            st.subheader(f"🔮 향후 {horizon}단계 미래 예측")
            future_model = ExponentialSmoothing(df[target_col], trend='add', seasonal=season_type, seasonal_periods=season_period).fit()
            future_preds = future_model.forecast(horizon)
            
            last_date = df[date_col].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=df[date_col], y=df[target_col], name='현재까지 데이터'))
            fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, name='미래 예측치', line=dict(color='green', width=3)))
            st.plotly_chart(fig_future, use_container_width=True)

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}. 데이터의 길이와 계절 주기를 확인하세요.")
else:
    st.info("👈 왼쪽 사이드바에 분석할 시계열 데이터를 업로드해주세요.")
