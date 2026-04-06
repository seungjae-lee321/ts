import streamlit as st
import pandas as pd
import numpy as np

# 앱의 제목을 설정합니다.
st.title("간단한 Streamlit 앱 예시 🎈")

# 텍스트 추가
st.write("Streamlit을 사용하면 데이터 앱을 쉽고 빠르게 만들 수 있습니다!")

# 사이드바 구성
st.sidebar.header("설정 패널")
num_points = st.sidebar.slider("데이터 포인트 수", min_value=10, max_value=200, value=50)

# 가상 데이터 생성
st.subheader("무작위 데이터 라인 차트")
chart_data = pd.DataFrame(
    np.random.randn(num_points, 3),
    columns=['A', 'B', 'C']
)

# 라인 차트 그리기
st.line_chart(chart_data)

# 사용자 입력 및 버튼 상호작용
st.subheader("상호작용 테스트")
user_input = st.text_input("이름을 입력하세요", "홍길동")

if st.button("인사하기 👋"):
    st.success(f"안녕하세요, {user_input}님! 반갑습니다.")
