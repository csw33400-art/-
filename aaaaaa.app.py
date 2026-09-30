import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# -------------------------------------------------------------
# 1. 페이지 레이아웃 및 타이틀 설정
# -------------------------------------------------------------
st.set_page_config(page_title="범용 회귀 분석 대시보드", layout="wide")
st.title("📈 범용 선형 및 비선형 회귀 분석 웹 대시보드")
st.write("원하는 엑셀/CSV 데이터를 업로드하고, $X$ 값을 자유롭게 조절하여 실시간 추론 결과와 시각화 그래프를 확인해보세요.")

# -------------------------------------------------------------
# 2. 사이드바: 엑셀/CSV 파일 업로드 및 데이터 설정
# -------------------------------------------------------------
st.sidebar.header("📁 1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("엑셀(.xlsx, .xls) 또는 CSV 파일을 업로드하세요", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("성공적으로 파일 데이터를 읽었습니다!")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    # 파일이 업로드되지 않았을 때 사용하는 기본 샘플 데이터셋
    st.sidebar.info("기본 제공 샘플 데이터셋을 사용 중입니다.")
    data = {
        'X_variable': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y_variable': [2.5, 3.8, 6.1, 8.2, 11.0, 14.5, 19.2, 25.1, 33.0, 43.5]
    }
    df = pd.DataFrame(data)

# 독립변수(X) 및 종속변수(Y) 열 선택
columns = list(df.columns)
x_col = st.sidebar.selectbox("독립변수 (X) 열 지정:", columns, index=0)
y_col = st.sidebar.selectbox("종속변수 (Y) 열 지정:", columns, index=1 if len(columns) > 1 else 0)

# 결측치 제거 및 변수 행렬 변환
df_clean = df[[x_col, y_col]].dropna()
X = df_clean[[x_col]].values
Y = df_clean[y_col].values

# -------------------------------------------------------------
# 3. 사이드바: 추론 X 값 입력 제어
# -------------------------------------------------------------
st.sidebar.header("🎯 2. 실시간 추론 X값 설정")
x_max_limit = float(np.max(X)) * 2.0 if len(X) > 0 else 100.0
default_x_val = float(np.max(X)) + 1.0 if len(X) > 0 else 11.0

user_x = st.sidebar.number_input(
    "예측하고 싶은 X 값을 직접 입력하거나 조절하세요:",
    min_value=0.0,
    max_value=x_max_limit,
    value=default_x_val,
    step=0.5
)

# -------------------------------------------------------------
# 4. 회귀 모델 학습 및 추론 연산
# -------------------------------------------------------------
# (1) 선형 회귀 (y = ax + b)
linear_model = LinearRegression()
linear_model.fit(X, Y)
a_linear = linear_model.coef_[0]
b_linear = linear_model.intercept_

# (2) 비선형 회귀 (지수 모델: y = a * exp(b * x))
def nonlinear_func(x, a, b):
    return a * np.exp(b * x)

X_flat = X.flatten()
try:
    popt, _ = curve_fit(nonlinear_func, X_flat, Y, p0=[1.0, 0.1], maxfev=5000)
    a_nonlinear, b_nonlinear = popt
    nonlinear_ok = True
except Exception:
    nonlinear_ok = False

# 추론 값 계산
pred_linear = linear_model.predict(np.array([[user_x]]))[0]
if nonlinear_ok:
    pred_nonlinear = nonlinear_func(user_x, a_nonlinear, b_nonlinear)

# -------------------------------------------------------------
# 5. 수치 결과 상단 카드(Metrics) 출력
# -------------------------------------------------------------
st.subheader(f"📊 입력값 {x_col} = {user_x:.2f} 에 대한 실시간 예측 결과")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="🔴 선형 회귀 모델 예측 Y값",
        value=f"{pred_linear:.4f}",
        delta=f"방정식: y = {a_linear:.2f}x + {b_linear:.2f}"
    )

with col2:
    if nonlinear_ok:
        st.metric(
            label="🟢 비선형 회귀 모델 예측 Y값",
            value=f"{pred_nonlinear:.4f}",
            delta=f"방정식: y = {a_nonlinear:.2f} * e^({b_nonlinear:.2f}x)"
        )
    else:
        st.warning("데이터 특성상 비선형 수치 최적화 모델이 수렴하지 않았습니다.")

# -------------------------------------------------------------
# 6. 실시간 동적 시각화 그래프 렌더링
# -------------------------------------------------------------
max_x_range = max(float(np.max(X)) + 2.0, float(user_x) + 2.0)
X_dense = np.linspace(float(np.min(X)), max_x_range, 100)

Y_pred_linear_line = linear_model.predict(X_dense.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(10, 5))

# 원본 관측 데이터
ax.scatter(X, Y, color='black', s=60, zorder=5, label='Actual Data')

# 회귀선 및 회귀곡선
ax.plot(X_dense, Y_pred_linear_line, color='red', linestyle='--', linewidth=2, label='Linear Model')

if nonlinear_ok:
    Y_pred_nonlinear_curve = nonlinear_func(X_dense, a_nonlinear, b_nonlinear)
    ax.plot(X_dense, Y_pred_nonlinear_curve, color='green', linewidth=2, label='Nonlinear Model')
    # 비선형 예측 포인트 마커
    ax.scatter([user_x], [pred_nonlinear], color='green', marker='^', s=160, zorder=6, label=f'Nonlinear Pred ({pred_nonlinear:.1f})')

# 선형 예측 포인트 마커
ax.scatter([user_x], [pred_linear], color='red', marker='X', s=160, zorder=6, label=f'Linear Pred ({pred_linear:.1f})')

ax.set_title(f"Interactive Regression Analysis (Input X = {user_x:.2f})", fontsize=13)
ax.set_xlabel(x_col, fontsize=11)
ax.set_ylabel(y_col, fontsize=11)
ax.legend(loc='upper left')
ax.grid(True, linestyle=':', alpha=0.7)

st.pyplot(fig)

# -------------------------------------------------------------
# 7. 업로드된 데이터 테이블 확인
# -------------------------------------------------------------
with st.expander("📋 로드된 데이터셋 원본 보기"):
    st.dataframe(df)