import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.LinearRegression import LinearRegression
import time

st.title("Linear Regression")

@st.cache_data
def generate_data():
    X = 2 * np.random.rand(100, 1)
    y = 3 + 4 * X + np.random.randn(100, 1)
    return X, y

X, y = generate_data()
x_range = np.linspace(np.min(X), np.max(X), 100)

st.markdown(""" 
Linear regression, istatistiksel bir modelleme tekniğidir. Bu yöntem, bağımlı değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi açıklamak için kullanılır. Temel olarak, bağımlı değişken ile bağımsız değişkenler arasında doğrusal bir ilişki olduğu varsayımına dayanır. Hedefi, veri setindeki bu ilişkiyi temsil eden bir doğru veya düzlemi bulmaktır. Bu doğru veya düzlem, veri noktalarına en uygun şekilde uyan ve gelecekteki tahminler için kullanılan bir model oluşturur.
""")

if 'B0' not in st.session_state:
    st.session_state.B0 = 0.0
if 'B1' not in st.session_state:
    st.session_state.B1 = 0.0


chart_1=st.empty()

with st.expander("Adjust Parameters"):
    st.session_state.B0 = st.slider("B0", min_value=0.0, max_value=10.0, value=st.session_state.B0)
    st.session_state.B1 = st.slider("B1", min_value=0.0, max_value=10.0, value=st.session_state.B1)
    st.latex(f"y = {st.session_state.B0:.2f}x + {st.session_state.B1:.2f}")

B0 = st.session_state.B0
B1 = st.session_state.B1

y_pred = B0 * x_range + B1

fig = go.Figure()
fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Data Points'))
fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Regression Line', line=dict(color='red')))
fig.update_layout(title='Linear Regression Visualization', 
                  xaxis_title='X', yaxis_title='y', 
                  margin=dict(l=20, r=20, t=40, b=20))
chart_1.plotly_chart(fig, use_container_width=True)

st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

@st.cache_resource
def model():
    results = []
    linearRegression = LinearRegression(0.01, 100)
    linearRegression.X = X
    linearRegression.Y = y
    linearRegression.W = np.zeros((X.shape[1], 1))
    linearRegression.b = 0
    linearRegression.m, linearRegression.n = X.shape
    errors = []

    for i in range(100):
        linearRegression.update_weights()
        y_predict = linearRegression.predict(x_range.reshape(-1, 1))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Data Points'))
        fig2.add_trace(go.Scatter(x=x_range, y=y_predict.reshape(-1), mode='lines', name='Regression Line', line=dict(color='red')))

        for j in range(len(X)):
            y_on_line = linearRegression.W[0] * X[j] + linearRegression.b
            fig2.add_trace(go.Scatter(x=[X[j][0], X[j][0]], y=[y[j][0], y_on_line[0]], mode='lines', line=dict(color='gray', width=1), showlegend=False))

        fig2.update_layout(title='Linear Regression Visualization', 
                           xaxis_title='X', yaxis_title='y', 
                           margin=dict(l=20, r=20, t=40, b=20))
        mse = np.mean((y - y_predict)**2)
        errors.append(mse)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=np.arange(i + 1), y=errors, mode='lines'))
        fig3.update_layout(title='Linear Regression Error', xaxis_title='Step', yaxis_title='MSE')

        results.append((fig2, mse, fig3))

    return results

models = model()

chart_2=st.empty()
chart_3=st.empty()
formule=st.empty()
step = st.slider("Step", min_value=0, max_value=99, value=0)

chart_2.plotly_chart(models[step][0], use_container_width=True)
formule.latex(f"Step {step}: Error (MSE) = {models[step][1]:.2f}")

if st.button("Run"):
    for i in range(100):
        chart_2.plotly_chart(models[i][0], use_container_width=True)
        chart_3.plotly_chart(models[i][2], use_container_width=True)
        formule.latex(f"Step {i}: Error (MSE) = {models[i][1]:.2f}")
        time.sleep(0.1)
