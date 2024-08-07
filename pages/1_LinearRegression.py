import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.LinearRegression import LinearRegression
from utils.LinearRegressionUtils import *
import time
import pandas as pd
st.set_page_config(layout="wide")
st.title("Linear Regression")

df=pd.read_csv("./datasets/CarPrice_Assignment.csv")

tab1,tab2=st.tabs(['Teori','Kod'])

with tab1:
    
    st.markdown("Veri Seti: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data")
    X=df['horsepower'].to_numpy().reshape(-1,1)
    y=df['price'].to_numpy()
    x_range = np.linspace(np.min(X), np.max(X), len(X))

    st.markdown(linear_aciklama) 
    st.markdown("***")
    st.markdown(linear_varsayımlar)
    st.markdown("***")
    st.markdown(basic_linear_formule)
    st.markdown("***")
    st.markdown(linear_closed_formule)
    st.markdown("***")
    st.markdown(linear_gradien_descent)
    st.markdown("***")
    
    st.markdown("## Örnek: ")
    
    st.dataframe(df)
    df_col=df.columns   

    X=df[st.selectbox("X:  ",df_col)].to_numpy().reshape(-1,1)
    y=df[st.selectbox("y:  ",df_col)].to_numpy().reshape(-1,1)
    x_range = np.linspace(np.min(X), np.max(X), len(X))
    
    
    st.markdown("## Normal Formül: ")
    st.latex("β₁ = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]")
    st.latex("β₀ = ȳ - β₁x̄")
    
    st.latex(f"x̄ = {X.mean()}")
    st.latex(f"ȳ = {y.mean()}")
    
    _df=pd.DataFrame()
    _df['X']=X.reshape(-1)
    _df['X mean']=X.mean()
    _df['(xi - x̄)']=X-X.mean()
    _df['y']=y.reshape(-1)
    _df['y mean']=y.mean()
    _df['(yi - ȳ)']=y-y.mean()
    _df['(xi - x̄)²']=(X-X.mean())*(X-X.mean())

    
    st.dataframe(_df,hide_index=True,use_container_width=True)
    b1 = ((X - X.mean()) * (y - y.mean())).sum() / ((X - X.mean()) ** 2).sum()
    b0 = y.mean() - b1 * X.mean()
    st.latex(f"β₁ ={b1}")
    st.latex(f"β₀ = {y.mean()} - {b1}*{X.mean()}= {b0}")
    
    if 'B0' not in st.session_state:
        st.session_state.B0 = 0.0
    if 'B1' not in st.session_state:
        st.session_state.B1 = 0.0


    chart_1=st.empty()

    with st.expander("Adjust Parameters"):
        st.session_state.B0 = st.slider("B0", min_value=0.0, max_value=1.0, value=st.session_state.B0)
        st.session_state.B1 = st.slider("B1", min_value=b1-110.0, max_value=b1+100.0, value=st.session_state.B1)
        st.latex(f"y = {st.session_state.B0:.2f}x + {st.session_state.B1:.2f}")

    B0 = st.session_state.B0
    B1 = st.session_state.B1

    y_pred = B1 * x_range + B0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Regression Line', line=dict(color='red')))
    fig.update_layout(title='Linear Regression Visualization', 
                    xaxis_title='X', yaxis_title='y')
    chart_1.plotly_chart(fig, use_container_width=True)

    st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    st.latex(f"MSE = {((y - (B1 * X + B0))**2).mean()}")    
    def model():
        results = []
        linearRegression = LinearRegression(0.000001, 100)
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
                            xaxis_title='X', yaxis_title='y')
            mse = np.mean((y - y_predict)**2)
            errors.append(mse)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=np.arange(i + 1), y=errors, mode='lines'))
            fig3.update_layout(title='Linear Regression Error', xaxis_title='Step', yaxis_title='MSE')

            results.append((fig2, mse, fig3,linearRegression))

        return results
    st.markdown("## Gradient descent: ")
    st.latex("MSE = (1/n) * Σ(yi - ŷi)²")
    st.latex("∂MSE/∂β₁ = (-2/n) * Σ(xi * (yi - ŷi))")
    st.latex("MSE/∂β₀ = (-2/n) * Σ(yi - ŷi)")
    
    
    chart_2=st.empty()
    if st.button("Run Gradient Descent"):
        models = model()
        
        chart_3=st.empty()
        formule=st.empty()
        formule2=st.empty()
        step = st.slider("Step", min_value=0, max_value=99, value=0)

        chart_2.plotly_chart(models[step][0], use_container_width=True)
        formule.latex(f"Step {step}: Error (MSE) = {models[step][1]:.2f}")

        for i in range(100):
            chart_2.plotly_chart(models[i][0], use_container_width=True)
            chart_3.plotly_chart(models[i][2], use_container_width=True)
            formule.latex(f"Step {i}: Error (MSE) = {models[i][1]:.2f}")
            formule2.latex(f"y = {models[i][-1].b} + ({models[i][-1].W[0][0]}*x)")
            time.sleep(0.1)
            
    st.markdown("***")
    st.markdown(multiple_linear_basic)
    st.markdown("***")
    st.markdown(multiple_linear_close_formule)
    st.markdown("***")
    st.markdown(multiple_linear_gd)
    st.markdown("***")
    
    with st.expander("Dataframe info"):
        st.dataframe(df,hide_index=True)
        
    x_cols=st.multiselect("X:  ",df_col,['horsepower'])
    y_col=st.selectbox("y:  ",df_col, key=2131)
    
    X=df[x_cols]
    y=df[y_col].to_numpy()

    st.markdown("## Normal Formül: ")
    st.latex("β = (X^T X)^{-1} X^T y")

    _df = pd.DataFrame()
    cols_w = {}
    # b0 = y.mean()  # Multiple linear regressionda b0 hesaplaması farklıdır
    X_np = X.to_numpy()
    X_T = X_np.transpose()
    X_T_X = np.dot(X_T, X_np)
    X_T_X_inv = np.linalg.inv(X_T_X)
    X_T_y = np.dot(X_T, y)
    beta = np.dot(X_T_X_inv, X_T_y)

    for i, col in enumerate(x_cols):
        x = df[col].to_numpy()
        _df[f'X_{col}'] = x
        _df[f'X_{col} mean'] = x.mean()
        _df[f'(xi_{col} - x̄_{col})'] = x - x.mean()
        _df[f'(xi{col} - x̄{col})²'] = (x - x.mean()) * (x - x.mean())
        cols_w[col] = beta[i]

    _df['y'] = y
    _df['y mean'] = y.mean()
    _df['(yi - ȳ)'] = y - y.mean()

    st.dataframe(_df, hide_index=True, use_container_width=True)

    for i, col in enumerate(x_cols):
        st.latex(f"β_{col} = {cols_w[col]}")
    
    
    
    y_pred = np.dot(X_np, beta)
    mse = np.mean((y - y_pred) ** 2)
    st.latex(f"MSE = {mse}")
    
    if len(x_cols)==2:
        from scipy.interpolate import griddata

        xx, yy = np.meshgrid(X_np[:, 0], X_np[:, 1])
        y_pred_2d = griddata((X_np[:, 0], X_np[:, 1]), y_pred, (xx, yy), method='linear')

        fig = go.Figure(data=[go.Scatter3d(x=X_np[:, 0], y=X_np[:, 1], z=y, mode='markers'),
                            go.Surface(x=xx, y=yy, z=y_pred_2d)])
        fig.update_layout(scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col))
        st.plotly_chart(fig)
        
    
   
    st.markdown("### SONUÇ: ")
    st.markdown(linear_end)
    st.markdown("***")
    


with tab2:
    
    for info,code in code_info:
        cell = st.empty()
        
       
        with cell.container():
            col1, col2 = st.columns(2)
            with col1:
                col1.markdown(info)
            
            with col2:
                col2.code(code, language='python')
    
        
        col1.markdown("----")
        