import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.LinearRegression import LinearRegression
import time
st.set_page_config(layout="wide")
st.title("Linear Regression")

tab1,tab2=st.tabs(['Teori','Kod'])

with tab1:
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







code_info=[
    ["# Gerekli Kütüphanelerin İçe Aktarılması\n\nBu bölümde, sınıfımız için gerekli olan NumPy ve typing modüllerini içe aktarıyoruz:\n\n- `numpy`: Sayısal işlemler ve diziler için kullanılacak.\n- `typing`: Tip belirtmek için kullanılacak, özellikle `Tuple` tipini kullanacağız.", 
    """import numpy as np
from typing import Tuple"""],
    
    ["# LinearRegression Sınıfı ve Constructor\n\nBu bölümde, LinearRegression sınıfını ve onun constructor metodunu tanımlıyoruz:\n\n- `learning_rate`: Gradyan inişi sırasında adım boyutu.\n- `iterations`: Eğitim sırasında yapılacak iterasyon sayısı.\n- `W`: Ağırlık vektörü (başlangıçta None).\n- `b`: Bias terimi (başlangıçta None).\n- `X`: Giriş özellikleri (başlangıçta None).\n- `Y`: Hedef değişken (başlangıçta None).", 
    """class LinearRegression:
    def __init__(self, learning_rate: float, iterations: int):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.W = None
        self.b = None
        self.X = None
        self.Y = None"""],
    
    ["# Fit Metodu\n\nBu metod, modeli verilen giriş (`X`) ve hedef (`Y`) verilerine uydurmak için kullanılır:\n\n1. Veri boyutlarını alır (`m`: örnek sayısı, `n`: özellik sayısı).\n2. Ağırlıkları (`W`) sıfırlarla, bias'ı (`b`) 0 ile başlatır.\n3. Giriş ve hedef verilerini kaydeder.\n4. Belirtilen sayıda iterasyon boyunca ağırlıkları günceller.\n5. Eğitilmiş modeli döndürür.", 
    """def fit(self, X: np.ndarray, Y: np.ndarray) -> 'LinearRegression':
        self.m, self.n = X.shape
        self.W = np.zeros((self.n, 1))
        self.b = 0
        self.X = X
        self.Y = Y
        for _ in range(self.iterations):
            self.update_weights()
        return self"""],
    
    ["# Ağırlık Güncelleme Metodu\n\nBu metod, gradyan inişi algoritmasını kullanarak ağırlıkları ve bias'ı günceller:\n\n1. Mevcut ağırlıklarla tahmin yapar.\n2. Ağırlıklar için gradyanı hesaplar (`dW`).\n3. Bias için gradyanı hesaplar (`db`).\n4. Ağırlıkları ve bias'ı günceller.\n\nFormül: \n- `W = W - learning_rate * dW`\n- `b = b - learning_rate * db`", 
    """def update_weights(self) -> None:
        Y_pred = self.predict(self.X)
        dW = -2 * self.X.T.dot(self.Y - Y_pred) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db"""],
    
    ["# Tahmin Metodu\n\nBu metod, verilen giriş özelliklerine (`X`) dayanarak tahminler yapar:\n\n- Formül: `Y_pred = X * W + b`\n- Matris çarpımı kullanarak hesaplama yapar.\n- Bias terimini ekler.", 
    """def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.W) + self.b"""],
    
    ["# Parametre Alma Metodu\n\nBu metod, eğitilmiş modelin ağırlıklarını ve bias değerini döndürür:\n\n- `W`: Ağırlık vektörü\n- `b`: Bias terimi\n\nDöndürülen değer bir tuple içinde paketlenir.", 
    """def get_parameters(self) -> Tuple[np.ndarray, float]:
        return self.W, self.b"""],
    
    ["# Ortalama Kare Hatası Hesaplama Metodu\n\nBu metod, gerçek değerler (`Y_true`) ile tahmin edilen değerler (`Y_pred`) arasındaki ortalama kare hatasını hesaplar:\n\n- Formül: MSE = mean((Y_true - Y_pred)^2)\n- Farkların karesini alır.\n- Bu karelerin ortalamasını hesaplar.\n\nBu değer, modelin performansını değerlendirmek için kullanılır.", 
    """def mean_squared_error(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.mean((Y_true - Y_pred) ** 2)"""]
]

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
        