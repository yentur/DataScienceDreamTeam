import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression

######### Açıklamalar ##############

st.markdown("""
    <style>
    .custom-font-head {
        font-size:20px;
        font-weight:bold;
        font-family:'Arial';
    }
    </style>
    <style>
    .custom-font-head-mid-small {
        font-size:30px;
        font-weight:bold;
        font-family:'Arial';
    }
    </style>
    <style>
    .custom-font-write {
        font-size:18px;
        font-weight:italic;
        font-family:'Arial';
    }
    </style>
    <style>
    .custom-font-head-mid {
        font-size:50px;
        font-weight:italic;
        font-family:'Arial';
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""<div style="text-align: center; margin: 20px">
            <p class="custom-font-head-mid"> Multiple Linear Regression </p> </div>""", unsafe_allow_html=True)

st.markdown("""<p class="custom-font-write">ikiden fazla bağımsız değişkenin bağımlı değişken ile ilişkisini matematiksel olarak modellemek için kullanılan bir tekniktir. 
                Genel olarak basit doğrusal regresyon'un genişletilmiş halidir.
                </p>""", unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Çoklu doğrusal regresyon genellikle şu şekilde ifade edilir: </p>',unsafe_allow_html=True)

st.latex("Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\cdots + \\beta_n X_n ")

st.write("""

- $$ \\beta_0 $$: intercept (doğrunun Y eksenini kestiği nokta)
- $$ \\beta_i $$: Bağımsız değişkenlerin katsayıları
- $$ X_i $$: Bağımsız değişkenler

""")

st.markdown('<p class="custom-font-head">Veri Seti Oluşturma</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="custom-font-write">Bağımlı değişkenin birden fazla bağımlı değişken arasındaki ilişkiyi incelemek için küçük bir veri seti oluşturalım. Veri setini daha rahat inceleyebilme için 2 bağımsız değişken oluşturalım.</p>',
    unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Oluşturulacak veri seti büyüklüğünü giriniz.</p>', unsafe_allow_html=True)

num_samples = st.slider("Veri Seti Büyüklüğü", 2, 100, 5)

st.markdown('<p class="custom-font-write">Oluşturulacak gürültüyü giriniz.</p>', unsafe_allow_html=True)

noise = st.slider("Gürültü ekleme", 0, 100, 10)

# make_regression kullanılarak veri seti oluşturuldu. / Prepare dataset with make_regression library
X, y = make_regression(n_samples=num_samples, n_features=2, noise=noise, random_state=43)

st.markdown('<p class="custom-font-head"> Veri setini inceleyelim </p>', unsafe_allow_html=True)
# Veri setini DataFrame olarak düzenleme
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['y'] = y


col1, col2 = st.columns([3, 7])
with col1:
    st.markdown(
        '<p class="custom-font-write">Veri setimiz: </p>',
        unsafe_allow_html=True)

    st.dataframe(df, height=350)

with col2:
    st.markdown('<p class="custom-font-write">3 boyutlu Dağılım grafiği: </p>', unsafe_allow_html=True)
    #func1
    # Create a 3D scatter plot
    fig = px.scatter_3d(df, x='X1', y='X2', z='y')
    fig.update_traces(marker=dict(size=5))

    # Show the plot in Streamlit
    st.plotly_chart(fig)

######OLS################

st.markdown('<p class="custom-font-head-mid-small"> Ordinary Least Squares (OLS) </p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-write"> Matrisler kullanılarak eğim ve intercept değerleri direk hesaplanır. OLS, normal denklem adı verilen matematiksel bir formülü kullanarak doğrudan katsayılar hesaplanır.   </p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-head"> Neden matris kullanılır? </p>', unsafe_allow_html=True)

st.markdown("""<p class="custom-font-write"> 

  - Matrisler birden fazla denklemin aynı anda çözülmesini sağlar. 
  
  - İşlemleri daha hızlı ve sistematik hale getirir.  
  
  </p>""", unsafe_allow_html=True)

st.write("##### X Matrisine $$ \\beta_0 $$ değerini hesaplayabilmek için 1 eklenir.")

# Sabit terim sütunu ekleme
df.insert(0, 'Intercept', 1)

# X ve y matrislerini oluşturma
X_matrix = df[['Intercept', 'X1', 'X2']].values
y_vector = df['y'].values.reshape(-1, 1)

# X matrix ve y vector'ü görselleştirme
col1, col2 = st.columns(2)

c1, c2 = st.columns(2)

with col1:
    st.markdown('<p class="custom-font-head"> X Matrisi ve Y Vektörü    </p>', unsafe_allow_html=True)

    # X ve y matrislerinin LaTeX gösterimi
    st.latex(r'''
        \mathbf{X} = \begin{bmatrix}
        1 & X1_1 & X2_1 \\
        1 & X1_2 & X2_2 \\
        \vdots & \vdots & \vdots \\
        1 & X1_n & X2_n
        \end{bmatrix}
    ''')

with c1:
    st.latex(r'''
        \mathbf{y} = \begin{bmatrix}
        y_1 \\
        y_2 \\
        \vdots \\
        y_n
        \end{bmatrix}
    ''')


# LaTeX formatında X matrisi ve y vektörü oluşturma
def create_latex_matrix(matrix):
    latex_str = f" \\begin{{bmatrix}}\n"
    for row in matrix:
        latex_str += " & ".join(f"{value:.2f}" for value in row) + " \\\\\n"
    latex_str += "\\end{bmatrix}"
    return latex_str

with col2:
    # X ve y matrislerini görselleştirme
    st.markdown('<p class="custom-font-head"> Veri Seti ve Matrisler    </p>', unsafe_allow_html=True)

    # X ve y matrislerinin LaTeX gösterimi
    X_latex = create_latex_matrix(X_matrix)
    st.latex(X_latex)

with c2:
    y_latex = create_latex_matrix(y_vector)
    st.latex(y_latex)

st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <p style="font-size: 24px; font-weight: bold; margin: 0 0 0 10px;">
            Katsayılar (β) nasıl hesaplanır?
        </p>
    </div>
    """, unsafe_allow_html=True)

formul = r"""
$$
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$
"""


st.markdown('<p class="custom-font-write"> Hata kareleri toplamının (SSR) minimum olması durumunda en iyi denklemi elde elmiş oluruz. Bunun içinde SSR\'ın türevi alınıp 0\'a eşitlenir.  </p>',unsafe_allow_html=True)

st.latex(r" {\partial} =(\mathbf{y-Xb})^T (\mathbf{y-Xb})")

st.markdown('<p class="custom-font-write"> Buradan beta\'nın formülü elde edilir.  </p>',unsafe_allow_html=True)
st.markdown(f"""
    <div style="display: flex; justify-content: center; margin: 20px;">
        <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
            <p style="font-size: 24px; font-weight: bold;">
                 {formul} </p></div></div>
    """, unsafe_allow_html=True)

XtX = r"""
$$
\mathbf{X}^T \mathbf{X}
$$
"""

st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <p style="font-size: 24px; font-weight: bold; margin: 0;">
            {XtX}</p>
        <p style="font-size: 24px; font-weight: bold; margin: 0 0 0 10px;">
            Matrisi nasıl hesaplanır?
        </p>
    </div>
    """, unsafe_allow_html=True)

# Matris çarpımı fonksiyonu
def matrix_multiply(A, B):
    return np.dot(A, B)

col1, col2 = st.columns(2)

with col1:
    # X_T * X ve X_T * y hesaplama
    X_T = X_matrix.T
    X_T_X = matrix_multiply(X_T, X_matrix)
    X_T_y = matrix_multiply(X_T, y_vector)

    # Hesaplamaları görselleştirme
    # X_T * X ve X_T * y'nin LaTeX gösterimi
    st.latex(r'''
        \begin{bmatrix}
        \sum_{i=1}^n 1 & \sum_{i=1}^n X1_i & \sum_{i=1}^n X2_i \\
        \sum_{i=1}^n X1_i & \sum_{i=1}^n X1_i^2 & \sum_{i=1}^n X1_i X2_i \\
        \sum_{i=1}^n X2_i & \sum_{i=1}^n X1_i X2_i & \sum_{i=1}^n X2_i^2
        \end{bmatrix}
    ''')

with col2:
    # X_T * X ve X_T * y'nin LaTeX gösterimi
    X_T_X_latex = create_latex_matrix(X_T_X, )

    st.latex(X_T_X_latex)

Xty = r"""
$$
\mathbf{X}^T \mathbf{y}
$$
"""

st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <p style="font-size: 24px; font-weight: bold; margin: 0;">
            {Xty}</p>
        <p style="font-size: 24px; font-weight: bold; margin: 0 0 0 10px;">
            Matrisi nasıl hesaplanır?
        </p>
    </div>
    """, unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.latex(r'''
         \begin{bmatrix}
        \sum_{i=1}^n y_i \\
        \sum_{i=1}^n X1_i y_i \\
        \sum_{i=1}^n X2_i y_i
        \end{bmatrix}
    ''')

with c2:
    X_T_y_latex = create_latex_matrix(X_T_y)
    st.latex(X_T_y_latex)

# Ters matris hesaplama
def invert_matrix(matrix):
    return np.linalg.inv(matrix)

X_T_X_inv = invert_matrix(X_T_X)

beta = matrix_multiply(X_T_X_inv, X_T_y)

st.markdown("""
   #### Matris tersi ve diğer matris hesaplamalarını bu [linkten](https://www.wikihow.com.tr/3x3-Matrisin-Tersi-Nas%C4%B1l-Al%C4%B1n%C4%B1r) bulabilirsiniz. 
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # (X_T * X)^(-1) ve β'nın LaTeX gösterimi
    st.latex(r'''
        (\mathbf{X}^T \mathbf{X})^{-1} = \text{ters matris}
    ''')

with col2:
    # (X_T * X)^(-1) ve β'nın LaTeX gösterimi
    X_T_X_inv_latex = create_latex_matrix(X_T_X_inv)
    st.latex(X_T_X_inv_latex)

st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <p style="font-size: 24px; font-weight: bold; margin: 0 0 0 10px;">
           Elde edilen matrislerden β hesaplanır. 
        </p>
    </div>
    """, unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    # Katsayıların LaTeX gösterimi
    st.latex(r'''
    \begin{bmatrix}
        \beta_0 \\
        \beta_1 \\
        \beta_2
        \end{bmatrix}
    ''')

with c2:
    beta_latex = create_latex_matrix(beta)
    st.latex(beta_latex)

def val_matris(beta):
    matrix = []
    for row in beta:
        for val in row:
            matrix.append(val)
    return matrix

matris = val_matris(beta)

last_formula = f""" y = {matris[0]:.2f} + {matris[1]:.2f}X1 + {matris[2]:.2f}X2 
"""

st.markdown(f"""
    <div style="display: flex; justify-content: center; margin: 20px;">
        <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
            <p style="font-size: 24px; font-weight: bold; margin: 0;">
                {last_formula}  </p></div></div>
    """, unsafe_allow_html=True)

# Tahmin yapma
def predict(X, beta):
    return np.dot(X, beta)

y_pred = predict(X_matrix, beta)

# Mean Squared Error (MSE) hesaplama
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred.flatten())**2)

mse = mean_squared_error(y_vector, y_pred)

# Performansı LaTeX ile gösterme


st.markdown('<p class="custom-font-head-mid-small"> Scikit-learn Kütüphanesi  </p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-write"> Linear Regresyon modeli sonucu elde edilen en iyi eğim ve intercept bularak doğrusal deklem elde edildi. </p>', unsafe_allow_html=True)

model = LinearRegression()
model.fit(X, y)
X1 = model.coef_[0]
X2 = model.coef_[1]
intercept = model.intercept_

st.markdown(f"""
    <div style="display: flex; justify-content: center; margin: 20px;">
        <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 10px;">
            <p style="font-size: 24px; font-weight: bold; margin: 0;">
                y = {intercept:.2f} + {X1:.2f}X1 + {X2:.2f}X2  </p></div></div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <p style="font-size: 24px; font-weight: bold; margin: 0 0 0 10px;">
           Model Performansı 
        </p>
    </div>
    """, unsafe_allow_html=True)

st.latex(r'''
    \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
''')

st.write(f"Mean Squared Error (MSE): {mse}")




