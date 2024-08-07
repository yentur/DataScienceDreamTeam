import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
from models.LinearRegression import LinearRegressionModel

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
            <p class="custom-font-head-mid"> Linear Regression </p> </div>""", unsafe_allow_html=True)

st.markdown("""<p class="custom-font-write">İki veya daha fazla değişkenin birbiri ile ilişkisini matematiksel olarak modellemek için kullanılan bir tekniktir. 
                X bağımsız ve y bağımlı değişkenlerin birbiri ile ilişkisini açıklamaya çalışır.
                </p>""", unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Doğrusal regresyon genellikle şu şekilde ifade edilir: </p>',unsafe_allow_html=True)

st.latex("Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\cdots + \\beta_n X_n ")

st.write("""

- $$ \\beta_0 $$: intercept (doğrunun Y eksenini kestiği nokta)
- $$ \\beta_i $$: Bağımsız değişkenlerin katsayıları
- $$ X_i $$: Bağımsız değişkenler


""")

st.markdown('<p class="custom-font-head-mid">Basit Linear Regresyon </p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Burada basit (tek değişkenli) Linear Regresyon eğrisinin nasıl oluşturulduğunu adım adım öğreneceğiz. Formülümüz :</p>', unsafe_allow_html=True)

st.latex("Y = \\beta_0 + \\beta_1 X_1 ")

#########Veri Seti Oluşturma / Prepare dataset ##############

st.markdown('<p class="custom-font-head">Veri Seti Oluşturma</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="custom-font-write">İki değişken arasındaki ilişkiyi incelemek için küçük bir veri seti oluşturalım.</p>',
    unsafe_allow_html=True)

st.markdown('<p class="custom-font-write">Oluşturulacak veri seti büyüklüğünü giriniz.</p>', unsafe_allow_html=True)

num_samples = st.slider("Veri Seti Büyüklüğü", 5, 1000, 10)

# make_regression kullanılarak veri seti oluşturuldu. / Prepare dataset with make_regression library
X, y = make_regression(n_samples=num_samples, n_features=1, noise=20, random_state=43)

st.markdown('<p class="custom-font-head"> Veri setini inceleyelim </p>', unsafe_allow_html=True)
df = pd.DataFrame(X, columns=['x'])
df['y'] = y
model = LinearRegressionModel(df)

col1, col2 = st.columns([3, 7])
with col1:
    st.markdown(
        '<p class="custom-font-write">Veri setimiz: </p>',
        unsafe_allow_html=True)

    st.dataframe(df, height=350)

with col2:
    st.markdown('<p class="custom-font-write">Dağılım grafiği: </p>', unsafe_allow_html=True)
    X_label = "Bağımsız Değişken (X)"
    y_label = "Bağımlı Değişken (y)"
    title = "Bağımlı ve bağımsız değişkenler için Dağılım grafiği"
    model.scatter_plot(df, 'x', 'y', X_label, y_label, title)

######### Belirlenen Eğimlerle Linear Regresyon Eğrisi Oluşturma ##############

st.markdown('<p class="custom-font-head"> 3 adım\'da Linear Regresyon eğrisini oluşturma </p>', unsafe_allow_html=True)

st.markdown('<p class="custom-font-head"> Adım 1:  </p>', unsafe_allow_html=True)

st.markdown("""<p class='custom-font-write'>
                    Ortalama y değeri ile ilk çizgimiz oluşturulur ve gerçek y değeri ile tahmin edilen y değerlerinin farkının karesi hesaplanır ve bunalar toplanır.
                    Bu hata karelerinin toplamı (SSR) olarak adlandırılır.    </p>""", unsafe_allow_html=True)

st.markdown('<p class="custom-font-head"> Sum of Squared Residuals (SSR) </p>', unsafe_allow_html=True)

st.markdown(r"""<p class='custom-font-write'> SSR formülü şu şekildedir:

$$
{SSR} = \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
</p>""", unsafe_allow_html=True)

st.write("""

- $$  y_i  $$: Veri setindeki gerçek değerler.
- $$  \\hat{y}_i  $$: Denklem tarafından tahmin edilen değerler.
- $$  n  $$: Elimizdeki veri sayısı.
""")

slope = st.slider("Çizgi grafiğinin eğimi", -50.0, 50.0, 10.0)

model.scatter_plot_ssr(slope, True)

col1, col2 = st.columns([4, 6])

with col1:
    intercept = model.scatter_plot_ssr(slope, False)
    sr = model.cal_sr(slope, intercept)
    df['FarkKaresi'] = sr
    st.dataframe(df, height=100)

with col2:
    SSR = sr.sum()
    st.write(f"Seçilen eğime göre artık değerlerin toplamı (SSR) : {SSR:.2f}")

st.markdown('<p class="custom-font-head"> Adım 2:  </p>', unsafe_allow_html=True)

st.markdown("""<p class='custom-font-write'>
                    Bir çok eğim ile SSR değerleri hesaplanır ve bir grafik çizilir. Bu grafikteki minumum SSR değerini oluşturan intecept ve eğim verimizin linear regresyon eğrisini oluşturmak için kullanılır.    </p>""",
            unsafe_allow_html=True)

number = st.number_input("Kaç eğim hesaplanacağını giriniz:", min_value=10, step=1)
min_slope = st.number_input("Minimum eğim değerini giriniz:", value=0, max_value=100, step=1)
max_slope = st.number_input("Maximum eğim değerini giriniz: ", value=5, min_value=5, step=1)
st.write("Eğime göre hesaplanan SSR değerleri:")

slopes, SSR_df = model.SSR_slope(min_slope, max_slope, number)
x_label = "Eğim"
y_label = "SSR"
title = "SSR vs Eğim"
model.scatter_plot(SSR_df, 'Eğim', 'SSR', x_label, y_label, title)

col1, col2 = st.columns([4, 6])

with col1:
    st.dataframe(SSR_df, height=200)

with col2:
    # Minimum SSR değerini bulma
    SSR_min = SSR_df['SSR'].min()
    st.markdown(f'<p class="custom-font-write"> Minumum SSR değeri : {SSR_min:.2f} </p>',
                unsafe_allow_html=True)

    # Minimum SSR değerine sahip olan slope'u bulma
    slope_min = SSR_df[SSR_df['SSR'] == SSR_min]['Eğim'].values[0]
    intercept_min = df['y'].mean() - slope_min * df['x'].mean()
    st.markdown(f'<p class="custom-font-write"> Minumum SSR değerindeki eğim : {slope_min:.2f} </p>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="custom-font-write"> Minumum SSR değerindeki intercept : {intercept_min:.2f} </p>',
                unsafe_allow_html=True)

st.markdown('<p class="custom-font-head"> Oluşturulan Linear Regresyon formülü </p>', unsafe_allow_html=True)

st.markdown(
    f"""
        <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
        <p style="font-size: 24px; font-weight: bold;">y = {slope_min:.2f}x + {intercept_min:.2f}</p>
    </div>
        """, unsafe_allow_html=True)

######### Scikit-learn kütüphanesi'nin Oluşturduğu Eğim ##############

st.markdown("""<div style="text-align: center; margin: 20px">
            <p class="custom-font-head-mid"> Scikit-learn Kütüphanesi </p> </div>""", unsafe_allow_html=True)

st.markdown('<p class="custom-font-write"> Python\'da kullanılan popüler makine öğrenmesi kütüphanesidir. </p>',
            unsafe_allow_html=True)

st.markdown('<p class="custom-font-write"> Basit kullanımı şu şekildedir: </p>', unsafe_allow_html=True)

code = '''
from sklearn.linear_model import LinearRegression
# Modeli oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X, y)

# Eğim (slope) ve y-kesimini (intercept) alma
slope = model.coef_[0]
intercept = model.intercept_

'''

st.code(code, language='python')

st.markdown('<p class="custom-font-write"> Linear Regresyon modeli sonucu elde edilen en iyi eğim ve intercept bularak doğrusal deklem elde edildi. </p>', unsafe_allow_html=True)

s, i = model.fit()

# Tablo verileri
data_table = {
    "Açıklama": ["En düşük SSR değeri ile elde edilen denklem", "Sklearn kütüphanesinin oluşturduğu denklem"],
    "Denklem": [f"y = {slope_min:.2f}x + {intercept_min:.2f}", f"y = {s:.2f}x + {i:.2f}"]
}

# DataFrame oluşturma
data_table = pd.DataFrame(data_table)

# Tabloyu Streamlit ile yazdırma
st.subheader("Denklemleri Karşılaştırma")
st.table(data_table)

st.markdown(
    '<p class="custom-font-write"> Elde ettiğimiz denklem ve kütüphaneden elde edilen denklem ne kadar hesaplanan eğimi arttırsakda tam olarak aynı denklemi elde edemiyoruz. </p>',
    unsafe_allow_html=True)

st.markdown("""<div style="text-align: center">
<p class='custom-font-head'> Neden ? </p> </div>""", unsafe_allow_html=True)

st.markdown("""
<p class='custom-font-write'> 
1. Burada kullandığımız yöntem hata kareler yöntemi (SSR) fakat bu kütüphane genellikle ortalama kareler hatası (MSE) ya da ortalama mutlak hata (MAE) yöntemini kullanır. 
</p>""", unsafe_allow_html=True)

st.markdown("""
<p class='custom-font-write'>  
2. Elde ettiğimiz denklemdeki eğim ve intercept hesaplamaları optimizasyon sürecinden geçmez bunun yerine belirlediğimiz bir eğim üzerinden hesaplama yaptığımız için en iyi denklemi vermeyebilir.
</p>""", unsafe_allow_html=True)

st.markdown("""
<p class='custom-font-write'> 
3. Eğimleri ve kaç eğim hesaplayacağımızı biz belirlediğimiz için yeterince hassas olmayabilir.
</p>""", unsafe_allow_html=True)

######### Gradient Descent Yöntemi ile sıfırdan denklemi elde etme ##############
st.markdown("""<div style="text-align: center">
<p class='custom-font-head'> Denklemimizi ne iyileştirir ? </p> </div>""", unsafe_allow_html=True)

st.markdown("""<p class='custom-font-write'> Gradient descent fonksiyonu ile katsayılaarın belirlenmesi bu denklemdeki katsayıların maliyet fonksiyonu ile optimize edilerek elde edilmesini sağlar.
                                             Bu fonksiyon belirli bir öğrenme oranı (learning rate) ve iterasyon sayısı (iterations) ile en iyi sonuç veren katsayıları elde etmeyi amaçlar.       
</p>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align: center">
            <p class="custom-font-head"> Maliyet Fonksiyonu (Cost Function) </p> </div>""", unsafe_allow_html=True)

st.latex(r"""
    J(\beta_0, \beta_1) = \frac{1}{2m} \sum_{i=1}^{m} \left( y_i - (\beta_0 + \beta_1 x_i) \right)^2
    """)

st.write("""<p class="custom-font-write">

- $$ J(\\beta_0, \\beta_1) $$: Maliyet fonksiyonu
- $$ m $$: Toplam veri sayısı
- $$ y_i $$: Gerçek değerler
- $$ \\beta_0 $$: Y-kesimi (intercept)
- $$ \\beta_1 $$: Eğim (slope)
- $$ x_i $$: Bağımsız değişkenler
    </p>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align: center">
            <p class="custom-font-head"> Gradient Descent Algoritması </p> </div>""", unsafe_allow_html=True)

st.markdown("""
            <p class="custom-font-write"> 1. Algoritma ilk olarak belirli bir eğim ve intercept değeri alarak başlar. 
                                          Bu değerler 0'da ya da herhangi bir rastgele sayı da olabilir. Biz fonksiyonumuzda 0'dan başlattık.
             </p> """, unsafe_allow_html=True)

st.markdown("""
            <p class="custom-font-write"> 2. Her bir veri için maliyet fonksiyonu hesaplanır. Maliyet fonksiyonlarının türevi ile parametrelerin nasıl değişmesi gerektiği bulunur.

             </p> """, unsafe_allow_html=True)

st.markdown("""
            <p class="custom-font-write"> 3. Maliyet fonksiyonu , toplam hata karelerinin ortalaması olarak hesaplanır ve bir liste içine atılır.
             </p> """, unsafe_allow_html=True)

st.markdown("""
            <p class="custom-font-write"> 4. Oluşturulan bu türevler öğrenme katsayısı ile çarpılarak eski parametre değerinden çıkarılarak yeni parametreler elde edilir. 
             </p> """, unsafe_allow_html=True)

st.markdown("""
            <p class="custom-font-write"> 5. Bu hesaplama her bir iterasyon için gerçekleştirilir.
             </p> """, unsafe_allow_html=True)

code = '''
def gradient_descent(X, y, learning_rate, iterations):
    #Başlangıç değerleri
    slope, intercept = 0, 0
    n = len(y)
    #Maliyet fonksiyonu listesi
    cost_list = []
    # iterasyon döngüsü başlatılır
    for _ in range(iterations):
        slope_gradient = 0
        intercept_gradient = 0
        cost = 0
        for i in range(n):
            x = X[i, 0]
            y_pred = slope * x + intercept
            error = y[i] - y_pred
            #Hata kareler toplamı
            cost += error ** 2
            #Gradyanlar, MSE'nin eğim ve intercept için türevleridir
            slope_gradient += -(2 / n) * x * error
            intercept_gradient += -(2 / n) * error

        #Maliyet fonksiyonu, hata karelerinin ortalaması
        cost = cost / (2 * n)
        cost_list.append(cost)
        if cost <= 1e-6:
            break
        #eğim ve intercept, hesaplanan gradyanlar ve learning_rate ile güncellenir
        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient

    return slope, intercept, cost_list

'''

st.code(code, language='python')

# Parametreler
learning_rate = st.number_input("Öğrenme Oranı:", value=0.1, step=0.01, format="%.2f")
iterations = st.number_input("İterasyon Sayısı:", value=1000, step=100)

slope_gd, intercept_gd, cost_list = model.gradient_descent(learning_rate, iterations)

st.markdown('<p class="custom-font-head"> Gradient Descent ile Oluşturulan Linear Regresyon formülü </p>',
            unsafe_allow_html=True)
st.markdown(
    f"""
        <div style="text-align: center; border: 5px solid red; border-radius: 30px; padding: 5px; margin: 10px;">
        <p style="font-size: 24px; font-weight: bold;">y = {slope_gd:.2f}x + {intercept_gd:.2f}</p>
    </div>
        """,
    unsafe_allow_html=True
)

model.plot_cost(cost_list)

# Tablo verileri
data_table2 = {
    "Açıklama": ["En düşük SSR değeri ile elde edilen denklem", "Sklearn kütüphanesinin oluşturduğu denklem",
                 "Gradient Descent optimizasyonu ile oluşturulan denklem"],
    "Denklem": [f"y = {slope_min:.2f}x + {intercept_min:.2f}", f"y = {s:.2f}x + {i:.2f}",
                f"y = {slope_gd:.2f}x + {intercept_gd:.2f}"]
}

# Tabloyu Streamlit ile yazdırma
st.subheader("Denklemleri Karşılaştırma")
st.table(data_table2)

st.markdown('<p class="custom-font-write"> Böylece sklearn kütüphanesi ile aynı denkleme ulaşmış olduk. </p>', unsafe_allow_html=True)

st.markdown("""<div style="text-align: center">
<p class='custom-font-head'> Hangi denklem daha iyi ? </p> </div>""", unsafe_allow_html=True)

######### Model Değerlendirme ##############
st.markdown('<p class="custom-font-head"> Model değerlendirme metrikleri: </p>', unsafe_allow_html=True)

st.markdown("""<p class='custom-font-write'> <b>1. R-kare :</b> Modelimizin doğruluğunu (accuracy) ölçer.  
Kareler toplamı ve artık karelerin toplamının bölünmesi ile elde edilir. 0 ile 1 arasında değer alır ve 1'e yakın olması modeli ne kadar iyi açıkladığını gösterir.
 Eğer negatif bir değer alırsa modelin yalış olduğunu gösterir. </p>""", unsafe_allow_html=True)

st.latex(r"""
R^2 = 1 - \frac{\sum (y_i - \hat{y_i})^2}{\sum (y_i - \bar{y})^2}
""")

score1 = model.r2_cal(slope_min, intercept_min)

score2 = model.r2_cal(slope_gd, intercept_gd)

st.markdown(f"<p class='custom-font-write'> En küçük SSR yöntemi ile elde edilen denklemin R-kare skoru: <b>{score1[0]}</b> </p>", unsafe_allow_html=True)

st.markdown(f"<p class='custom-font-write'> Gradient Descent yöntemi ile elde edilen denklemin R-kare skoru: <b>{score2[0]}</b> </p>", unsafe_allow_html=True)

st.markdown("""<p class='custom-font-write'> <b>2. Mean Square Error (MSE) :</b> Gerçek değerler ile tahmin edilen değerler arasındaki farkın karasinin toplamıdır. 
                Böylece tahmin edilen değerler ile gerçek değerlere ne kadar yakın olduğunu hesaplar. Her zaman pozitif değer alır ve 0'a ne kadar yakınsa model gerçeğe o kadar yakın demektir.  </p>""", unsafe_allow_html=True)

st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

mse1 = model.mse_cal(slope_min, intercept_min)

mse2 = model.mse_cal(slope_gd, intercept_gd)

st.markdown(f"<p class='custom-font-write'> En küçük SSR yöntemi ile elde edilen denklemin MSE skoru: <b>{mse1[0]}</b> </p>", unsafe_allow_html=True)

st.markdown(f"<p class='custom-font-write'> Gradient Descent yöntemi ile elde edilen denklemin MSE skoru: <b>{mse2[0]}</b> </p>", unsafe_allow_html=True)