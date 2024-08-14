import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
import pandas as pd

st.set_page_config(layout="wide")
st.title("Linear Regression")


st.markdown(
    """Linear regresyonda amaç girdi olarak kabul ettiğimiz bağımlı değişken olan x değerlerine karşılık gelen hedef değişkenimiz y ile doğrusal bir fonksiyon bulmak.
                Genellikle çok büyük boyutlu olmayan sayısal verilerdeki doğrusal ilişki yakalamak veya ileriye dönük tahminlerde bulunmak için kullanılmaktadır.
                Genelde kullanılan formullerin aşşağıda görüdüğünüz aynı anlama gelen farklı alanlara ve basitliğe sahip gösterimler var.
                """
)
st.latex(r"y = f(x) + \epsilon")
# Basit gösterim
st.latex(
    r"""
    y = \beta_0 + \beta_1 x
    """
)

# Spesifik gösterim
st.latex(
    r"""
    y = \beta_0 + \beta_1 x + \epsilon
    """
)

st.markdown(
    """
                Epsilon değeri gerçek dünya etkisindeki bilinmeyen ve tam olarak modellenemeyen faktördür. Çeşitli veri hatalarını içerir ve independenty and identically distrubed (i.d.d) olarak bilinir yani bağımsızdırlar  ve aynı olasılık dağılımına aittirler.
              """
)
st.markdown("""β0 : Y-intercept (kesim noktası), modelin başlangıç değeri. """)
st.markdown(
    """β1 : Eğim katsayısı, bağımsız değişkenin bağımlı değişken üzerindeki etkisi olarak özetlenebilir """
)
st.markdown("""Peki bu değerler nasıl bulunur? """)

st.latex(
    r"\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}"
)

st.latex(r"\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}")


# Örnek veri seti oluşturma
np.random.seed(42)
x = 2 * np.random.rand(100)
y = 4 + 3 * x + np.random.randn(100)

# Ortalama değerler
x_mean = np.mean(x)
y_mean = np.mean(y)

# Eğim (beta_1) hesaplama
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta_1 = numerator / denominator

# Kesişim noktası (beta_0) hesaplama
beta_0 = y_mean - beta_1 * x_mean


# Streamlit başlığı
st.title("Lineer Regresyon Uygulaması")

# Kod dizinini göster
code = """
    import numpy as np

    # Örnek veri seti oluşturma
    np.random.seed(42) # Rastegele sayıların kullanım tekrarında aynı sayılar olması için sabitliyoruz.
    x = 2 * np.random.rand(100) # 0-100 arasında sayıları 2 ile çarparak genişletiyoruz.
    y = 4 + 3 * x + np.random.randn(100) # dogrusal fonkisoyuna tabii olan bir y değerleri olusturuyoruz.

    # Ortalama değerler
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Eğim (beta_1) hesaplama 
    OrtalamaFark = np.sum((x - x_mean) * (y - y_mean)) # Ortalama farkların carpımı aralarındaki kovaryansı temsil etmektedir.
    xinVaryansı = np.sum((x - x_mean) ** 2) # Burada da x'in farklarının karesi ile x'in varyansını temsil etmiş oluyoruz.
    beta_1 = OrtalamaFark / xinVaryansı    # İki değerin bölümü bize eğimi vermektedir.

    # Kesişim noktası (beta_0) hesaplama
    beta_0 = y_mean - beta_1 * x_mean # Son olarak linear denklemden β1x'i cıkararak x'in sıfırdaki y değerni bulmuş oluyoruz. 
    
    
     # MSE hesaplama
    y_pred = beta_0 + beta_1 * x  #mse hesaplamak icin y thaminlerini buluyuoruz

    mse = np.mean((y - y_pred) ** 2) #Gerçek değerler ile tahmin edilen değerler arasındaki farkların kareleri hesaplaplandıktan sonra ortalamasını kullanıyoruz.
    """
st.code(code, language="python")

# Hesaplamaları göster
st.write(f"Ortalama X = {x_mean:.3f}")
st.write(f"Ortalama Y = {y_mean:.3f}")
st.write(f"Eğim (beta_1)= {beta_1:.3f}")
st.write(f"Kesişim noktası (beta_0) ={beta_0:.3f}")

y_pred = beta_0 + beta_1 * x
st.subheader("MSE Denklemi")
st.latex(r" MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2")

# MSE hesaplama
mse = np.mean((y - y_pred) ** 2)
st.write(f"Mean Squared Error (MSE): {mse:.3f}")


# Grafik oluşturma
fig = go.Figure()

# Veri noktalarını ekleme
fig.add_trace(
    go.Scatter(
        x=x, y=y, mode="markers", name="Veri Noktaları", marker=dict(color="cyan")
    )
)

# Regresyon doğrusunu ekleme
fig.add_trace(
    go.Scatter(
        x=x,
        y=beta_0 + beta_1 * x,
        mode="lines",
        name="Regresyon Doğrusu",
        line=dict(color="red"),
    )
)

# Grafiği stilize etme
fig.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title="Basit Lineer Regresyon",
    xaxis_title="X Değeri",
    yaxis_title="Y Değeri",
)

# Grafiği gösterme
st.plotly_chart(fig)


# Rastgele veri oluştur
np.random.seed(42)
x = 2 * np.random.rand(100)
y = 4 + 3 * x + np.random.randn(100)

st.title("Uygulamadaki beta değerlerini değiştirerek sonuçlarını gözlemleyelim.")

# Kaydırma çubukları
b0 = st.slider(
    "𝛽0 (başlangıç değeri)", min_value=-10.0, max_value=10.0, value=0.0, step=0.01
)
b1 = st.slider("𝛽1 (eğim)", min_value=-10.0, max_value=10.0, value=1.0, step=0.01)

# Regresyon modelini hesapla
y_pred = b0 + b1 * x

# Grafik oluştur
scatter_trace = go.Scatter(
    x=x, y=y, mode="markers", name="Veri Noktaları", marker=dict(color="white")
)
line_trace = go.Scatter(
    x=x, y=y_pred, mode="lines", name="Regresyon Doğrusu", line=dict(color="red")
)

# Farkları gösteren dikey çizgiler
difference_traces = [
    go.Scatter(
        x=[x[i], x[i]],
        y=[y[i], y_pred[i]],
        mode="lines",
        line=dict(color="gray", dash="dash", width=0.5),
        showlegend=False,
    )
    for i in range(len(x))
]

# Layout ayarları
layout = go.Layout(
    title="Lineer Regresyon",
    xaxis=dict(title="X", color="white"),
    yaxis=dict(title="Y", color="white"),
    plot_bgcolor="black",
    paper_bgcolor="black",
    font=dict(color="white"),
)

# Grafik çizimi
fig = go.Figure(data=[scatter_trace, line_trace] + difference_traces, layout=layout)

# Grafiği Streamlit ile göster
st.plotly_chart(fig)


st.header("Gradyan descent(inişi) ile parametre arayışı")


# Formüllerin Gösterimi
st.latex(
    r"""
    J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
    """
)

st.latex(
    r"""
    h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
    """
)

st.latex(
    r"""
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
    """
)

st.latex(
    r"""
    \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
    """
)

st.title("Gradyan İnişi ve Doğrusal Regresyon Formülleri")

# Maliyet Fonksiyonu
st.subheader("Maliyet Fonksiyonu (Cost Function)")
st.latex(
    r"""
        J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
        """
)
st.write(
    """
        **Açıklama:**
        Maliyet fonksiyonu, tahminlerin gerçek değerlere ne kadar yakın olduğunu ölçer. 
        Bu formül, tüm örnekler üzerindeki ortalama kare hata (mean squared error) hesaplamaktadır. 
           """
)

# Hipotez Fonksiyonu
st.subheader("Hipotez Fonksiyonu (Hypothesis Function)")
st.latex(
    r"""
        h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
        """
)
st.write(
    """
        **Açıklama:**
        Hipotez fonksiyonu, doğrusal regresyon modelinin tahmin fonksiyonudur. 
        Modelin giriş özellikleri ile parametrelerin çarpımının toplamını verir.
        """
)

# Gradyan İnişi Güncelleme Kuralı
st.subheader("Gradyan İnişi Güncelleme Kuralı (Gradient Descent Update Rule)")

st.latex(
    r"""
        \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
        """
)
st.write(
    """
        **Açıklama:**
        Bu formül, her parametrenin gradyanını kullanarak güncellenmesini sağlar. 
        Burada, alpha öğrenme oranı ve çarpan durumunda olan maliyet fonksiyonunun θj parametresine göre türevidir.
        """
)

# Ağırlık Güncelleme Kuralı
st.subheader("Ağırlık Güncelleme Kuralı (Weight Update Rule)")

st.latex(
    r"\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})"
)
st.latex(
    r"""
        \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
        """
)
st.write(
    """
        **Açıklama:**
        Bu formül, doğrusal regresyon için gradyan inişi algoritmasının ağırlık güncelleme kuralıdır. 
        Her bir ağırlık, gradyan (türev) bilgisi kullanılarak güncellenir. 
        Burada parametrelere göre maliyet fonksiyonunun gradyanını hesaplar.
        """
)

st.write(
    """
        **İşlem Sırası:**
        1. **Hipotez Fonksiyonu:** Modelin tahminlerini oluşturur.
        2. **Maliyet Fonksiyonu:** Tahminlerin doğruluğunu değerlendirir.
        3. **Gradyan İnişi Güncelleme Kuralı:** Parametrelerin gradyanını hesaplar.
        4. **Ağırlık Güncelleme Kuralı:** Her bir ağırlığı gradyan bilgisi ile günceller.
        """
)


st.title("Doğrusal Regresyon ve Gradyan İnişi Örneği")

# Kod dizinini göster
code = """
    aynı veri setini oluşturuyoruz.
    np.random.seed(42)
    x = 2 * np.random.rand(100)
    y = 4 + 3 * x + np.random.randn(100)

    # Öğrenme oranı ve iterasyon yanı tekrar sayılarını belirliyoruz
    alpha = st.slider('Öğrenme Oranı (α)', 0.001, 0.1, 0.01, 0.001)
    iterations = st.slider('İterasyon Sayısı', 100, 5000, 1000, 100)

    # Hipotez fonksiyonunu yani y_pred (tahmini değerler)
    def hypothesis(theta0, theta1, x):
        return theta0 + theta1 * x

    # Maliyet fonksiyonunu 1/2 ile özellştirilmiş bir mse fonksiyonudur amaç alfa ve iterasyon uygulamasının kolaylaşması
    def compute_cost(theta0, theta1, x, y):
        return (1 / (2 * len(x))) * np.sum((hypothesis(theta0, theta1, x) - y) ** 2)

    # Gradyan inişi ile θ değerlerinin güncellenmesi 
    def gradient_descent(x, y, alpha, iterations): # öğrenme oranı ve tekrar sayısı belirtilir.
        theta0 = 0
        theta1 = 0
        costs = []
    
        for _ in range(iterations): #tekrar sayısı boyunca gradyan hesaplanır
            # Gradyanı hesapla
            gradients0 = (1 / len(x)) * np.sum(hypothesis(theta0, theta1, x) - y)
            gradients1 = (1 / len(x)) * np.sum((hypothesis(theta0, theta1, x) - y) * x)

            # Parametreleri güncelle # her tekrarda parametre güncellenir
            theta0 -= alpha * gradients0
            theta1 -= alpha * gradients1

            # Maliyet fonksiyonun güncellenmesi
            costs.append(compute_cost(theta0, theta1, x, y))

        return theta0, theta1, costs

    # Gradyan inişi işlemi
    theta0, theta1, costs = gradient_descent(x, y, alpha, iterations) 
    """
st.code(code, language="python")


# Veri setini oluştur
np.random.seed(42)
x = 2 * np.random.rand(100)
y = 4 + 3 * x + np.random.randn(100)

# Öğrenme oranı ve iterasyon sayısı
alpha = st.slider("Öğrenme Oranı (α)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
iterations = st.slider("İterasyon Sayısı", 100, 12000, 1000, 100)


# Hipotez fonksiyonunu tanımla
def hypothesis(theta0, theta1, x):
    return theta0 + theta1 * x


# Maliyet fonksiyonunu tanımla
def compute_cost(theta0, theta1, x, y):
    return (1 / (2 * len(x))) * np.sum((hypothesis(theta0, theta1, x) - y) ** 2)


# Gradyan inişi ile parametreleri güncelle
def gradient_descent(x, y, alpha, iterations):
    theta0 = 0
    theta1 = 0
    costs = []

    for _ in range(iterations):
        # Gradyanı hesapla
        gradients0 = (1 / len(x)) * np.sum(hypothesis(theta0, theta1, x) - y)
        gradients1 = (1 / len(x)) * np.sum((hypothesis(theta0, theta1, x) - y) * x)

        # Parametreleri güncelle
        theta0 -= alpha * gradients0
        theta1 -= alpha * gradients1

        # Maliyet fonksiyonunu kaydet
        costs.append(compute_cost(theta0, theta1, x, y))

    return theta0, theta1, costs


# Gradyan inişi işlemi
theta0, theta1, costs = gradient_descent(x, y, alpha, iterations)

# Veri seti ve tahminleri görselleştir
st.subheader("Veri Seti ve Hipotez Fonksiyonu")
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
ax.spines["left"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.scatter(x, y, color="blue", label="Veri")
x_vals = np.linspace(0, 2, 100)
y_vals = hypothesis(theta0, theta1, x_vals)
ax.plot(x_vals, y_vals, color="red", label="Tahmin")
ax.set_xlabel("x", color="white")
ax.set_ylabel("y", color="white")
ax.legend()
st.pyplot(fig)

# Maliyet fonksiyonunun evrimi
st.subheader("Maliyet Fonksiyonunun Evrimi")
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["right"].set_color("white")
ax.spines["left"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.plot(range(iterations), costs, color="green")
ax.set_xlabel("İterasyon Sayısı", color="white")
ax.set_ylabel("Maliyet", color="white")
ax.set_ylim(0, 2)
st.pyplot(fig)

# Sonuçları yazdır
st.subheader("Sonuçlar")
st.write(f"Sonuç: θ0 = {theta0:.3f}, θ1 = {theta1:.3f}")
st.write(f"Son Maliyet: {costs[-1]:.3f}")

y_ = theta0 + theta1 * x

# MSE hesaplama
mseG = np.mean((y - y_) ** 2)
st.write(f"Mean Squared Error (MSE): {mseG:.3f}")


st.subheader("Linear Regresyon Similasyonu")

np.random.seed(42)
x = 2 * np.random.rand(100)
y = 4 + 3 * x + np.random.randn(100)


# Gradyan inişi fonksiyonu
def gradient_descent(x, y, learning_rate=0.001, epochs=500):
    m = x.size
    b0 = 0.0
    b1 = 0.0
    history = []

    for epoch in range(epochs):
        y_pred = b0 + b1 * x
        error = y - y_pred
        b0 -= learning_rate * (-2 * error.sum()) / m
        b1 -= learning_rate * (-2 * (x * error).sum()) / m
        if epoch % 10 == 0:  # Her 10 adımda bir kaydet
            history.append((b0, b1, y_pred.copy()))  # y_pred'i kopyalayarak kaydet

    return b0, b1, history


# Öğrenme oranı ve epoch sayısı için kaydırma çubukları
learning_rate = st.slider(
    "Öğrenme Oranı",
    min_value=0.001,
    max_value=0.05,
    value=0.001,
    step=0.001,
    key="learning_rate",
)
epochs = st.slider(
    "Epoch Sayısı", min_value=100, max_value=8000, value=100, step=100, key="epochs"
)

# Buton ekle ve tıklama durumunu kontrol et
if st.button("Hesapla"):
    b0, b1, history = gradient_descent(x, y, learning_rate, epochs)

    # İlk grafiği oluştur
    scatter_trace = go.Scatter(
        x=x, y=y, mode="markers", name="Veri Noktaları", marker=dict(color="white")
    )
    layout = go.Layout(
        title="Lineer Regresyon - Adım Adım",
        xaxis=dict(title="X", color="white"),
        yaxis=dict(title="Y", color="white"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    fig = go.Figure(data=[scatter_trace], layout=layout)
    plot = st.plotly_chart(fig, use_container_width=True)

    # Simülasyon grafiği
    for b0, b1, y_pred in history:
        line_trace = go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            name="Regresyon Doğrusu",
            line=dict(color="red"),
        )

        # Farkları gösteren dikey çizgiler
        difference_traces = [
            go.Scatter(
                x=[x[i], x[i]],
                y=[y[i], y_pred[i]],
                mode="lines",
                line=dict(color="gray", dash="dash", width=0.5),
                showlegend=False,
            )
            for i in range(len(x))
        ]

        # Verileri güncelle
        fig = go.Figure(
            data=[scatter_trace, line_trace] + difference_traces, layout=layout
        )
        plot.plotly_chart(fig, use_container_width=True)

        # Adımlar arasında kısa bir duraklama ekleyin
        time.sleep(0.1)

    # Nihai parametre değerlerini göster
    st.write(f"𝛽0: {b0:.3f}")
    st.write(f"𝛽1: {b1:.3f}")
    y_p = b0 + b1 * x
    mseP = np.mean((y - y_p) ** 2)
    st.write(f"Mean Squared Error (MSE): {mseP:.3f}")


df = pd.DataFrame({"X": x, "Y": y})

x_range = np.linspace(x.min(), x.max(), 100)


df["ManuelLinear"] = beta_0 + beta_1 * df["X"]
df["GradyanLinear"] = theta0 + theta1 * df["X"]
df_first_10 = df.head(10)


st.title("Veri Kümesinin İlk 10 Örneği")

st.write("İlk 10 Örnek:")
st.dataframe(df_first_10)
