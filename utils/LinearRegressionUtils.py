
linear_aciklama="""
## Lineer Regresyon
Linear regression, istatistiksel bir modelleme tekniğidir. Bu yöntem, bağımlı değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi açıklamak için kullanılır. Temel olarak, bağımlı değişken ile bağımsız değişkenler arasında doğrusal bir ilişki olduğu varsayımına dayanır. Hedefi, veri setindeki bu ilişkiyi temsil eden bir doğru veya düzlemi bulmaktır. Bu doğru veya düzlem, veri noktalarına en uygun şekilde uyan ve gelecekteki tahminler için kullanılan bir model oluşturur.
    """
        
linear_varsayımlar="""### Lineer regresyonun temel varsayımları şunlardır:

**1. Doğrusallık:** Bağımlı değişken ile bağımsız değişkenler arasında doğrusal bir ilişki olmalıdır. Bu, bağımsız değişkenlerdeki bir birimlik değişimin, bağımlı değişkende sabit bir miktarda değişime neden olması gerektiği anlamına gelir.

* **Açıklama:** Bu varsayım, modelin bağımlı ve bağımsız değişkenler arasındaki ilişkiyi doğru bir şekilde yakalayabilmesi için önemlidir. Eğer ilişki doğrusal değilse, lineer regresyon modeli uygun olmayabilir ve farklı bir model kullanılması gerekebilir.
* **Nasıl Kontrol Edilir:** Bağımlı ve bağımsız değişkenler arasındaki ilişkiyi görselleştirmek için bir dağılım grafiği çizilebilir. Eğer noktalar düz bir çizgi etrafında toplanıyorsa, doğrusallık varsayımı karşılanmış demektir.
* **İhlal Edildiğinde Ne Olur:** Doğrusallık varsayımı ihlal edilirse, modelin tahminleri yanlış olabilir ve modelin açıklayıcı gücü düşük olur.

**2. Bağımsızlık:** Hatalar (artıklar) birbirinden bağımsız olmalıdır. Bu, bir hatanın değerinin, diğer hataların değerlerini etkilememesi gerektiği anlamına gelir.

* **Açıklama:** Bu varsayım, özellikle zaman serisi verileriyle çalışırken önemlidir. Eğer hatalar birbirine bağlıysa, modelin standart hataları yanlış hesaplanabilir ve sonuçlar yanıltıcı olabilir.
* **Nasıl Kontrol Edilir:** Hataların otokorelasyonunu kontrol etmek için Durbin-Watson testi gibi istatistiksel testler kullanılabilir.
* **İhlal Edildiğinde Ne Olur:** Bağımsızlık varsayımı ihlal edilirse, modelin standart hataları yanlış hesaplanabilir ve sonuçlar yanıltıcı olabilir.

**3. Sabit Varyans (Homoskedastisite):** Hataların varyansı tüm bağımsız değişken değerleri için sabit olmalıdır.

* **Açıklama:** Bu varsayım, modelin tüm bağımsız değişken değerleri için aynı hassasiyete sahip olmasını sağlar. Eğer hataların varyansı değişiyorsa, modelin bazı tahminleri diğerlerinden daha az güvenilir olabilir.
* **Nasıl Kontrol Edilir:** Hataların bağımsız değişkenlere göre grafiğini çizerek homoskedastisite kontrol edilebilir. Eğer noktalar yatay bir bant etrafında dağılmışsa, homoskedastisite varsayımı karşılanmış demektir.
* **İhlal Edildiğinde Ne Olur:** Homoskedastisite varsayımı ihlal edilirse (heteroskedastisite), modelin standart hataları yanlış hesaplanabilir ve sonuçlar yanıltıcı olabilir.

**4. Normal Dağılım:** Hatalar normal dağılıma sahip olmalıdır.

* **Açıklama:** Bu varsayım, modelin güven aralıklarının ve hipotez testlerinin doğru bir şekilde hesaplanabilmesi için önemlidir.
* **Nasıl Kontrol Edilir:** Hataların histogramını veya Q-Q grafiğini çizerek normal dağılım kontrol edilebilir.
* **İhlal Edildiğinde Ne Olur:** Normal dağılım varsayımı ihlal edilirse, modelin güven aralıkları ve hipotez testleri yanlış olabilir.

**5. Çoklu Doğrusal Bağlantı (Multicollinearity) Olmaması:** Bağımsız değişkenler arasında yüksek korelasyon olmamalıdır.

* **Açıklama:** Bu varsayım, modelin katsayılarının doğru bir şekilde tahmin edilebilmesi için önemlidir. Eğer bağımsız değişkenler arasında yüksek korelasyon varsa, katsayıların tahminleri kararsız ve güvenilmez olabilir.
* **Nasıl Kontrol Edilir:** Bağımsız değişkenler arasındaki korelasyon matrisini inceleyerek veya Varyans Şişirme Faktörü (VIF) gibi istatistiksel ölçütleri kullanarak çoklu doğrusal bağlantı kontrol edilebilir.
* **İhlal Edildiğinde Ne Olur:** Çoklu doğrusal bağlantı varsa, modelin katsayıları kararsız ve güvenilmez olabilir. """



basic_linear_formule="""
### Basit Doğrusal Regresyon Formülü:

```py
y = β₀ + β₁x + ε
```

**Formülün Açıklaması:**

* **y:** Bağımlı değişken. Tahmin etmeye çalıştığımız değişkendir.
* **x:** Bağımsız değişken. Bağımlı değişkeni etkilediği düşünülen değişkendir.
* **β₀:** Kesim noktası (y-kesişimi). Regresyon doğrusunun y eksenini kestiği noktadır. x=0 olduğunda y'nin beklenen değerini temsil eder.
* **β₁:** Eğim. Bağımsız değişkendeki bir birimlik değişimin bağımlı değişkendeki ortalama değişimi temsil eder.
* **ε:** Hata terimi. Modelin açıklayamadığı rastgele değişkenliği temsil eder.

**Formülün Detaylı Açıklaması:**

* **β₀ (Kesim Noktası):** Regresyon doğrusunun y eksenini kestiği noktadır. Bu, bağımsız değişken (x) sıfır olduğunda bağımlı değişkenin (y) beklenen değerini temsil eder.
* **β₁ (Eğim):** Bağımsız değişkendeki bir birimlik değişimin bağımlı değişkendeki ortalama değişimi temsil eder. Pozitif bir eğim, x arttıkça y'nin de arttığını, negatif bir eğim ise x arttıkça y'nin azaldığını gösterir.
* **ε (Hata Terimi):** Modelin açıklayamadığı rastgele değişkenliği temsil eder. Bu, verilerdeki gürültü, ölçüm hataları veya modelde dahil edilmeyen diğer faktörlerden kaynaklanabilir.

**Örnek:**

Bir öğrencinin ders çalışma saati (x) ile sınav notu (y) arasındaki ilişkiyi modellemek istediğimizi varsayalım. Basit doğrusal regresyon kullanarak, aşağıdaki denklemi elde edebiliriz:

```py
y = 50 + 5x
```

Bu denklemde:

* **β₀ = 50:** Kesim noktası. Hiç ders çalışmayan bir öğrencinin beklenen sınav notu 50'dir.
* **β₁ = 5:** Eğim. Her bir saatlik ek ders çalışmanın sınav notunu ortalama 5 puan artırması beklenir.

**Sonuç:**

Basit doğrusal regresyon, iki değişken arasındaki ilişkiyi modellemek için güçlü bir araçtır. Formül, bağımsız değişkenin değerine dayanarak bağımlı değişkenin değerini tahmin etmek için kullanılabilir. Bu, gelecekteki olayları tahmin etmek veya iki değişken arasındaki ilişkiyi anlamak için yararlı olabilir.
"""


linear_closed_formule="""Basit doğrusal regresyonun kapalı formülü, regresyon katsayılarını (β₀ ve β₁) doğrudan hesaplamak için kullanılan matematiksel bir ifadedir. Bu formül, en küçük kareler yöntemine dayanır ve verilerdeki hata karelerinin toplamını minimize eden katsayıları bulur.

**Kapalı Formül:**

β₀ ve β₁ katsayıları için kapalı formüller şunlardır:


$$β₁ (Eğim):$$


$$
β₁ = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]
$$


$$β₀ (Kesim Noktası):$$

$$
β₀ = ȳ - β₁x̄
$$

**Formülün Açıklaması:**

* **Σ:** Toplam sembolü, belirtilen aralıktaki tüm değerlerin toplamını ifade eder.
* **xi:** Bağımsız değişkenin i. gözlemi.
* **yi:** Bağımlı değişkenin i. gözlemi.
* **x̄:** Bağımsız değişkenin ortalaması.
* **ȳ:** Bağımlı değişkenin ortalaması.

**Formülün Detaylı Açıklaması:**

$$β₁ (Eğim)$$

* $$Σ[(xi - x̄)(yi - ȳ)]:$$ Bağımsız ve bağımlı değişkenler arasındaki kovaryansı temsil eder. Bu değer, x ve y'nin birlikte nasıl değiştiğini gösterir.
* $$Σ[(xi - x̄)²]:$$ Bağımsız değişkenin varyansını temsil eder. Bu değer, x değerlerinin ortalamadan ne kadar yayıldığını gösterir.

Eğim (β₁), kovaryansın varyansa bölünmesiyle hesaplanır. Bu, bağımsız değişkendeki bir birimlik değişimin bağımlı değişkendeki ortalama değişimi gösterir.

**β₀ (Kesim Noktası):**

Kesim noktası (β₀), bağımlı değişkenin ortalaması (ȳ) ile eğimin (β₁) bağımsız değişkenin ortalamasıyla (x̄) çarpımının farkı olarak hesaplanır. Bu, regresyon doğrusunun y eksenini kestiği noktayı belirler.

**Örnek:**

Aşağıdaki veri setini ele alalım:

| x (Çalışma Saati) | y (Sınav Notu) |
|---|---|
| 2 | 60 |
| 4 | 70 |
| 6 | 80 |
| 8 | 90 |

Bu veriler için β₀ ve β₁ katsayılarını kapalı formül kullanarak hesaplayabiliriz:

$$
1. 
x̄  \\text{ ve } ȳ  \\text{ hesapla }:
   x̄ = (2 + 4 + 6 + 8) / 4 = 5
   ȳ = (60 + 70 + 80 + 90) / 4 = 75
$$
$$
2. 
β₁ \\text{ hesapla }:
   Σ[(xi - x̄)(yi - ȳ)] = (2-5)(60-75) 
   + (4-5)(70-75) + (6-5)(80-75) 
   + (8-5)(90-75) = 60
   
   Σ[(xi - x̄)²] = (2-5)² + (4-5)² + (6-5)² + (8-5)² = 20
   β₁ = 60 / 20 = 3
$$
$$
3. β₀ \\text{ hesapla }:
   β₀ = ȳ - β₁x̄ = 75 - 3 * 5 = 60
$$
Bu nedenle, regresyon denklemi:

$$
y = 60 + 3x
$$

**Sonuç:**

Kapalı formül, regresyon katsayılarını doğrudan hesaplamak için etkili bir yöntemdir. Bu formül, verilerdeki kalıpları anlamak ve gelecekteki değerleri tahmin etmek için kullanılan regresyon modellerini oluşturmak için temel bir araçtır.
"""


linear_gradien_descent="""

**Gradient Descent ile Basit Doğrusal Regresyon:**

1. **Hata Fonksiyonu (Maliyet Fonksiyonu):**

   Basit doğrusal regresyonda genellikle **Ortalama Kare Hata (MSE)** kullanılır:

$$
   MSE = (1/n) * Σ(yi - ŷi)²
$$

   * `n`: Veri noktalarının sayısı
   * `yi`: Gerçek değer
   * `ŷi`: Tahmin edilen değer (`ŷi = β₀ + β₁xi`)

2. **Gradient Hesaplama:**

   MSE'yi minimize etmek için β₀ ve β₁ parametrelerine göre gradyanı (türevi) hesaplamamız gerekir:

$$
   ∂MSE/∂β₀ = (-2/n) * Σ(yi - ŷi)
   ∂MSE/∂β₁ = (-2/n) * Σ(xi * (yi - ŷi))
$$

3. **Parametre Güncelleme:**

   Hesaplanan gradyanları kullanarak β₀ ve β₁ parametrelerini güncelleriz:

$$
   β₀ = β₀ - α * (∂MSE/∂β₀)
   β₁ = β₁ - α * (∂MSE/∂β₁)
$$

   * `α`: Öğrenme oranı (adım boyutu). Bu hiperparametre, her adımda parametrelerin ne kadar güncelleneceğini kontrol eder.

4. **İterasyon:**

   2. ve 3. adımları belirli bir iterasyon sayısı veya hata fonksiyonu belirli bir değere ulaşana kadar tekrarlarız.

**Gradient Descent Algoritmasının Adımları (Özet):**

1. **Başlangıç Değerleri:** β₀ ve β₁ için rastgele başlangıç değerleri seçin.
2. **Gradyan Hesaplama:** MSE'nin β₀ ve β₁'e göre gradyanını hesaplayın.
3. **Parametre Güncelleme:** Gradyanları ve öğrenme oranını kullanarak β₀ ve β₁'i güncelleyin.
4. **Yakınsama Kontrolü:** Hata fonksiyonu belirli bir değere ulaşana veya belirli bir iterasyon sayısı tamamlanana kadar 2. ve 3. adımları tekrarlayın.

**Örnek:**

Önceki örnekteki veri setini kullanarak Gradient Descent ile β₀ ve β₁'i bulalım:

**Veri Seti:**

| x (Çalışma Saati) | y (Sınav Notu) |
|---|---|
| 2 | 60 |
| 4 | 70 |
| 6 | 80 |
| 8 | 90 |

**Başlangıç Değerleri:**

* β₀ = 0
* β₁ = 0
* α = 0.01 (learning rate)

**İterasyon 1:**

1. **Tahmin:** $$ ŷi = β₀ + β₁xi = 0 + 0 * xi = 0 $$ `(tüm iterasyon için)`
2. **Gradyan Hesaplama:**
   * $$ ∂MSE/∂β₀ = (-2/4) * (60-0 + 70-0 + 80-0 + 90-0) = -150 $$
   * $$ ∂MSE/∂β₁ = (-2/4) * (2*(60-0) + 4*(70-0) + 6*(80-0) + 8*(90-0)) = -700 $$
3. **Parametre Güncelleme:**
   * $$ β₀ = 0 - 0.01 * (-150) = 1.5 $$
   * $$ β₁ = 0 - 0.01 * (-700) = 7 $$ 

**İterasyon 2 ve Sonrası:**

Yukarıdaki adımları tekrarlayarak β₀ ve β₁ değerlerini güncellemeye devam ederiz. Her iterasyonda MSE azalacak ve parametreler optimal değerlere yaklaşacaktır.

**Sonuç:**

Gradient Descent, iteratif bir süreçle β₀ ve β₁'i bulmamızı sağlar. Bu yöntem, büyük veri setleri ve karmaşık modeller için özellikle faydalıdır. Öğrenme oranı (α) ve iterasyon sayısı gibi hiperparametrelerin doğru seçimi, algoritmanın performansı için önemlidir.


"""


linear_end="""



**1. Normal Denklemler (Kapalı Form Çözümü):**

Bu yöntem, en küçük kareler yöntemini kullanarak regresyon katsayılarını (β₀ ve β₁) doğrudan hesaplamak için matematiksel bir formül kullanır. Daha önce bahsettiğimiz kapalı formül bu yönteme dayanır.

**Avantajları:**

* Hızlı ve etkilidir, özellikle küçük veri setleri için.
* İterasyon gerektirmez.
* Öğrenme oranı gibi hiperparametre ayarlaması gerektirmez.

**Dezavantajları:**

* Büyük veri setleri için hesaplama açısından yoğun olabilir.
* Çok sayıda özellik (değişken) olduğunda performansı düşebilir.

**2. Stokastik Gradyan İnişi (SGD):**

Gradient descent'in bir varyasyonudur. Her adımda tüm veri seti yerine rastgele seçilen bir veri noktası veya mini-batch kullanarak parametreleri günceller.

**Avantajları:**

* Büyük veri setleri için daha hızlı olabilir.
* Yerel minimumlara takılma olasılığı daha düşüktür.

**Dezavantajları:**

* Yakınsama daha gürültülü olabilir.
* Hiperparametre ayarlaması daha zor olabilir.

**3. Mini-Batch Gradyan İnişi:**

Gradient descent ve SGD arasında bir denge kurar. Her adımda rastgele seçilen küçük bir veri grubu (mini-batch) kullanarak parametreleri günceller.

**Avantajları:**

* SGD'den daha kararlı yakınsama sağlar.
* Büyük veri setleri için etkilidir.

**Dezavantajları:**

* Hiperparametre ayarlaması gerekebilir.

**4. Diğer Optimizasyon Algoritmaları:**

* **Momentum:** Gradient descent'e momentum ekleyerek yerel minimumlardan kaçınmaya yardımcı olur.
* **Adagrad:** Öğrenme oranını her parametre için ayrı ayrı ayarlar.
* **RMSprop:** Adagrad'ın bir iyileştirmesidir.
* **Adam:** Momentum ve RMSprop'u birleştiren popüler bir optimizasyon algoritmasıdır.

**Hangi Yöntemi Seçmeli?**

Hangi yöntemin en uygun olduğu, veri setinin boyutu, özellik sayısı ve hesaplama kaynakları gibi faktörlere bağlıdır.

* **Küçük veri setleri için:** Normal denklemler veya gradient descent uygun olabilir.
* **Büyük veri setleri için:** SGD, mini-batch gradient descent veya diğer optimizasyon algoritmaları daha etkili olabilir.

**Sonuç:**

Basit doğrusal regresyon için çeşitli yöntemler mevcuttur. Her yöntemin avantajları ve dezavantajları vardır. En uygun yöntemi seçmek için veri setinin özelliklerini ve hesaplama kaynaklarını dikkate almak önemlidir."""





multiple_linear_basic="""
## Çoklu Doğrusal Regresyon

Basit doğrusal regresyonun bir uzantısı olan **çoklu doğrusal regresyon**, bir bağımlı değişken ile **birden fazla bağımsız değişken** arasındaki ilişkiyi modellemek için kullanılır. Bu yöntem, birden fazla faktörün bağımlı değişken üzerindeki etkisini analiz etmemizi sağlar.

**Çoklu Doğrusal Regresyon Formülü:**

$$ 
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
$$ 

**Formülün Açıklaması:**

* **y:** Bağımlı değişken. Tahmin etmeye çalıştığımız değişkendir.
* **x₁, x₂, ..., xₙ:** Bağımsız değişkenler. Bağımlı değişkeni etkilediği düşünülen değişkenlerdir.
* **β₀:** Kesim noktası (y-kesişimi). Regresyon doğrusunun y eksenini kestiği noktadır. Tüm bağımsız değişkenler sıfır olduğunda y'nin beklenen değerini temsil eder.
* **β₁, β₂, ..., βₙ:** Regresyon katsayıları. Her bir bağımsız değişkenin bağımlı değişken üzerindeki etkisini temsil eder. Diğer bağımsız değişkenler sabit tutulduğunda, ilgili bağımsız değişkendeki bir birimlik değişimin bağımlı değişkendeki ortalama değişimi gösterir.
* **ε:** Hata terimi. Modelin açıklayamadığı rastgele değişkenliği temsil eder.

**Formülün Detaylı Açıklaması:**

* **β₀ (Kesim Noktası):** Tüm bağımsız değişkenler sıfır olduğunda bağımlı değişkenin beklenen değerini temsil eder.
* **β₁, β₂, ..., βₙ (Regresyon Katsayıları):** Her bir bağımsız değişkenin bağımlı değişken üzerindeki etkisini ölçer. Pozitif bir katsayı, ilgili bağımsız değişken arttıkça bağımlı değişkenin de arttığını, negatif bir katsayı ise ilgili bağımsız değişken arttıkça bağımlı değişkenin azaldığını gösterir.
* **ε (Hata Terimi):** Modelin açıklayamadığı rastgele değişkenliği temsil eder. Bu, verilerdeki gürültü, ölçüm hataları veya modelde dahil edilmeyen diğer faktörlerden kaynaklanabilir.

**Örnek:**

Bir evin fiyatını (y) tahmin etmek istediğimizi varsayalım. Bağımsız değişkenler olarak evin metrekaresi (x₁), oda sayısı (x₂) ve konumu (x₃) kullanabiliriz. Çoklu doğrusal regresyon kullanarak, aşağıdaki denklemi elde edebiliriz:

$$ 
y = 100000 + 500x₁ + 10000x₂ + 5000x₃
$$ 

Bu denklemde:

* **β₀ = 100000:** Kesim noktası. Metrekaresi, oda sayısı ve konumu sıfır olan bir evin (ki bu gerçekçi değil) beklenen fiyatı 100.000 TL'dir.
* **β₁ = 500:** Metrekare katsayısı. Diğer faktörler sabit tutulduğunda, evin metrekaresi her bir birim arttığında fiyatın ortalama 500 TL artması beklenir.
* **β₂ = 10000:** Oda sayısı katsayısı. Diğer faktörler sabit tutulduğunda, evin oda sayısı her bir birim arttığında fiyatın ortalama 10.000 TL artması beklenir.
* **β₃ = 5000:** Konum katsayısı. Diğer faktörler sabit tutulduğunda, evin konumu bir birim iyileştiğinde (örneğin, daha merkezi bir konuma geçtiğinde) fiyatın ortalama 5.000 TL artması beklenir.

**Sonuç:**

Çoklu doğrusal regresyon, birden fazla faktörün bağımlı değişken üzerindeki etkisini analiz etmek için güçlü bir araçtır. Formül, bağımsız değişkenlerin değerlerine dayanarak bağımlı değişkenin değerini tahmin etmek için kullanılabilir. Bu, gelecekteki olayları tahmin etmek veya birden fazla değişken arasındaki karmaşık ilişkileri anlamak için yararlı olabilir.

"""



multiple_linear_close_formule="""## Çoklu Doğrusal Regresyonun Kapalı Formülü

Basit doğrusal regresyonun kapalı formülünün bir uzantısı olan **çoklu doğrusal regresyonun kapalı formülü**, birden fazla bağımsız değişkenin olduğu durumlarda regresyon katsayılarını (β₀, β₁, β₂, ..., βₙ) doğrudan hesaplamak için kullanılır. Bu formül, matris cebiri kullanır ve en küçük kareler yöntemine dayanarak verilerdeki hata karelerinin toplamını minimize eden katsayıları bulur.

**Kapalı Formül (Matris Gösterimi):**

$$ 
β = (XᵀX)⁻¹Xᵀy
$$ 

**Formülün Açıklaması:**

* **β:** Regresyon katsayılarını içeren bir vektördür (β₀, β₁, β₂, ..., βₙ).
* **X:** Bağımsız değişkenlerin değerlerini içeren bir matristir. İlk sütun genellikle 1'lerden oluşur (β₀ katsayısı için) ve sonraki sütunlar her bir bağımsız değişkenin değerlerini içerir.
* **Xᵀ:** X matrisinin transpozudur.
* **(XᵀX)⁻¹:** (XᵀX) matrisinin tersidir.
* **y:** Bağımlı değişkenin değerlerini içeren bir vektördür.

**Formülün Detaylı Açıklaması:**

Bu formül, regresyon katsayılarını bulmak için en küçük kareler yöntemini matris cebiri kullanarak uygular. (XᵀX)⁻¹Xᵀ matrisi, **pseudo-inverse** (sözde ters) olarak adlandırılır ve X matrisi tam ranklı olmadığında bile çözüm bulmayı sağlar.

**Örnek:**

Aşağıdaki veri setini ele alalım:

| x₁ (Metrekare) | x₂ (Oda Sayısı) | y (Ev Fiyatı) |
|---|---|---|
| 100 | 2 | 200000 |
| 150 | 3 | 300000 |
| 200 | 4 | 400000 |

Bu veriler için β₀, β₁ ve β₂ katsayılarını kapalı formül kullanarak hesaplayabiliriz:

1. **X ve y matrislerini oluştur:**

$$ 
X = [[1, 100, 2],
     [1, 150, 3],
     [1, 200, 4]]

y = [[200000],
     [300000],
     [400000]]
$$ 

2. **β vektörünü hesapla:**

$$ 
β = (XᵀX)⁻¹Xᵀy
$$ 

Bu hesaplamayı gerçekleştirmek için Python'daki NumPy kütüphanesi gibi bir matris kütüphanesi kullanabilirsiniz.

**Sonuç:**

Çoklu doğrusal regresyonun kapalı formülü, regresyon katsayılarını doğrudan hesaplamak için etkili bir yöntemdir. Bu formül, özellikle büyük veri setleri ve birden fazla bağımsız değişkenle çalışırken, regresyon modellerini oluşturmak için önemli bir araçtır. Ancak, matris işlemleri hesaplama açısından yoğun olabilir ve büyük veri setleri için sayısal kararlılık sorunlarına yol açabilir. Bu durumlarda, gradyan inişi gibi iteratif optimizasyon algoritmaları daha uygun olabilir.
"""

multiple_linear_gd="""
## Çoklu Doğrusal Regresyon için Gradient Descent

Gradient Descent, çoklu doğrusal regresyon modellerinin parametrelerini optimize etmek için de kullanılabilir. Bu durumda, algoritma birden fazla bağımsız değişkeni (x₁, x₂, ..., xₙ) ve bunlara karşılık gelen katsayıları (β₁, β₂, ..., βₙ) ele alır.

**Gradient Descent ile Çoklu Doğrusal Regresyon:**

1. **Hata Fonksiyonu (Maliyet Fonksiyonu):**

   Çoklu doğrusal regresyonda da genellikle **Ortalama Kare Hata (MSE)** kullanılır:

   $$ 
   MSE = (1/n) * Σ(yi - ŷi)²
   $$ 

   * `n`: Veri noktalarının sayısı
   * `yi`: Gerçek değer
   * `ŷi`: Tahmin edilen değer (`ŷi = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ... + βₙxₙᵢ`)

2. **Gradient Hesaplama:**

   MSE'yi minimize etmek için β₀, β₁, β₂, ..., βₙ parametrelerine göre gradyanı (türevi) hesaplamamız gerekir:

   $$ 
   ∂MSE/∂β₀ = (-2/n) * Σ(yi - ŷi)
   $$ 
   $$ 
   ∂MSE/∂β₁ = (-2/n) * Σ(x₁ᵢ * (yi - ŷi))
   $$ 
   $$ 
   ∂MSE/∂β₂ = (-2/n) * Σ(x₂ᵢ * (yi - ŷi))
   $$ 
   $$ 
   ...
   $$ 
   $$ 
   ∂MSE/∂βₙ = (-2/n) * Σ(xₙᵢ * (yi - ŷi))
   $$ 

3. **Parametre Güncelleme:**

   Hesaplanan gradyanları kullanarak β₀, β₁, β₂, ..., βₙ parametrelerini güncelleriz:


   $$ 
   β₀ = β₀ - α * (∂MSE/∂β₀)
   $$ 
   $$ 
   β₁ = β₁ - α * (∂MSE/∂β₁)
   $$ 
   $$
   β₂ = β₂ - α * (∂MSE/∂β₂)
   $$ 
   $$
   ...
   $$ 
   $$
   βₙ = βₙ - α * (∂MSE/∂βₙ)
   $$ 


   * `α`: Öğrenme oranı (adım boyutu).

4. **İterasyon:**

   2. ve 3. adımları belirli bir iterasyon sayısı veya hata fonksiyonu belirli bir değere ulaşana kadar tekrarlarız.

**Gradient Descent Algoritmasının Adımları (Özet):**

1. **Başlangıç Değerleri:** β₀, β₁, β₂, ..., βₙ için rastgele başlangıç değerleri seçin.
2. **Gradyan Hesaplama:** MSE'nin β₀, β₁, β₂, ..., βₙ'e göre gradyanını hesaplayın.
3. **Parametre Güncelleme:** Gradyanları ve öğrenme oranını kullanarak β₀, β₁, β₂, ..., βₙ'i güncelleyin.
4. **Yakınsama Kontrolü:** Hata fonksiyonu belirli bir değere ulaşana veya belirli bir iterasyon sayısı tamamlanana kadar 2. ve 3. adımları tekrarlayın.

**Örnek:**

Önceki örnekteki ev fiyat tahmini veri setini kullanarak Gradient Descent ile β₀, β₁ ve β₂'yi bulalım:

**Veri Seti:**

| x₁ (Metrekare) | x₂ (Oda Sayısı) | y (Ev Fiyatı) |
|---|---|---|
| 100 | 2 | 200000 |
| 150 | 3 | 300000 |
| 200 | 4 | 400000 |

**Başlangıç Değerleri:**

* β₀ = 0
* β₁ = 0
* β₂ = 0
* α = 0.000001 (Öğrenme Oranı)

**İterasyon 1:**

1. **Tahmin:** $$ ŷi = β₀ + β₁x₁ᵢ + β₂x₂ᵢ = 0 + 0 * x₁ᵢ + 0 * x₂ᵢ = 0 $$ `(tüm iterasyonlar için)`
2. **Gradyan Hesaplama:**
   * $$∂MSE/∂β₀ = (-2/3) * (200000-0 + 300000-0 + 400000-0) = -600000$$
   * $$∂MSE/∂β₁ = (-2/3) * (100*(200000-0) + 150*(300000-0) + 200*(400000-0)) = -140000000$$
   * $$∂MSE/∂β₂ = (-2/3) * (2*(200000-0) + 3*(300000-0) + 4*(400000-0)) = -2600000$$
3. **Parametre Güncelleme:**
   * $$β₀ = 0 - 0.000001 * (-600000) = 0.6$$
   * $$β₁ = 0 - 0.000001 * (-140000000) = 140$$
   * $$β₂ = 0 - 0.000001 * (-2600000) = 2.6$$

**İterasyon 2 ve Sonrası:**

Yukarıdaki adımları tekrarlayarak β₀, β₁ ve β₂ değerlerini güncellemeye devam ederiz. Her iterasyonda MSE azalacak ve parametreler optimal değerlere yaklaşacaktır.

**Sonuç:**

Gradient Descent, iteratif bir süreçle çoklu doğrusal regresyon modellerinin parametrelerini bulmamızı sağlar. Bu yöntem, büyük veri setleri ve karmaşık modeller için özellikle faydalıdır. Öğrenme oranı (α) ve iterasyon sayısı gibi hiperparametrelerin doğru seçimi, algoritmanın performansı için önemlidir.

**Not:** Bu örnekte sadece birkaç iterasyon gösterilmiştir. Gerçek uygulamalarda, yakınsama elde etmek için daha fazla iterasyon gerekebilir. Ayrıca, öğrenme oranının doğru seçimi önemlidir. Çok büyük bir öğrenme oranı, algoritmanın ıraksamasına neden olabilirken, çok küçük bir öğrenme oranı yakınsamayı yavaşlatabilir.

**Önemli:** Çoklu doğrusal regresyon için Gradient Descent uygulanırken, **özellik ölçeklendirme (feature scaling)** gibi teknikler kullanmak önemlidir. Bu, farklı ölçeklere sahip özelliklerin algoritmanın performansını olumsuz etkilemesini önler.




"""



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
