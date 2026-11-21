# 🍲 Üniversite Yemekhanesi Talep & Gıda İsrafı Karar Destek Sistemi
## 🎓 Bitirme Projesi Jüri Sunumu, Savunma ve Detaylı Çalışma Rehberi

> **Bu Rehberin Amacı:** Projeye dair hiçbir şeyi ezberlemek zorunda kalmadan; günlük hayattan gerçek yemekhane senaryoları ve somut rakamlarla projenin mantığını kavramanı ve hocana/jüriye karşı kendinden emin bir şekilde savunma yapmanı sağlamaktır.

---

## 🎬 1. BÖLÜM: Sahne ve Büyük Resim (Hocaya Giriş Konuşması)

### 🎙️ Hocaya Hitap Ederken Kullanacağın Konuşma Metni:
> *"Hocam, projemize başlamadan önce üniversite yemekhanelerinin her sabah karşılaştığı çok kritik bir operasyonel çıkmazı inceledik. Bir üniversite yemekhanesinde her sabah saat 08:00'de aşçı mutfağa girdiğinde şu ikilemle baş başa kalır:*
> 
> * **1. Senaryo (Aşırı Pişirme):** *'Ya yemek yetmezse, öğrenciler aç kalırsa rezil oluruz' korkusuyla aşçı gereğinden 50-60 kg fazla yemek pişirir. Saat 14:00 olduğunda yemekhane kapanır ve tencerelerde kalan onlarca kilo sıcak yemek çöpe gider. Üniversite her ay yüz binlerce lira hammadde zararına uğrar, ciddi bir gıda israfı ve karbon salınımı oluşur.*
> * **2. Senaryo (Eksik Pişirme):** *Aşçı 'İsraf olmasın' diye temkinli davranıp az yemek pişirir. Bu sefer saat 12:45'te yemek biter. Dersinden çıkıp yemekhaneye gelen yüzlerce öğrenci aç kalır, rektörlüğe şikayet dilekçeleri yağar, öğrenci memnuniyeti sıfıra iner.*
> 
> *Biz bu projede; aşçının veya idarenin sübjektif hisleriyle değil; **hava durumu, sınav takvimi, haftanın günü, dolap sıcaklığı ve menü türünü** analiz eden bir **Yapay Zeka Karar Destek Sistemi** geliştirdik. Sistemimiz aşçıya sadece 'Kaç kg pişirmelisin?' demekle kalmıyor; olası riskleri P50/P90 emniyet sınırlarıyla hesaplıyor, Newsvendor kuralıyla maliyeti optimize ediyor ve Açıklanabilir Yapay Zeka (XAI) ile her kararın arkasındaki gerekçeyi şefe şeffafça açıklıyor."*

---

## 🛠️ 2. BÖLÜM: 10 Haftalık Süreçte Ne Yaptık? (Örneklerle Detaylı Anlatım)

### 1. Hafta: Literatür ve Problem Çerçevesi
* **Ne Yaptık?** Dünyada ve Türkiye'deki üniversitelerde gıda israfının nedenlerini araştırdık.
* **Canlı Örnek:** Yapılan akademik çalışmalar gösteriyor ki yemekhanelerdeki israfın %70'i yemeğin tadından değil, **yanlış talep tahmininden (yani gereksiz fazla pişirmeden)** kaynaklanıyor.
* **Metodoloji:** ABD Çevre Koruma Ajansı'nın (**EPA Food Recovery Hierarchy**) hiyerarşisini ve yöneylem araştırmasındaki **Newsvendor (Gazeteci Çocuk)** matematiksel stok modelini temel aldık.

---

### 2. Hafta: Veri Temizleme & Kurumsal Özelleştirme
* **Ne Yaptık?** Kaggle'dan temin ettiğimiz 8.000 satırlık ham açık veri setini aldık.
* **Hocaya Anlatım:** 
  > *"Hocam, ilk indirdiğimiz veri setinde süpermarket rafları, lüks restoranlar ve eve paket servis yapan hayalet mutfaklar vardı. Bir süpermarketteki konserve bisküvi israfı ile bir üniversite yemekhanesindeki sulu yemek israfının dinamiği aynı olamaz. Projemiz %100 'Üniversite Yemekhanesi' odaklı olduğu için süpermarket ve restoranları tamamen eledik; geriye sadece **1.572 günlük saf Üniversite Yemekhanesi (`Cafeteria`)** verisini bıraktık."*

---

### 3. Hafta: Öznitelik Mühendisliği (Feature Engineering)
* **Ne Yaptık?** Ham verideki sütunlar yapay zekanın derin ilişkileri öğrenmesi için yeterli değildi. Modele "aşçı aklı" kazandırmak için matematiksel yeni sütunlar türettik.

#### Somut Örneklerle Yeni Değişkenlerimiz:
1. **`Planned_Surplus` (Planlanan Fazlalık = Pişirilen - Beklenen Talep):**
   * *Örnek:* Aşçı o gün 340 kg kuru fasulye pişirmeyi planlamış, ama beklenen öğrenci talebi 300 kg. $340 - 300 = +40\text{ kg}$ planlanan fazlalık var. Model daha sabahın başında tencereye 40 kg fazla yemek konduğunu anlar.
2. **`Order_Cover` (Sipariş Karşılama Oranı = Pişirilen / Talep):**
   * $340 / 300 = 1.13$ (Yani talepten **%13 daha fazla** üretilmiş). Bu oran 1.0'ın altındaysa (örneğin 0.80) model anlar ki: *"İsraf sıfır olacak ama öğrenciler aç kalacak!"*
3. **`Temp_Deviation` (Soğuk Zincir Sapması):**
   * Her yemeğin bir ideal saklama derecesi vardır (Tavuk/Et: **1.9°C**, Süt/Yoğurt: **3.9°C**, Sıcak Yemek: **19.9°C**).
   * *Örnek:* Et dolabı arızalandı ve 2°C yerine **9°C** oldu. Aradaki fark: $+7.1^\circ\text{C}$. Model hemen alarma geçer: *"Burada bakteri ürer, bu et bozulabilir ve çöpe gidebilir!"*
4. **Takvim & Zaman:** Günü (`DayOfWeek`), ayı (`Month`) ve hafta sonu mu (`IsWeekend`) olduğunu sayısallaştırdık. Cuma günleri öğrencilerin erken memlekete gitme eğilimini model bu sayede yakalar.

---

### 4. Hafta: Baseline (Referans) Modeller
* **Ne Yaptık?** Yapay zeka kurmadan önce, "Hiç yapay zeka olmasaydı düz mantıkla ne kadar hata yapardık?" sorusunun cevabını çıkardık.

| Model Yaklaşımı | Nasıl Çalışır? | MAE (Kaç kg Yanıldı?) | $R^2$ Başarısı |
| :--- | :--- | :---: | :---: |
| **1. Sabit Medyan Kuralı** | *"Her gün geçmiş ortalama kadar (10 kg) israf olur varsayalım."* | **3.02 kg** | Negatif (-0.07) |
| **2. Kategori Medyanı** | *"Et için etin ortalamasını, ekmek için ekmeğin ortalamasını alalım."* | **3.07 kg** | Negatif (-0.06) |
| **3. Naif Fazlalık Kuralı** | *"Planlanan 40 kg fazlalığın geçmiş oranına (%20) göre 8 kg israf olur diyelim."* | **1.11 kg** | %72.2 |

* **Hocaya Vurgu:** *"Hocam, sadece ortalamaya bakarak yemek pişiren bir yemekhane her gün ortalama 3 kilodan fazla sapma yaşar. 3. haftada türettiğimiz `Planned_Surplus` değişkeni bile tek başına hatayı 1.11 kg'a düşürdü."*

---

### 5. ve 6. Hafta: Yapay Zeka Modelleri & Zaman Bazlı Doğrulama (Time-Split)

#### Neden Rastgele Değil de "Zaman Bazlı (Time-Split)" Böldük?
* **Hocanın En Sevdiği Tuzak Soru:** *"Veriyi rastgele mi böldünüz?"*
* **Cevabınız:** *"Kesinlikle hayır hocam. Zaman serisi ve talep tahmininde rastgele (Random Split) bölme yapılırsa geleceğin bilgisi geçmişe sızar (Data Leakage). Gerçek hayatta bir aşçı 15 Kasım'ın yemeğini pişirirken 20 Kasım'ın verisini bilemez, sadece dünü bilebilir. Bu yüzden verimizi **Ocak – Eylül (1.196 gün)** eğitim kümesi, hiç görülmemiş **Ekim – Aralık (376 gün)** ise test kümesi olarak ayırdık."*

#### Eğittiğimiz 3 Modelin Görevleri ve Başarı Karnesi:

1. **İsraf Miktarı Regresyon Modeli (`HistGradientBoostingRegressor`):**
   * **MAE (Ortalama Hata):** Sadece **0.92 kg (920 gram)!**
   * **$R^2$ Skoru:** **%87.2** *(Baseline 3 kg hata yaparken, yapay zekamız hatayı 1 kilonun altına indirdi).*
2. **Kantil (Quantile P50 / P90) Modeli:**
   * **P50 (Medyan):** Normal şartlarda aşılmayacak taahhüt miktarı.
   * **P90 (Emniyet Sınırı):** En olumsuz koşulda bile israfın kalacağı tavan sınır (%86.7 kapsama doğruluğu).
3. **İsraf Nedeni Sınıflandırma Modeli (`HistGradientBoostingClassifier`):**
   * **Doğruluk Oranı (Accuracy):** **%85.6**
   * Yemeğin neden artacağını önceden haber verir:
     * *Düşük Talep (Low Demand)* mi? ➔ F1-Skor: %83
     * *Aşırı Pişirme (Overproduction)* mi? ➔ F1-Skor: %74
     * *İsraf Yok (No Waste)* mi? ➔ F1-Skor: %95

---

### 7. Hafta: Asimetrik Maliyetlendirme (Newsvendor Karar Kuralı)

#### Gerçek Hayat Örneğiyle Mantık:
* Normalde simetrik modellerde 1 kg fazla üretmek ile 1 kg eksik üretmek eşit kabul edilir. Ama yemekhanede eşit değildir!
  * **Fazla Üretim Maliyeti ($C_o = 35$ ₺/kg):** Pişip artan 1 kg yemeğin hammadde ve çöpe dökme maliyeti.
  * **Eksik Üretim Maliyeti ($C_u = 75$ ₺/kg):** Yemek erken bittiğinde aç kalan öğrencinin tepkisi, idari kriz ve dışarıdan acil börek/ekmek getirtme maliyeti.

#### Newsvendor Kritik Fraktili Formülü:
$$CR = \frac{C_u}{C_u + C_o} = \frac{75}{75 + 35} = \frac{75}{110} \approx \%68.2$$

* **Aşçıya Çıkan Tavsiye:**
  > *"Öğrenciyi aç bırakmanın cezası (75 ₺) yemek artırmaktan (35 ₺) çok daha ağırdır. Kritik oran %68 olduğu için, ortalama talep 300 kg olsa bile, risk almamak adına **emniyet payıyla birlikte 318 kg** pişirirsen toplam beklenen zararını en aza indirirsin."*

---

## 🖥️ 3. BÖLÜM: Arayüzün (Dashboard) Canlı Kullanım Senaryosu

Arayüzü açtığında hocana ekranı şu sırayla anlatacaksın:

### 🌟 1. Sekme: "Günlük Tahmin & Karar Destek" (Aşçının Sabahı)
**Canlı Senaryo:**  
* Tarih: **21 Kasım Cuma**
* Menü: **Sıcak Yemek / Et Menüsü**
* Beklenen Talep: **300 kg**
* Aşçının Pişirmeyi Düşündüğü: **340 kg**
* Hava: **Yağmurlu**
* Dolap Sıcaklığı: **8.5°C** (İdeal olan 2°C'den yüksek!)
* Akademik Durum: **Normal Gün**

Aşçı **"🚀 Karar Destek & İsraf Tahmini Çalıştır"** butonuna basar:

1. **Noktasal İsraf Kartı:** *"Bugün yaklaşık **12.4 kg** yemek artacak."*
2. **P50 Medyan İsraf:** *"Normal şartlarda en az **8.0 kg** artar; bunu şimdiden askıda yemeğe ayırın."*
3. **P90 Emniyet Tavanı:** *"En kötü yağmurlu gün senaryosunda bile israf **16.5 kg**'ı geçmeyecek; yurtlara ve barınağa en fazla 16.5 kg'lık kap ayırın."*
4. **Ana Risk Nedeni Kartı:** *"Düşük Talep & Depolama Uyarısı"*
5. **Soğuk Zincir Alarmı:** Ekranda sarı uyarı kutusu çıkar:  
   *`⚠️ Soğuk Zincir Uyarısı: Et dolabı 8.5°C! İdeal olan 1.9°C'den 6.6°C sapmıştır, acil soğutucu kontrolü yapın!`*

---

### 🔍 2. Sekme 1'deki Yıldız Özellik: Açıklanabilir Yapay Zeka (XAI)
Hocan sorarsa: *"Yapay zeka 12.4 kg dedi ama neden dedi?"*  
Hemen sonuç kartlarının altındaki **XAI Tornado Grafiğini** gösterirsin:

* 🔴 **Kırmızı Çubuklar (İsrafı Tetikleyenler):**
  * *Planlanan Fazlalık (+40 kg):* İsraf riskini **+%45 artırdı**.
  * *Dolap Sıcaklık Sapması (+6.6°C):* İsraf riskini **+%18 artırdı**.
  * *Yağmurlu Hava:* İsraf riskini **+%12 artırdı** (Öğrenciler yağmurda kampüse gelmeyebilir).
* 🟢 **Yeşil Çubuklar (İsrafı Frenleyenler):**
  * *Hafta Sonu Olmaması:* İsrafı **-%8 azalttı**.
* **Modelin Aşçıya Notu:**  
  > *"💡 **XAI Yorumu:** Sayın Aşçı, bu yemeğin artmasındaki en büyük sebep %45 oranında talepten fazla pişirmenizdir. Pişirme miktarını 340 kg yerine 315 kg'a çekerseniz tahmini israf 12.4 kg'dan 2.1 kg'a düşecektir."*

---

### ⚖️ 3. Sekme: "Maliyet & Asimetrik Karar Analizi"
Sol menüden fazla üretim maliyeti ($C_o$) ve eksik üretim maliyeti ($C_u$) girilir. Ekranda **1.000 senaryolu Monte Carlo simülasyon eğrisi** çizilir:
* Mavi çizgi: Ortalama talep noktası (300 kg).
* Yeşil kesikli çizgi: Maliyeti en aza indiren optimum pişirme noktası (318 kg).
* **Dinamik Mantık:** Eğer aşçı sebze çorbası gibi ucuz bir menü yapıyorsa ve eksik kalması daha büyük dertse sistem emniyet stoku ekler. Eğer çok pahalı ithal antrikot pişiriyorsa ($C_o > C_u$) sistem *"Fazla pişirme, temkinli ol"* der.

---

### 🔄 4. Sekme: "Gıda İkinci Hayat (EPA Hiyerarşisi)"
Yemek arttı diyelim, çöpe mi dökülecek? **Hayır!**  
Sistem 4 kademeli kurtarma planı sunar:
1. **İnsan Tüketimi (Feed People):** Kalan temiz et ve pilav hijyenik vakumlanıp sınavdaki kütüphane öğrencilerine ve gece yurtlarına gönderilir.
2. **İleri Dönüşüm (Upcycling):** Kalan sebzeler ertesi günün çorbasına katılır; bayatlayan ekmekler fırınlanıp çorba krutonu yapılır.
3. **Hayvan Besleme (Feed Animals):** Uygun artıklar kampüs sokak hayvanları odaklarına ve belediye barınağına teslim edilir.
4. **Kompost & Toprak (Compost):** Yenmeyecek organik atıklar reaktöre atılıp kampüs peyzajı için organik gübreye dönüştürülür (Sıfır Çöp Sahası Atığı).

---

### 📊 5. Sekme: "Veri Analitiği & Model Başarımı"
Hocaya akademik ispatların sunulduğu ekrandır:
* 1.572 günlük yemekhane istatistikleri (Günlerin %46.6'sında sıfır israf başarılmış).
* Baseline modeller ile bizim Gradient Boosting modelimizin karşılaştırma tablosu (Hata: 3.02 kg ➔ 0.92 kg).
* **Global XAI (Permütasyon Öznitelik Önemi):** Modelin genel zeka haritası:
  * Planlanan Fazlalık: **%51.0**
  * Pişirilen Miktar: **%26.9**
  * Öğrenci Talebi: **%16.2**
  * Karşılama Oranı: **%3.6**
  * Dolap Sıcaklığı & Hava: **%2.3**

---

## 🎯 4. BÖLÜM: Sınavda / Jüride Gelebilecek 5 Kritik Soru ve Cevapları

### ❓ Soru 1: "Neden derin öğrenme (LSTM, Yapay Sinir Ağları) değil de Gradient Boosting (HistGradientBoosting) kullandınız?"
> **Cevabınız:**  
> *"Hocam, tabular (tablo) verilerde ve operasyonel iş problemlerinde Gradient Boosting algoritmaları (XGBoost, LightGBM, HistGradientBoosting) literatürde derin öğrenmeden çok daha yüksek başarı göstermekte ve daha az veriyle aşırı öğrenmeye (overfitting) düşmeden genelleme yapabilmektedir. Ayrıca HistGradientBoosting, kategorik verileri yerel olarak işleyebilmekte ve milisaniyeler içinde tahmin üreterek web arayüzünde sıfır gecikmeyle çalışmaktadır."*

### ❓ Soru 2: "Neden tek bir regresyon modeli yetmedi de Quantile (P50/P90) ve Sınıflandırma modelleri de eklediniz?"
> **Cevabınız:**  
> *"Çünkü yemekhane operasyonu tek boyutlu değildir. Aşçıya sadece 'Bugün 10 kg israf olur' demek eksik bilgi verir. Aşçı şunu da bilmelidir:*
> * *1) En kötü günde bile kurtarmam gereken maksimum tavan nedir? (Bunu P90 Quantile modeli söyler).*
> * *2) Bu yemek neden artacak? Aşırı pişirdiğim için mi, yoksa dolap bozulduğu için mi? (Bunu Sınıflandırma modeli söyler).*  
> *Bu 3 model birleştiğinde gerçek bir Karar Destek Sistemi (DSS) oluşmaktadır."*

### ❓ Soru 3: "Açıklanabilir Yapay Zeka (XAI) eklemenizin kurumsal faydası nedir?"
> **Cevabınız:**  
> *"Hocam makine öğrenmesi modelleri genelde 'Kara Kutu' olmakla eleştirilir. Aşçı sisteme güvenmezse kullanmaz. Biz XAI ile aşçıya sadece tahmin vermiyoruz; 'Sen bugün gereğinden 40 kg fazla yemek koyduğun için israf riski %45 arttı' gerekçesini gösteriyoruz. Bu sayede sistem personeli eğiten ve hatalı üretim alışkanlığını değiştiren bir yönetim aracına dönüşüyor."*

### ❓ Soru 4: "Eğer üniversite gerçek yemekhane verisini vermezse bu sistem çalışır mı?"
> **Cevabınız:**  
> *"Evet hocam. Zaten 2. haftada tam olarak bunu modelledik. Kurum verisi gizlilik nedeniyle alınamadığında açık veri seti yemekhane şartlarına göre filtrelenip (1.572 satır) başarıyla eğitilmiştir. Üniversite yarın kendi turnike geçiş ve kazan verilerini CSV olarak sisteme yüklediğinde, transfer learning veya yeniden eğitimle 10 saniye içinde o üniversiteye adapte olabilmektedir."*

### ❓ Soru 5: "Bu projenin TÜBİTAK 2209-A ve sürdürülebilirlik (Yeşil Kampüs) boyutu nedir?"
> **Cevabınız:**  
> *"Projemiz Birleşmiş Milletler Sürdürülebilir Kalkınma Amaçlarından **Hedef 12 (Sorumlu Tüketim ve Üretim)** ve **Hedef 13 (İklim Eylemi)** ile birebir örtüşmektedir. Günlük ortalama 15-20 kg gıdanın çöpe gitmesini önlemek; yılda tonlarca metan gazı salınımını engellemekte, binlerce litre sanal su tasarrufu sağlamakta ve üniversite bütçesine doğrudan ekonomik katkı sunmaktadır."*

---

## 📁 5. BÖLÜM: Dosya Sözlüğü (Ne Nerede?)

* [arayuz.py](file:///c:/Users/seyit/Desktop/BitirmeProjesi/arayuz.py): Streamlit ile yazılmış 4 sekmeli ana karar destek web uygulaması.
* [food_waste_regressor_model.pkl](file:///c:/Users/seyit/Desktop/BitirmeProjesi/food_waste_regressor_model.pkl): İsraf miktarını (kg) tahmin eden regresyon modeli (MAE: 0.92 kg).
* [quantile_models.pkl](file:///c:/Users/seyit/Desktop/BitirmeProjesi/quantile_models.pkl): P50 ve P90 risk emniyet tavanı modelleri.
* [food_waste_classifier_model.pkl](file:///c:/Users/seyit/Desktop/BitirmeProjesi/food_waste_classifier_model.pkl): İsraf sebebini %85.6 doğrulukla bulan sınıflandırıcı.
* [yemekhane_veriseti.csv](file:///c:/Users/seyit/Desktop/BitirmeProjesi/yemekhane_veriseti.csv): Süpermarketlerden temizlenmiş 1.572 günlük saf yemekhane veri seti.
* [yemekhane_oznitelikler.csv](file:///c:/Users/seyit/Desktop/BitirmeProjesi/yemekhane_oznitelikler.csv): Türetilmiş değişkenlerin (Planned Surplus, Temp Deviation vb.) bulunduğu nihai veri seti.
* [baslat.bat](file:///c:/Users/seyit/Desktop/BitirmeProjesi/baslat.bat): Windows'ta tek tıkla arayüzü başlatan betik.
* [requirements.txt](file:///c:/Users/seyit/Desktop/BitirmeProjesi/requirements.txt): Gerekli kütüphaneler (`streamlit`, `scikit-learn`, `plotly`, `shap` vb.).
