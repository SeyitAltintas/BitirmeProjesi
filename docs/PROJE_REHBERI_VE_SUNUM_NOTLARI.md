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
> *Biz bu projede; aşçının veya idarenin sübjektif hisleriyle değil; **hava durumu, kampüs etkinlikleri, KYK burs döngüsü, haftanın günü ve menü türünü** analiz eden bir **Yapay Zeka Karar Destek Sistemi** geliştirdik. Sistemimiz aşçıya sadece 'Kaç kg pişirmelisin?' demekle kalmıyor; olası riskleri P50/P90 emniyet sınırlarıyla hesaplıyor, Newsvendor kuralıyla maliyeti optimize ediyor ve Açıklanabilir Yapay Zeka (XAI) ile her kararın arkasındaki gerekçeyi şefe şeffafça açıklıyor."*

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
3. **Takvim & Zaman:** Günü (`DayOfWeek`), ayı (`Month`) ve hafta sonu mu (`IsWeekend`) olduğunu sayısallaştırdık. Cuma günleri öğrencilerin erken memlekete gitme eğilimini model bu sayede yakalar.

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
   * **MAE (Ortalama Hata):** **1.08 kg**
   * **$R^2$ Skoru:** **%85.1** *(Baseline 3 kg hata yaparken, yapay zekamız hatayı yaklaşık 1 kg'a indirdi).*
2. **Kantil (Quantile P50 / P90) Modeli:**
   * **P50 (Medyan):** Normal şartlarda aşılmayacak taahhüt miktarı.
   * **P90 (Emniyet Sınırı):** En olumsuz koşulda bile israfın kalacağı tavan sınır (test döneminde **%85.1 kapsama**; hedef %90 — hedefin biraz altında kaldı).
3. **İsraf Nedeni Sınıflandırma Modeli (`HistGradientBoostingClassifier`):**
   * **Doğruluk Oranı (Accuracy):** **%82.7**
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

Arayüzde yan panel yoktur; tüm girişler ana sayfadadır. 5 sekme vardır.

### 1. Sekme: "Günlük Tahmin" (Aşçının Sabahı)
**Girişler:** tarih, menü kategorisi, hava durumu, kampüs etkinliği (Normal gün / Etkinlik), beklenen talep (kg), planlanan üretim (kg).
Burs dönemi ve Cuma etkisi tarihten **otomatik** hesaplanır ve ekranda etiket olarak gösterilir.

**Çıktılar ("Tahmini çalıştır" sonrası):**
1. **Tahmini israf (kg)** ve planlanan üretime oranı.
2. **P50 (medyan):** ön taahhüt için ayrılacak miktar.
3. **P90 (emniyet sınırı):** kurtarma kapasitesi için tavan.
4. **Ana israf nedeni** ve olasılığı (5 sınıf).
5. **XAI grafiği:** Her faktör (fazla üretim, burs dönemi, Cuma, etkinlik, hava) nötr değere çekilip model yeniden çalıştırılır; tahmindeki fark (kg) o faktörün **gerçek katkısıdır**. Kırmızı çubuk israfı artırır, yeşil azaltır.

> Hocaya not: XAI katkıları sabit kural değil, modelin kendi çıktısından (karşı-olgusal analiz) hesaplanır.

### 2. Sekme: "Maliyet Analizi"
Fazla üretim maliyeti (Cₒ) ve eksik üretim maliyeti (Cᵤ) bu sekmenin üstünde girilir. 1.000 senaryolu Monte Carlo simülasyonuyla toplam maliyet eğrisi çizilir; optimum üretim noktası (Newsvendor) gösterilir. Cᵤ > Cₒ ise sistem emniyet payı ekler.

### 3. Sekme: "İkinci Hayat" (EPA Hiyerarşisi)
Artan yemek için 4 kademeli plan: insan tüketimi, ileri dönüşüm, hayvan besleme, kompost.

### 4. Sekme: "Veri & Model Başarımı"
* 1.572 günlük yemekhane istatistikleri, israf dağılımı ve nedenleri.
* Baseline modeller ile üretim modelinin karşılaştırması.
* **Genel öznitelik önemi:** Üretim modelinin test dönemi (Ekim–Aralık) üzerinde **canlı hesaplanan** permütasyon önemi (sabit rakam değildir).

### 5. Sekme: "Model Karşılaştırması"
HGB, RandomForest, XGBoost ve CatBoost'un metrik tablosu, grafikleri ve son girdi için dört modelin tahmini.

---

## 🎯 4. BÖLÜM: Sınavda / Jüride Gelebilecek 5 Kritik Soru ve Cevapları

### ❓ Soru 1: "Neden derin öğrenme (LSTM, Yapay Sinir Ağları) değil de Gradient Boosting (HistGradientBoosting) kullandınız?"
> **Cevabınız:**  
> *"Hocam, tabular (tablo) verilerde ve operasyonel iş problemlerinde Gradient Boosting algoritmaları (XGBoost, LightGBM, HistGradientBoosting) literatürde derin öğrenmeden çok daha yüksek başarı göstermekte ve daha az veriyle aşırı öğrenmeye (overfitting) düşmeden genelleme yapabilmektedir. Ayrıca HistGradientBoosting, kategorik verileri yerel olarak işleyebilmekte ve milisaniyeler içinde tahmin üreterek web arayüzünde sıfır gecikmeyle çalışmaktadır."*

### ❓ Soru 2: "Neden tek bir regresyon modeli yetmedi de Quantile (P50/P90) ve Sınıflandırma modelleri de eklediniz?"
> **Cevabınız:**  
> *"Çünkü yemekhane operasyonu tek boyutlu değildir. Aşçıya sadece 'Bugün 10 kg israf olur' demek eksik bilgi verir. Aşçı şunu da bilmelidir:*
> * *1) En kötü günde bile kurtarmam gereken maksimum tavan nedir? (Bunu P90 Quantile modeli söyler).*
> * *2) Bu yemek neden artacak? Aşırı pişirdiğim için mi, yoksa talep düşük kaldığı için mi? (Bunu Sınıflandırma modeli söyler).*  
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

## 🆕 4.5 BÖLÜM: Sonradan Eklenen Yenilikler

### A) Kampüse Özgü 3 Yeni Değişken
| Değişken | Nasıl Belirlenir? | Mantık |
| :--- | :--- | :--- |
| **`Burs_Donemi`** (KYK burs döngüsü) | **Otomatik** (tarihten): ayın 6–15'i `Burs_Sonrasi`, 20'si ve sonrası `Ay_Sonu`, diğerleri `Dengeli_Donem` | Burs yatınca öğrenci dışarıda yer, ay sonu bütçe daralınca yemekhaneye yönelir. |
| **`Is_Friday`** (Cuma etkisi) | **Otomatik** (tarihten) | Cuma günü öğrenciler erken ayrılır, talep düşer, israf artar. |
| **`Campus_Event`** (Kampüs etkinliği) | **Aşçı tek seçimle girer** (Normal gün / Etkinlik) | Sınav haftası, şenlik gibi günler takvimden bilinemez; tek manuel girdi budur. |

> **Sınırlılık:** `Campus_Event`, Kaggle'daki `Promotion` sütununun yeniden adlandırılmış halidir; `Burs_Donemi` ve `Is_Friday` tarihten kural ile türetilmiştir. Etkileri küçüktür ve gerçek yemekhane verisiyle doğrulanmadır (ayrıntı: `VERI_SETI_DONUSUMU_VE_HOCA_SUNUM_METNI.md`).

Böylece aşçının girdiği alan sayısı artmadı: tarih, menü, hava, talep, üretim ve tek bir etkinlik seçimi.

### B) Çıkarılan Değişkenler
`Storage_Temperature` ve `Shelf_Life_Days` modelden çıkarıldı (önem %0.3; depolama sıcaklığını aşçı kendisi ayarladığı için tahmin değeri yok).

### C) Güncel Üretim Modeli Başarımı (Test: Ekim–Aralık, 376 gün)
* Regresyon (HGB): **MAE 1.08 kg**, **R² 0.85** (referans: sabit medyan 3.02 kg)
* Kantil modeli: **P90 kapsama %85.1** (hedef %90; hedefin altında. P50 kapsaması %71.5 olup veride %46.6 sıfır israf günü olması nedeniyle medyan modeli düşük kalıyor)
* İsraf nedeni sınıflandırıcı: **%82.7 doğruluk**

### D) Arayüz Yenilemesi
* Yan panel kaldırıldı; maliyet parametreleri (Cₒ, Cᵤ) "Maliyet Analizi" sekmesine, günlük girişler ana sayfaya taşındı.
* XAI ve genel öznitelik önemi artık sabit değil, modelden hesaplanıyor.
* Tasarım sadeleştirildi: tek vurgu rengi, animasyonsuz, kart tabanlı düzen.
* Yeni **5. sekme: Model Karşılaştırması**.

### E) Model Karşılaştırması (HGB, RandomForest, XGBoost, CatBoost)
Aynı veri, aynı zaman bazlı bölme, aynı 13 öznitelik. Script: `scripts/model_egit_karsilastir.py`, çıktı: `models/comparison/model_karsilastirma.csv`. HGB satırı, `models/` içindeki gerçek üretim modelinin test sonucudur.

| Model | MAE (kg) | RMSE (kg) | R² | Sınıflandırma doğruluğu | Eğitim süresi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HistGradientBoosting (üretim) | 1.081 | 3.055 | 0.851 | %82.7 | 0.9 sn |
| RandomForest | 1.000 | **2.958** | **0.860** | %81.9 | 0.4 sn |
| XGBoost | 1.016 | 3.275 | 0.829 | %82.4 | 0.6 sn |
| CatBoost | **0.960** | 3.072 | 0.849 | **%83.0** | 9.9 sn |

* **Dürüst yorum:** Dört model birbirine çok yakın; en iyi MAE CatBoost'ta (0.96), en iyi R² RandomForest'ta. Fark ≈ 0.12 kg ve tek bir test dönemine dayanıyor, istatistiksel olarak kesin değil. Hata payı büyük ölçüde verideki doğal gürültüden geliyor.
* **HGB neden üretimde?** Kategorikleri doğal işler (ek kodlama gerekmez), ek kütüphane istemez ve P50/P90 kantil modellerini aynı yapıda sunar. Kantil tahmini diğer üç modelle karşılaştırılmadı.
* **Hocaya söylenecek:** *"Dört modeli denedik, sonuçlar yakın çıktı; bu da modelden çok verinin sınırlı olduğunu gösteriyor. Sonraki adım çapraz doğrulama ve gerçek yemekhane verisidir."*
* Not: Karşılaştırma modelleri yalnızca israf miktarı (ve sınıflandırma) için eğitildi.

---

## 📁 5. BÖLÜM: Dosya Sözlüğü (Ne Nerede?)

```
BitirmeProjesi/
├── arayuz.py                  Streamlit karar destek uygulaması (5 sekme)
├── baslat.bat                 Windows'ta tek tıkla başlatma
├── requirements.txt           Gerekli kütüphaneler
├── README.md                  Kurulum ve ekip çalışma rehberi
├── data/
│   ├── food_waste_dataset.csv        Kaggle ham veri (8.000 satır)
│   ├── yemekhane_veriseti.csv        Yalnızca Cafeteria (1.572 satır)
│   └── yemekhane_oznitelikler.csv    Türetilmiş değişkenlerle nihai veri
├── models/
│   ├── food_waste_regressor_model.pkl   İsraf miktarı (HGB)
│   ├── quantile_models.pkl              P50 / P90 modelleri
│   ├── food_waste_classifier_model.pkl  İsraf nedeni sınıflandırıcı
│   └── comparison/
│       ├── rf_model.pkl, xgb_model.pkl, catboost_model.pkl
│       └── model_karsilastirma.csv      Karşılaştırma sonuç tablosu
├── scripts/
│   └── model_egit_karsilastir.py        RF / XGBoost / CatBoost eğitimi ve karşılaştırma
└── docs/
    ├── PROJE_REHBERI_VE_SUNUM_NOTLARI.md
    └── VERI_SETI_DONUSUMU_VE_HOCA_SUNUM_METNI.md
```

> Not: Üretim modellerini (`models/*.pkl`) eğiten script depoda yoktur; sadece karşılaştırma scripti vardır. Tam tekrarlanabilirlik için ayrı bir eğitim scripti eklenmesi önerilir.
