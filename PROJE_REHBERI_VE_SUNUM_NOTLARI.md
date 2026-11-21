# 🍲 Üniversite Yemekhanesi Talep & Gıda İsrafı Karar Destek Sistemi
## 🎓 Bitirme Projesi & Jüri Sunum Rehberi

Bu belge; projenin amacını, bugüne kadar yapılan tüm teknik adımları, geliştirilen arayüzün detaylarını ve **hocanızın/jürinin sorabileceği kritik sorulara verilecek yanıtları** içermektedir.

---

## 📌 BÖLÜM 1: Projenin Büyük Resmi (Hocaya 1 Dakikalık Özet)

> **"Hocam projemizin amacı nedir?"**  
> Üniversite yemekhanelerinde her sabah yaşanan en büyük operasyonel kriz şudur: **"Bugün kaç porsiyon yemek pişirmeliyiz?"**
> * **Fazla pişirilirse:** Yemekler artar, çöpe gider, kurum ciddi maddi zarara uğrar ve gıda israfı oluşur.
> * **Az pişirilirse:** Yemek saat 12:45'te biter; öğrenciler aç kalır, mağduriyet ve şikayetler başlar.
> 
> Biz bu projede; aşçıların hisleriyle değil; **hava durumu, haftanın günü, menü çeşidi, sınav haftası ve geçmiş tüketim verilerini** işleyen bir yapay zeka karar destek sistemi kurduk. Sistemimiz hem optimum pişirme miktarını ve belirsizlik payını (P50/P90) hesaplamakta hem de artan yemeği **Açıklanabilir Yapay Zeka (XAI)** ve **EPA İkinci Hayat** protokolleriyle kurtarmaktadır.

---

## 🗓️ BÖLÜM 2: 10 Haftalık Planda Neler Yaptık? (Adım Adım Yol Haritası)

### 1. Hafta: Literatür Taraması ve Problem Tanımı
* Gıda israfının nedenleri incelendi.
* **EPA (Amerikan Çevre Koruma Ajansı) Gıda Kurtarma Hiyerarşisi** incelendi (İnsan Besleme ➔ Hayvan Besleme ➔ Kompost ➔ Bertaraf).
* Yöneylem araştırmasındaki **Newsvendor (Gazeteci Çocuk)** stok optimizasyon modeli projeye teorik taban olarak seçildi.

### 2. Hafta: Veri Temizleme & Kuruma Özelleştirme
* İlk veri setinde (8.000 satır) süpermarketler, restoranlar ve paket servis mutfaklar vardı.
* **Biz bu projeyi sadece üniversite yemekhanesine özel yapacağımız için:** Süpermarket ve restoran verileri tamamen ayıklandı.
* Geriye sadece **1.572 günlük saf Üniversite Yemekhanesi (`Cafeteria`)** verisi bırakıldı (`yemekhane_veriseti.csv`).

### 3. Hafta: Öznitelik Mühendisliği (Feature Engineering)
Yapay zekanın veriyi daha iyi anlayabilmesi için ham veriden matematiksel akıllar türetildi (`yemekhane_oznitelikler.csv`):
1. **`Planned_Surplus` (Planlanan Fazlalık):** $Pişirilen - Beklenen\ Talep$. Baştan fazla yemek yapılıp yapılmadığını gösterir.
2. **`Order_Cover` (Sipariş Karşılama Oranı):** $Pişirilen / Talep$. Örneğin 1.20 ise talepten %20 fazla yemek pişirilmiştir.
3. **`Temp_Deviation` (Soğuk Zincir Sapması):** Dolap sıcaklığı ile ürünün ideal saklama sıcaklığı (Et için 2°C, Süt için 4°C) arasındaki fark. Dolap bozulursa model israf alarmı verir.
4. **Takvim Özellikleri:** Haftanın günü (`DayOfWeek`), Hafta sonu mu (`IsWeekend`), Ay (`Month`).

### 4. Hafta: Baseline (Referans) Modeller
Yapay zeka modellerinin başarısını kıyaslamak için basit kural tabanlı modeller kuruldu:
* **Sabit Medyan Kuralı:** Geçmiş ortalamayı tahmin eder. **MAE (Hata): 3.02 kg**
* **Naif Fazlalık Kuralı:** Planlanan fazlalığın belli bir oranının çöpe gideceğini varsayar. **MAE: 1.11 kg**

### 5. Hafta: Yapay Zeka Modellerinin Eğitilmesi
Sadece yemekhane verisi kullanılarak **3 adet gelişmiş makine öğrenmesi modeli** eğitildi:
1. **Regresyon Modeli (`food_waste_regressor_model.pkl`):** Günün israf miktarını (kg) tahmin eder (`HistGradientBoostingRegressor`).
2. **Kantil (Quantile) Modelleri (`quantile_models.pkl`):** 
   * **P50:** %50 ihtimalle aşılmayacak medyan israf taahhüdü.
   * **P90:** En kötü günde bile kurtarma kapasitesi ayrılacak emniyet tavanı.
3. **Sınıflandırma Modeli (`food_waste_classifier_model.pkl`):** İsrafın *neden* olacağını (Fazla Üretim, Düşük Talep, Son Kullanma Tarihi, Dolap Sıcaklık Hatası) önceden sınıflandırır.

### 6. Hafta: Zaman Bazlı Doğrulama (Time-Split Testi) & Hata Karnesi
* Veri rastgele değil, gerçek hayata uygun olarak **Zaman Bazlı** bölündü:
  * **Eğitim Seti (Geçmiş):** Ocak – Eylül (1.196 gün)
  * **Test Seti (Gelecek):** Ekim – Aralık (376 gün)
* **Modelimizin Başarısı:**
  * **MAE (Ortalama Hata):** Sadece **0.92 kg** *(Baseline 3.02 kg hata yaparken, yapay zekamız hatayı 920 grama indirdi!)*
  * **$R^2$ Başarı Skoru:** **%87.2**
  * **İsraf Sebebi Doğruluğu:** **%85.6**

### 7. Hafta: Asimetrik Maliyetlendirme (Newsvendor Karar Kuralı)
* Yemekhanede maliyetler eşit değildir:
  * Fazla üretirsen: Kilo başı hammadde çöpe gider ($C_o = 35$ ₺/kg).
  * Eksik üretirsen: Öğrenci aç kalır, mağduriyet ve ikame maliyeti oluşur ($C_u = 75$ ₺/kg).
* Sistem $C_u > C_o$ olduğunda aşçıya ortalama talebin üzerine **emniyet stoku ekleyerek** pişirmesini, aksi halde temkinli kalmasını öneren matematiksel kuralı çalıştırır.

### 8. Hafta: Streamlit Karar Destek Web Arayüzü (`arayuz.py`)
* Kullanıcı dostu, 4 sekmeli modern bir karar destek paneli geliştirildi.
* **Açıklanabilir Yapay Zeka (XAI):** Modelin tahminlerinin arkasındaki faktörleri aşçıya ve jüriye renkli etki çubuklarıyla açıklayan modül eklendi.

---

## 🖥️ BÖLÜM 3: Arayüzün Detaylı Rehberi (Hangi Bölüm Ne İşe Yarar?)

### 1. Sol Menü (Sidebar)
* **Proje Künyesi:** Proje adı, kapsamı ve TÜBİTAK 2209-A uyumluluk rozeti.
* **Model Durumu:** 3 modelin devrede olduğunu ve MAE hata skorunu gösteren onay kutusu.
* **Asimetrik Maliyet Parametreleri:** Aşçının veya müdürün fazla üretim ($C_o$) ve eksik üretim ($C_u$) birim maliyetlerini değiştirebildiği giriş alanı.

---

### 2. Sekme 1: 🔮 Günlük Tahmin & Karar Destek *(Ana Ekran)*
Aşçının her sabah mutfağa girdiğinde kullandığı ekrandır.

#### A) Giriş Parametreleri:
* **Tarih:** Gün ve ayı otomatik sisteme işler.
* **Menü Kategorisi:** Sıcak Yemek, Ekmek, Et/Tavuk, Süt, Sebze, Meyve.
* **Beklenen Talep (kg):** Kaç kişilik öğrenci bekleniyor?
* **Planlanan Pişirme (kg):** Aşçı mutfakta kaç kg hazırlamayı düşünüyor?
* **Hava Durumu:** Güneşli, Bulutlu, Yağmurlu, Sıcak.
* **Depolama Sıcaklığı (°C):** Dolabın o anki derecesi. (İdealden 4°C saparsa sistem alarm verir).
* **Akademik Takvim:** Normal gün mü, sınav haftası mı?

#### B) Çıktı Kartları ve Grafikler:
* **Noktasal İsraf Tahmini:** Günün sonunda kaç kg yemek artacağını söyler.
* **P50 Medyan İsraf:** Normal şartlarda artması beklenen miktar.
* **P90 Emniyet Sınırı:** En olumsuz koşulda bile aşılmayacak tavan risk.
* **Ana Risk Nedeni:** Yemeğin neden artacağını sınıflandırır (Örn: *Düşük Talep* veya *Aşırı Pişirme*).
* **🔍 Açıklanabilir Yapay Zeka (XAI) Paneli:** Modelin neden bu tahmini yaptığını gösterir:
  * Kırmızı Çubuklar: İsrafı tetikleyen faktörler (Örn: +40 kg fazla üretim, yağmurlu hava).
  * Yeşil Çubuklar: İsrafı önleyen faktörler (Örn: ideal dolap sıcaklığı, sınav yoğunluğu).

---

### 3. Sekme 2: ⚖️ Maliyet & Asimetrik Karar Analizi
* **Kritik Hizmet Oranı:** $CR = \frac{C_u}{C_u + C_o}$
* **1.000 Senaryolu Monte Carlo Simülasyonu:** Farklı pişirme miktarlarında oluşacak toplam maliyeti hesaplar ve maliyet eğrisinin dip yaptığı **Optimum Pişirme Noktasını** aşçıya tavsiye eder.
* **Dinamik Mantık:** Hangi maliyet daha yüksekse karar mesajı ona göre otomatik şekillenir.

---

### 4. Sekme 3: 🔄 Gıda İkinci Hayat (EPA Kurtarma Kataloğu)
Artan temiz gıdanın çöpe gitmesini önleyen 4 kademeli eylem planı:
1. **İnsan Tüketimi:** Kampüs Askıda Yemek, Gece Yurt İkramı, Belediye Gıda Bankası.
2. **İleri Dönüşüm (Upcycling):** Ertesi günün "Şefin Spesiyali" çorbası, kruton ekmek, meyve kompostosu.
3. **Hayvan Besleme:** Kampüs sokak hayvanları odakları ve barınaklar.
4. **Kompost & Toprak:** Kampüs bahçeleri için organik gübre üretimi.

---

### 5. Sekme 4: 📊 Veri Analitiği & Model Başarımı
* 1.572 günlük yemekhane veri dağılımı (Sıfır israf oranı, ortalama tüketim).
* **Akademik Model Karşılaştırma Tablosu:** Baseline vs Bizim Modelimiz ($MAE$, $R^2$).
* **Model Zeka Haritası (Permutation Feature Importance):** Modelin genel olarak neye en çok önem verdiğini gösteren global açıklanabilirlik grafiği.

---

## 🎯 BÖLÜM 4: Hocanın / Jürinin Sorabileceği Zor Sorular ve Cevapları

### Soru 1: "Neden veriyi rastgele (%80 tren, %20 test) bölmediniz de Zaman Bazlı (Time-Split) böldünüz?"
> **Cevabınız:**  
> *"Hocam zaman serisi ve talep verilerinde rastgele bölme yapılırsa geleceğin bilgisi geçmişe sızar (Data Leakage). Gerçek hayatta bir aşçı yarının talebini tahmin ederken sadece dünü ve geçmiş ayları bilebilir. Bu yüzden Ocak–Eylül aylarını eğitim, hiç görülmemiş Ekim–Aralık aylarını test olarak ayırarak en gerçekçi doğrulamayı yaptık."*

### Soru 2: "P50 ve P90 ne demektir? Neden tek bir tahminle yetinmediniz?"
> **Cevabınız:**  
> *"Noktasal tahmin tek bir sayı verir (örneğin 10 kg). Ancak gıda operasyonlarında belirsizlik vardır. **P50**, medyan tahmindir (%50 ihtimalle israf bu sayıyı aşmaz). **P90** ise aşırı talep düşüşü olsa bile israfın kalacağı emniyet tavanıdır. Aşçı askıda yemek ve kurtarma paydaşlarına P50 kadar kesin taahhüt verirken, P90 kadar kurtarma kapasitesini hazırda tutar."*

### Soru 3: "Baseline (Referans) modelleriniz neydi, yapay zekanız ne kadar fark yarattı?"
> **Cevabınız:**  
> *"Hocam ilk olarak düz mantıkla çalışan Sabit Medyan kuralını denedik; 3.02 kg hata yaptı. Daha sonra öznitelik mühendisliği ile türettiğimiz Naif kuralı denedik; 1.11 kg hata verdi. Son olarak eğittiğimiz HistGradientBoosting modelimiz ise ortalama hatayı **0.92 kg'a (920 grama)** indirerek %87.2 açıklama gücüne ulaştı."*

### Soru 4: "Modeliniz bir kara kutu (Black Box) mu? Tahminin gerekçesini biliyor musunuz?"
> **Cevabınız:**  
> *"Hayır hocam, sistemimize **Açıklanabilir Yapay Zeka (XAI)** modülü entegre ettik. Sistem her tahminde israfı tetikleyen veya önleyen faktörleri (örneğin aşırı pişirme, yağmurlu hava veya raf ömrü) anlık bir etki grafiğiyle görselleştirip şefe gerekçesini açıklamaktadır."*

### Soru 5: "Newsvendor asimetrik maliyet analizini neden kullandınız?"
> **Cevabınız:**  
> *"Çünkü yemekhanede fazla yemek üretmekle eksik üretmek aynı zararı doğurmaz. Yemeğin erken bitip öğrencinin aç kalması ($C_u = 75$ ₺) kurumsal olarak yemeğin artmasından ($C_o = 35$ ₺) daha ağır bir maliyettir. Bu asimetriyi Newsvendor kritik fraktil formülüyle optimize ederek toplam zararı en aza indiren pişirme miktarını hesaplattık."*

---

## 📁 BÖLÜM 5: Dosya Haritası

* **`arayuz.py`**: Streamlit karar destek web uygulaması.
* **`food_waste_regressor_model.pkl`**: İsraf kg tahmin modeli.
* **`quantile_models.pkl`**: P50 ve P90 risk emniyet modelleri.
* **`food_waste_classifier_model.pkl`**: İsraf nedeni sınıflandırıcı model.
* **`yemekhane_veriseti.csv`**: Süpermarketlerden arındırılmış 1.572 satırlık temiz yemekhane verisi.
* **`yemekhane_oznitelikler.csv`**: Yeni değişkenlerin üretildiği işlenmiş veri.
* **`baslat.bat`**: Windows'ta tek tıkla arayüzü başlatan betik.
* **`requirements.txt`**: Gerekli Python kütüphaneleri listesi.
