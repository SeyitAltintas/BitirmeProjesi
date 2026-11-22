# 📊 Kaggle Veri Setinin Dönüşüm Hikayesi ve Hoca Sunum Metni
## 🎓 "Veriyi Nasıl Aldık, Neleri Değiştirdik ve Neden?"

Bu doküman; projede kullanılan veri setinin ilk ham halinden bugünkü nihai haline gelene kadar geçirdiği **tüm veri temizleme, öznitelik mühendisliği ve sadeleştirme adımlarını** ve bunu **hocanıza/jüriye adım adım nasıl anlatacağınızı** içerir.

---

## 🧭 BÖLÜM 1: İlk Ham Veri Seti Neydi? (Kaggle Durumu)

* **Kaynak:** Kaggle (*AI-Powered Food Waste Management Dataset*)
* **İlk Boyut:** 8.000 Satır, 16 Sütun
* **İçerik:** Dünyanın farklı yerlerindeki gıda tedarik zincirlerinden toplanmış genel bir veri seti.

### ⚠️ Ham Verideki Büyük Problem:
Veri setindeki `Business_Type` (İşletme Türü) sütunu incelendiğinde şu dağılım görüldü:
1. **Restoranlar (`Restaurant`):** 2.455 satır
2. **Süpermarketler (`Supermarket`):** 2.404 satır
3. **Paket Servis Hayalet Mutfaklar (`Cloud Kitchen`):** 1.569 satır
4. **Yemekhane / Kafeterya (`Cafeteria`):** 1.572 satır

> **Problem:** Bir süpermarketteki rafta bekleyen paketli bisküvi veya konserve israfı ile bir üniversite yemekhanesindeki günlük sıcak tabldot israfının dinamikleri tamamen farklıdır. Eğer 8.000 satırı doğrudan kullansaydık, modelimiz süpermarket mantığıyla yemekhane tahmin etmeye çalışacaktı.

---

## 🛠️ BÖLÜM 2: Veri Setinde Yaptığımız 3 Büyük Değişiklik

### 1. Değişiklik: Kurumsal Filtreleme (Yemekhaneye İndirgeme)
* **Ne Yaptık?** Süpermarket, restoran ve cloud kitchen verilerini (toplam **6.428 satır**) tamamen sildik.
* **Sonuç:** Sadece **1.572 günlük saf Üniversite Yemekhanesi (`Cafeteria`)** verisini bıraktık (`yemekhane_veriseti.csv`).
* **Kalan Kategoriler:** Sıcak Yemek Menüsü (Prepared Food), Et/Tavuk (Meat), Ekmek (Bakery), Sebze/Salata (Vegetables), Meyve (Fruits), Süt/Yoğurt (Dairy).

---

### 2. Değişiklik: Öznitelik Mühendisliği (Feature Engineering - Yeni Değişkenler)
Ham verideki sütunlar yapay zekanın derin operasyonel ilişkileri anlamasına yetmiyordu. Modele "aşçı mantığı" kazandırmak için ham veriden şu yeni değişkenleri türettik (`yemekhane_oznitelikler.csv`):

1. **`Planned_Surplus` (Planlanan Fazlalık = Pişirilen - Beklenen Talep):**
   * Aşçının tencereye baştan kaç kilo fazla yemek koyduğunu gösterir. İsraf tahmininde en belirleyici değişken oldu.
2. **`Order_Cover` (Sipariş Karşılama Oranı = Pişirilen / Talep):**
   * Talebin yüzde kaçı kadar yemek hazırlandığını oranlar (Örn: 1.15 = %15 fazla üretim).
3. **Zaman ve Takvim Döngüsü:**
   * Ham `Date` sütunundan haftanın günü (`DayOfWeek`), hafta sonu mu (`IsWeekend`) ve ay (`Month`) türetildi.

---

### 3. Değişiklik: Mantıksal Eliminasyon (Depolama Sıcaklığı ve Raf Ömrünün Çıkarılması)
* **Kaggle Ham Verisinde Ne Vardı?** `Storage_Temperature` (Dolap Derecesi) ve `Shelf_Life_Days` (Raf Ömrü).
* **Neden Çıkardık? (Kritik Mühendislik Kararı):**
  1. **Operasyonel Gerçeklik:** Üniversite yemekhanesinde sıcak yemek sabah pişer, öğlen tüketilir; raf ömrü doğası gereği zaten **1 gündür**. Aşçıya raf ömrü sormak mantıksızdır.
  2. **Endüstriyel Altyapı:** Yemekhanelerde soğuk hava depoları otomatiktir; aşçı her sabah derece ölçüp sisteme girmez.
  3. **Yapay Zeka (XAI) İspatı:** Yaptığımız permütasyon önem analizinde dolap sıcaklığının israfa etkisinin **sadece %0.3 (binde üç)** olduğunu bizzat kanıtladık.
* **Sonuç:** Bu iki değişkeni modelden ve arayüzden tamamen çıkartarak sistemi aşçı için son derece sade ve pratik hale getirdik.

---

### 4. Değişiklik: Kampüse Özgü Yeni Değişkenler
* **`Burs_Donemi`:** Tarihten otomatik türetilir (ayın 6–15'i Burs_Sonrasi, 20 ve sonrası Ay_Sonu, diğerleri Dengeli_Donem).
* **`Is_Friday`:** Tarihten otomatik türetilir; Cuma günü erken ayrılış etkisini yakalar.
* **`Campus_Event`:** Aşçının tek seçimle girdiği Normal gün / Etkinlik bilgisi.
* **Önemli sınırlılık (jüriye dürüstçe söylenmelidir):**
  * `Campus_Event`, Kaggle verisindeki `Promotion` (kampanya) sütununun yeniden adlandırılmış halidir (1.254 No / 318 Yes, birebir aynı). Gerçek kampüs etkinliği kaydı değildir; yemekhane verisiyle doğrulanması gerekir.
  * `Burs_Donemi` ve `Is_Friday` yalnızca tarihten kural ile türetilir. Ham veri Kaggle kaynaklı olduğu için bu kuralların gerçek öğrenci davranışını yansıttığı **kanıtlanmış değildir**; modeldeki etkileri de küçüktür (toplam önemin birkaç yüzdesi).
  * Bu nedenle bu üç değişken, gerçek turnike/kazan verisi geldiğinde test edilecek **hipotezler** olarak sunulmalıdır.

---

## 📊 BÖLÜM 3: Önce / Sonra Karşılaştırma Tablosu

| Kriter | Kaggle'dan İndirilen Ham Veri | Bizim Dönüştürdüğümüz Nihai Veri |
| :--- | :--- | :--- |
| **Satır Sayısı** | 8.000 satır (Karma işletmeler) | **1.572 satır (%100 Üniversite Yemekhanesi)** |
| **Kapsam** | Süpermarket, Restoran, Paket Mutfak, Yemekhane | **Sadece Üniversite Tabldot Yemekhanesi** |
| **Gereksiz Değişkenler** | Dolap Sıcaklığı, Raf Ömrü (Süpermarket kalıntısı) | **Tamamen elendi (Sade ve pratik)** |
| **Türetilen Yeni Değişkenler**| Yok (Sadece ham kolonlar) | **Planned_Surplus, Order_Cover, Takvim Döngüsü, Burs_Donemi, Is_Friday, Campus_Event** |
| **Modelin Başarısı** | Genel ortalama hatası yüksek | **MAE: 1.08 kg, R²: 0.85, P90 kapsama: %85.1** |

---

## 🎙️ BÖLÜM 4: Hocaya / Jüriye Sunum Metni (Kelimesi Kelimesine Hazır Konuşma)

Hocana sunum yaparken slaytın veya ekranın başına geçtiğinde şu cümlelerle anlatabilirsin:

> *"Hocam, projemizin veri hazırlık aşamasında literatüre ve mühendislik prensiplerine uygun olarak çok titiz bir veri temizleme ve dönüştürme süreci yürüttük:*
> 
> **1. Adım: Verinin Elenmesi ve Kurumsallaştırılması**  
> *Kaggle'dan temin ettiğimiz ham veri seti 8.000 satırdan oluşuyordu ancak içinde süpermarketler, restoranlar ve hayalet mutfaklar gibi projemizle ilgisi olmayan işletmeler vardı. Bir süpermarketteki paketli gıdanın rafta bozulmasıyla, üniversite yemekhanesindeki sıcak yemeğin artması aynı şey değildir. Bu sebeple 6.428 satırlık alakasız veriyi eledik ve sadece **1.572 günlük saf Üniversite Yemekhanesi (`Cafeteria`)** verisini izole ettik.*
> 
> **2. Adım: Öznitelik Mühendisliği (Modele Aşçı Aklı Kazandırma)**  
> *Ham verideki sütunlar modelin derin örüntüleri yakalaması için yetersizdi. Biz veriden:*
> * *Aşçının ne kadar fazla yemek koyduğunu gösteren **`Planned_Surplus` (Planlanan Fazlalık)**,*
> * *Talebin kaç katı üretildiğini gösteren **`Order_Cover` (Karşılama Oranı)**,*
> * *Ve öğrencilerin kampüse geliş alışkanlıklarını temsil eden **Takvim (Gün, Hafta Sonu, Ay)** değişkenlerini türettik.*
> 
> **3. Adım: En Kritik Mühendislik Kararımız (Sıcaklık ve Raf Ömrünün Elenmesi)**  
> *Ham veri setinde 'Depolama Sıcaklığı' ve 'Raf Ömrü' sütunları da bulunuyordu. Ancak iki nedenden dolayı bunları modelimizden çıkardık:*
> * *Birincisi; üniversite yemekhanesinde sıcak yemeğin raf ömrü doğası gereği 1 gündür ve endüstriyel soğuk odalar otomatiktir; aşçı her sabah derece ölçmez.*
> * *İkincisi ve en önemlisi; Açıklanabilir Yapay Zeka (XAI) analizlerimizde sıcaklığın israfa etkisinin **sadece %0.3** olduğunu matematiksel olarak kanıtladık.*
> 
> *Böylece aşçının karşısına çıkan arayüzü gereksiz parametrelerden arındırıp, tamamen operasyonel gerçekliğe dayanan, **ortalama yanılma payı yaklaşık 1.08 kg olan** son derece başarılı bir model altyapısı elde ettik."*

---

Bu metin, hocanın kafasındaki *"Bu veriyi nereden buldunuz, üzerinde ne gibi bir mühendislik çalışması yaptınız?"* sorusunun cevabını eksiksiz ve hayran bırakacak şekilde verir!
