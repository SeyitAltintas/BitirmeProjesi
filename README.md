# 🍽️ Yemekhane Talep & Gıda İsrafı Tahmin ve Karar Destek Sistemi

Bu proje, üniversite/kurum yemekhanelerinde oluşan gıda israfını en aza indirmek ve günlük porsiyon talebini doğru tahmin etmek amacıyla makine öğrenmesi modelleriyle desteklenmiş etkileşimli bir karar destek sistemidir.

## 🚀 Özellikler
- **Talep ve İsraf Tahmini:** Sınıflandırma, regresyon ve kantil regresyon modelleriyle porsiyon talebi ve olası israf miktarı analizi.
- **İnteraktif Arayüz:** Streamlit tabanlı modern ve kullanıcı dostu kontrol paneli.
- **Görselleştirme:** Plotly ile dinamik grafikler, risk dağılımı ve senaryo analizleri.

## 🛠️ Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/SeyitAltintas/BitirmeProjesi.git
cd BitirmeProjesi
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run arayuz.py
```
*(Windows kullanıcıları doğrudan `baslat.bat` dosyasını çalıştırabilir.)*

## 📁 Proje Yapısı
- `arayuz.py`: Streamlit karar destek sistemi web arayüzü
- `food_waste_dataset.csv`: Eğitim ve analiz veri seti
- `*.pkl`: Eğitilmiş makine öğrenmesi modelleri (sınıflandırıcı, regresör, kantil modelleri)
- `requirements.txt`: Bağımlılıklar listesi
- `baslat.bat`: Windows için hızlı başlatma betiği

---

## 👥 Ekip Olarak Ortak Çalışma Rehberi (Adım Adım)

Bu bölüm, projeyi iki veya daha fazla kişi geliştirirken kodların karışmaması ve senkronize kalınması için hazırlanmıştır.

### 1. Adım: Arkadaşı Depoya Ortak (Collaborator) Olarak Ekleme *(Tek Seferlik)*
Arkadaşınızın kod gönderebilmesi (`git push`) için depoya yazma izninin olması gerekir:
1. **Seyit (Depo Sahibi):** GitHub'da depoya gidin: [github.com/SeyitAltintas/BitirmeProjesi](https://github.com/SeyitAltintas/BitirmeProjesi)
2. Yukarıdaki sekmelerden **Settings** > sol menüden **Collaborators** seçeneğine tıklayın.
3. **Add people** butonuna basarak arkadaşınızın GitHub kullanıcı adını veya e-postasını yazıp davet gönderin.
4. **Arkadaşı:** E-postasına gelen bağlantıdan veya GitHub bildirimlerinden **Accept Invitation** (Daveti Kabul Et) butonuna tıklamalıdır.

---

### 2. Adım: Arkadaşın Projeyi Bilgisayarına İndirmesi *(Tek Seferlik)*
Arkadaşınız bilgisayarında bir klasör açıp (örneğin Masaüstü) terminal/komut satırında şu komutları çalıştırır:

```bash
# 1. Depoyu bilgisayara klonlayın (indirin)
git clone https://github.com/SeyitAltintas/BitirmeProjesi.git

# 2. Proje klasörünün içine girin
cd BitirmeProjesi

# 3. Gerekli Python kütüphanelerini kurun
pip install -r requirements.txt

# 4. Projeyi çalıştırıp test edin
streamlit run arayuz.py
```

---

### 3. Adım: Günlük Çalışma Döngüsü (Altın Kurallar)

#### ⚠️ Kural 1: Kod Yazmaya Başlamadan Önce HER ZAMAN Çekin!
Arkadaşınız bir şey yüklemiş olabilir. Bilgisayarınızdaki kodun güncel olduğundan emin olmak için kod yazmaya başlamadan önce terminalde mutlaka şunu çalıştırın:
```bash
git pull
```

#### 🚀 Kural 2: Kodunuzu Yazdıktan Sonra GitHub'a Gönderme
Kodunuzu yazdınız, test ettiniz ve çalıştığından emin oldunuz. Sırasıyla şu 3 komutu verin:
```bash
# 1. Yapılan tüm değişiklikleri ekleyin
git add .

# 2. Ne yaptığınızı anlatan bir mesaj ile kaydedin
git commit -m "Arayuze yeni filtreleme butonu eklendi"

# 3. GitHub'a yukleyin
git push
```

---

### 4. Adım: Daha Güvenli Çalışma Yöntemi (Branch / Dal Kullanımı)
İkiniz aynı anda farklı dosyalar üzerinde çalışırken birbirinizin kodunu ezmemek için **Branch (Dal)** açarak çalışmanız önerilir:

1. **Yeni bir dal açın ve o dala geçin:**
   ```bash
   git checkout -b ozellik-grafikler
   ```
2. **Kodlarınızı yazın, kaydedin ve commit edin:**
   ```bash
   git add .
   git commit -m "Grafik tasarimi yenilendi"
   ```
3. **Kendi dalınızı GitHub'a gönderin:**
   ```bash
   git push origin ozellik-grafikler
   ```
4. **GitHub'da Birleştirme (Pull Request):**
   - GitHub'a girdiğinizde sarı bir bildirimle **"Compare & pull request"** butonu görürsünüz.
   - Bu butona basıp birleştirme isteği açın. Diğer kişi kodu inceleyip **"Merge pull request"** diyerek ana koda (`main`) ekler.
5. **Ana dala geri dönmek ve güncel hali almak için:**
   ```bash
   git checkout main
   git pull
   ```

---

### 5. Adım: Çakışma (Merge Conflict) Olursa Ne Yapmalı?
İkiniz de aynı dosyanın aynı satırını değiştirip göndermeye çalışırsanız Git `CONFLICT (content)` uyarısı verir.

**Çözümü çok basittir:**
1. VS Code'da çakışan dosyayı açtığınızda Git size şu seçenekleri sunar:
   - *Accept Current Change* (Sizin yazdığınız kalsın)
   - *Accept Incoming Change* (Arkadaşınızın yazdığı kalsın)
   - *Accept Both Changes* (Her ikisi de kalsın)
2. Doğru olanı seçip dosyayı kaydedin.
3. Ardından tekrar:
   ```bash
   git add .
   git commit -m "Cakisma cozuldu"
   git push
   ```

---

### 📋 Hızlı Komut Özeti (Cheat Sheet)

| Komut | Ne İşe Yarar? |
|---|---|
| `git pull` | GitHub'daki en güncel kodları bilgisayarınıza indirir. |
| `git status` | Hangi dosyalarda değişiklik yaptığınızı listeler. |
| `git add .` | Değiştirdiğiniz dosyaları göndermeye hazırlar. |
| `git commit -m "mesaj"` | Değişiklikleri yerel olarak onaylar ve etiketler. |
| `git push` | Commit edilmiş kodları GitHub'a yükler. |
| `git checkout -b <dal-adi>` | Yeni bir çalışma dalı (branch) açar. |
| `git checkout main` | Ana dala geri döner. |
