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
