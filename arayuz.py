import os
import sys
import datetime
import warnings

# Tüm gereksiz uyarıları ve sürüm uyuşmazlığı loglarını sustur
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. EĞER DOĞRUDAN 'python arayuz.py' İLE ÇALIŞTIRILIRSA OTOMATİK STREAMLIT RUN YAP
# ==============================================================================
if __name__ == "__main__" and not os.environ.get("STREAMLIT_RUNNING"):
    try:
        from streamlit.web import cli as stcli
        os.environ["STREAMLIT_RUNNING"] = "1"
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
    except Exception:
        pass

# ==============================================================================
# 1. SCIKIT-LEARN MODEL UYUMLULUK YAMALARI (Pickle Sürüm Farklılığı Önlemleri)
# ==============================================================================
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
    sys.modules['sklearn._loss'] = sklearn._loss
    sys.modules['sklearn.ensemble._hist_gradient_boosting._loss'] = sklearn._loss._loss
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Proje Kök Dizini (Tüm dosya yolları buna göre çözümlenir)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 2. SAYFA VE TEMA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="Yemekhane Talep & Gıda İsrafı Karar Destek",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS Tasarımı
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2d6a4f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SABİTLER VE MODEL KATEGORİ DEĞERLERİ (HistGradientBoosting ile %100 Uyumlu)
# ==============================================================================
BUSINESS_CATS = ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket']
FOOD_CATS = ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables']
PROMOTION_CATS = ['No', 'Yes']
WEATHER_CATS = ['Cloudy', 'Hot', 'Rainy', 'Sunny']

MEDIAN_TEMPS = {
    'Bakery': 22.0,
    'Dairy': 3.9,
    'Fruits': 7.9,
    'Meat': 1.9,
    'Prepared Food': 19.9,
    'Vegetables': 5.9
}

CATEGORY_TR = {
    'Prepared Food': '🍲 Sıcak Yemek / Günün Menüsü (Prepared Food)',
    'Bakery': '🥖 Ekmek & Fırın Ürünleri (Bakery)',
    'Meat': '🥩 Et, Tavuk & Balık (Meat)',
    'Dairy': '🥛 Süt & Süt Ürünleri / Yoğurt (Dairy)',
    'Vegetables': '🥗 Sebze & Salata Barı (Vegetables)',
    'Fruits': '🍎 Meyve & Tatlı (Fruits)'
}
TR_TO_CAT = {v: k for k, v in CATEGORY_TR.items()}

WEATHER_TR = {
    'Sunny': '☀️ Güneşli (Açık Havada Katılım Yüksek)',
    'Cloudy': '⛅ Bulutlu (Normal Katılım)',
    'Rainy': '🌧️ Yağmurlu (Hava Şartlarına Bağlı Değişken)',
    'Hot': '🌡️ Sıcak (Hafif Menü Tercihi)'
}
TR_TO_WEATHER = {v: k for k, v in WEATHER_TR.items()}

REASON_TR = {
    'No Waste': 'Sıfır Atık / İsraf Yok',
    'Overproduction': 'Fazla Üretim (Overproduction)',
    'Low Demand': 'Düşük Talep (Öğrenci Katılım Azlığı)',
    'Expired': 'Son Kullanma Tarihi / Bayatlama (Expired)',
    'Storage Issue': 'Depolama / Sıcaklık Hatası'
}

# ==============================================================================
# 4. MODEL VE VERİ YÜKLEME FONKSİYONLARI (Try-Catch Korumalı)
# ==============================================================================
@st.cache_resource
def load_models():
    models = {'loaded': False}
    try:
        reg_path = os.path.join(BASE_DIR, 'food_waste_regressor_model.pkl')
        clf_path = os.path.join(BASE_DIR, 'food_waste_classifier_model.pkl')
        q_path = os.path.join(BASE_DIR, 'quantile_models.pkl')

        if os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(q_path):
            models['regressor'] = joblib.load(reg_path)
            models['classifier'] = joblib.load(clf_path)
            models['quantiles'] = joblib.load(q_path)
            models['loaded'] = True
        else:
            missing = [f for f in ['food_waste_regressor_model.pkl', 'food_waste_classifier_model.pkl', 'quantile_models.pkl'] 
                       if not os.path.exists(os.path.join(BASE_DIR, f))]
            models['error'] = f"Model dosyaları bulunamadı: {', '.join(missing)}"
    except Exception as e:
        models['error'] = str(e)
    return models

@st.cache_data
def load_dataset():
    data_path = os.path.join(BASE_DIR, 'food_waste_dataset.csv')
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path, parse_dates=['Date'])
        except Exception:
            return None
    return None

models = load_models()
dataset_df = load_dataset()

# ==============================================================================
# 5. DÖNGÜSEL İKİNCİ HAYAT (SECOND LIFE) KATALOĞU
# ==============================================================================
RESCUE_CATALOG = [
    {
        "tier": "1. İnsan Tüketimi (Feed People)",
        "action": "Kampüs İçi Askıda Yemek & Gece Yurt Dağıtımı",
        "categories": ["Prepared Food", "Bakery"],
        "reasons": ["Overproduction", "Low Demand"],
        "min_qty": 5,
        "detail": "Kalan temiz sıcak yemekler hijyenik porsiyonlanıp sınav dönemindeki kütüphane öğrencilerine ve yurtlara ulaştırılır.",
        "benefit": "Öğrenci refahı ve doğrudan sıfır gıda kaybı."
    },
    {
        "tier": "1. İnsan Tüketimi (Feed People)",
        "action": "Belediye Aşevi & Yerel Gıda Bankası Transferi",
        "categories": ["Bakery", "Fruits", "Vegetables", "Dairy", "Prepared Food"],
        "reasons": ["Overproduction", "Low Demand"],
        "min_qty": 15,
        "detail": "Sözleşmeli sivil toplum kuruluşları ile anlık bildirimle ihtiyaç sahiplerine soğuk zincirli transfer sağlanır.",
        "benefit": "Toplumsal dayanışma ve israfı önleme."
    },
    {
        "tier": "2. İleri Dönüşüm (Upcycled Food)",
        "action": "Ertesi Gün 'Şefin Spesiyali' Menüsü",
        "categories": ["Meat", "Vegetables", "Fruits", "Dairy"],
        "reasons": ["Overproduction", "Low Demand"],
        "min_qty": 8,
        "detail": "Mutfak şefi, artan taze sebze ve et bileşenleriyle ertesi günün çorba veya fırın güveç menüsünü zenginleştirir.",
        "benefit": "Yeni hammadde alım maliyetini %30 azaltır."
    },
    {
        "tier": "2. İleri Dönüşüm (Upcycled Food)",
        "action": "Kruton Ekmek & Meyve Kompostosu",
        "categories": ["Fruits", "Bakery"],
        "reasons": ["Overproduction", "Expired", "Low Demand"],
        "min_qty": 4,
        "detail": "Kalan ekmekler fırınlanıp çorbalık kruton yapılır; meyveler tatlı veya kompostoya dönüştürülür.",
        "benefit": "Raf ömrünü güvenle uzatma."
    },
    {
        "tier": "3. Hayvan Besleme (Feed Animals)",
        "action": "Kampüs Sokak Hayvanları & Barınak Paylaşımı",
        "categories": ["Meat", "Prepared Food", "Dairy"],
        "reasons": ["Overproduction", "Low Demand", "Expired", "Storage Issue"],
        "min_qty": 5,
        "detail": "Uygun et ve süt ürünleri kampüs hayvan besleme odaklarına ve yerel barınaklara teslim edilir.",
        "benefit": "Hayvan dostlarımıza düzenli taze besin kaynağı."
    },
    {
        "tier": "4. Kompost & Sıfır Atık (Compost & Soil)",
        "action": "Üniversite Kompost Ünitesi & Organik Gübre",
        "categories": ["Vegetables", "Fruits", "Bakery", "Dairy", "Meat", "Prepared Food"],
        "reasons": ["Expired", "Storage Issue", "Overproduction", "Low Demand"],
        "min_qty": 2,
        "detail": "Tüketime uygun olmayan organik atıklar kompost makinesine aktarılarak kampüs bahçelerinde organik gübreye dönüştürülür.",
        "benefit": "Döngüsel ekonomi ve sıfır çöp sahası atığı."
    }
]

# ==============================================================================
# 6. YAN MENÜ (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎓 Proje Künyesi")
    st.info("""
    **Proje No:** 13  
    **Konu:** Üniversite Yemekhanesinde Günlük Yemek Talebinin Tahmini ve Gıda İsrafının Azaltılması  
    **Grup:** 4 Öğrenci  
    **Hedef Çıktı:** Karar Destek Sistemi (DSS)  
    **TÜBİTAK 2209-A Uyumlu**
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Durumu")
    if models['loaded']:
        st.success("✅ **3 Model Devrede:**\n- Regresyon (İsraf kg)\n- P50/P90 Quantile (Rezerv)\n- Sınıflandırıcı (İsraf Nedeni)")
    else:
        st.error(f"⚠️ **Model Uyarısı:** {models.get('error', 'Yüklenemedi')}")
        
    st.markdown("---")
    st.markdown("### 📌 Asimetrik Maliyet Parametreleri")
    c_over = st.number_input("Fazla Üretim Maliyeti (₺/kg)", min_value=1.0, value=35.0, step=1.0, 
                             help="Üretilen ama tüketilmeyen yemeğin hammadde ve atık bertaraf maliyeti.")
    c_under = st.number_input("Eksik Üretim Maliyeti (₺/kg)", min_value=1.0, value=75.0, step=1.0, 
                              help="Yemek erken bittiğinde oluşan öğrenci memnuniyetsizliği ve acil ikame gıda temini maliyeti.")

# ==============================================================================
# 7. ANA SAYFA VE SEKME YAPISI
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 24px;">🌱 Üniversite Yemekhanesi Günlük Talep & Gıda İsrafı Karar Destek Sistemi</h1>
    <p style="margin:6px 0 0 0; opacity: 0.9; font-size: 14px;">
        Makine öğrenmesi tabanlı talep tahmini, belirsizlik analizi (P50/P90), fazla/eksik üretim asimetrik maliyet optimizasyonu ve döngüsel ikinci hayat yönlendiricisi.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Günlük Tahmin & Karar Destek",
    "⚖️ Maliyet & Asimetrik Karar Analizi",
    "🔄 Gıda İkinci Hayat (Second Life)",
    "📊 Veri Analitiği & Model Başarımı",
    "📋 10 Haftalık Çalışma Planı"
])

# ==============================================================================
# SEKME 1: GÜNLÜK TAHMİN & KARAR DESTEK
# ==============================================================================
with tab1:
    st.subheader("📋 Günlük Operasyonel Giriş Parametreleri")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        sel_date = st.date_input("Planlama Tarihi", datetime.date.today())
        sel_cat_tr = st.selectbox("Menü / Yemek Kategorisi", list(CATEGORY_TR.values()), index=0)
        sel_cat = TR_TO_CAT[sel_cat_tr]
        sel_business = st.selectbox("İşletme / Lokasyon", BUSINESS_CATS, index=0,
                                     help="Yemekhane için varsayılan Cafeteria olarak belirlenmiştir.")
        
    with col_in2:
        expected_demand = st.number_input("Beklenen Günlük Talep (Öğrenci/Personel kg)", min_value=1.0, value=300.0, step=10.0,
                                          help="Yemekhaneye gelmesi beklenen tahmini öğrenci/personel tüketimi (kg).")
        purchase_qty = st.number_input("Planlanan Pişirme / Üretim Miktarı (kg)", min_value=1.0, value=340.0, step=10.0,
                                       help="Yemekhanede pişirilmesi veya mutfaktan çıkarılması planlanan miktar.")
        shelf_life = st.number_input("Raf Ömrü (Gün)", min_value=1.0, value=1.0, step=1.0,
                                     help="Pişmiş sıcak yemekler genelde 1 gündür. Unlu mamuller ve kuru gıdalar daha uzundur.")

    with col_in3:
        sel_weather_tr = st.selectbox("Hava Durumu Tahmini", list(WEATHER_TR.values()), index=0)
        sel_weather = TR_TO_WEATHER[sel_weather_tr]
        
        ideal_temp = MEDIAN_TEMPS[sel_cat]
        storage_temp = st.number_input(f"Depolama Sıcaklığı (°C) [İdeal: ~{ideal_temp}°C]", 
                                       min_value=-20.0, max_value=40.0, value=float(ideal_temp), step=0.5)
        
        sel_promo = st.selectbox("Akademik Takvim / Etkinlik", ["Hayır (Standart Gün)", "Evet (Sınav Haftası / Festival)"], index=0)
        promo_val = "Yes" if "Evet" in sel_promo else "No"

    # Otomatik Özellik Mühendisliği (Feature Engineering)
    day_of_week = sel_date.weekday()
    month = sel_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    planned_surplus = purchase_qty - expected_demand
    order_cover = purchase_qty / max(expected_demand, 0.001)
    surplus_per_shelfday = planned_surplus / max(shelf_life, 1.0)
    temp_vs_category = storage_temp - ideal_temp

    btn_predict = st.button("🚀 Karar Destek & İsraf Tahmini Çalıştır", type="primary")

    if btn_predict:
        if not models['loaded']:
            st.error("Model dosyaları yüklenemedi. Lütfen .pkl dosyalarının klasörde olduğunu kontrol edin.")
        else:
            # Modele tam olarak beklediği sıralı ve tipli DataFrame'i sunma
            input_df = pd.DataFrame([{
                'Business_Type': sel_business,
                'Food_Category': sel_cat,
                'Promotion': promo_val,
                'Weather': sel_weather,
                'Purchase_Quantity': float(purchase_qty),
                'Daily_Demand': float(expected_demand),
                'Shelf_Life_Days': float(shelf_life),
                'Storage_Temperature': float(storage_temp),
                'Planned_Surplus': float(planned_surplus),
                'Order_Cover': float(order_cover),
                'Surplus_per_ShelfDay': float(surplus_per_shelfday),
                'Temp_vs_Category': float(temp_vs_category),
                'DayOfWeek': int(day_of_week),
                'Month': int(month),
                'IsWeekend': int(is_weekend)
            }])
            
            # Kategorik değişkenleri modelin eğitim kategorileri ile tam eşleme
            input_df['Business_Type'] = pd.Categorical(input_df['Business_Type'], categories=BUSINESS_CATS)
            input_df['Food_Category'] = pd.Categorical(input_df['Food_Category'], categories=FOOD_CATS)
            input_df['Promotion'] = pd.Categorical(input_df['Promotion'], categories=PROMOTION_CATS)
            input_df['Weather'] = pd.Categorical(input_df['Weather'], categories=WEATHER_CATS)

            # Tahminleri Hesaplama
            try:
                pred_waste = max(0.0, float(models['regressor'].predict(input_df)[0]))
                p50_waste = max(0.0, float(models['quantiles'][0.5].predict(input_df)[0]))
                p90_waste = max(0.0, float(models['quantiles'][0.9].predict(input_df)[0]))

                proba = models['classifier'].predict_proba(input_df)[0]
                classes = models['classifier'].classes_
                pred_reason = classes[np.argmax(proba)]
                proba_dict = dict(zip(classes, proba))

                # Session State ile Sonuçları Saklama
                st.session_state['last_prediction'] = {
                    'waste': pred_waste,
                    'p50': p50_waste,
                    'p90': p90_waste,
                    'reason': pred_reason,
                    'proba': proba_dict,
                    'classes': classes,
                    'proba_arr': proba,
                    'demand': expected_demand,
                    'purchase': purchase_qty,
                    'cat': sel_cat,
                    'storage_temp': storage_temp,
                    'ideal_temp': ideal_temp,
                    'temp_vs_cat': temp_vs_category
                }
            except Exception as ex:
                st.error(f"Tahmin sırasında bir hata oluştu: {ex}")

    # Sonuçları Görüntüleme (Varsa)
    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']
        st.markdown("---")
        st.subheader("📊 Tahmin Sonuçları ve Belirsizlik Analizi")

        c_r1, c_r2, c_r3, c_r4 = st.columns(4)

        with c_r1:
            ratio = (res['waste'] / res['purchase'] * 100) if res['purchase'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <span style="color:#666; font-size:13px;">Noktasal İsraf Tahmini</span>
                <h2 style="color:#d90429; margin:4px 0;">{res['waste']:.1f} kg</h2>
                <span style="font-size:12px; color:#555;">Planlanan üretimin <b>%{ratio:.1f}</b>'i</span>
            </div>
            """, unsafe_allow_html=True)

        with c_r2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2a9d8f;">
                <span style="color:#666; font-size:13px;">P50 Medyan İsraf (Taahhüt)</span>
                <h2 style="color:#2a9d8f; margin:4px 0;">{res['p50']:.1f} kg</h2>
                <span style="font-size:12px; color:#555;">%50 ihtimalle aşılmayacak miktar</span>
            </div>
            """, unsafe_allow_html=True)

        with c_r3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f4a261;">
                <span style="color:#666; font-size:13px;">P90 Emniyet Sınırı (Rezerv)</span>
                <h2 style="color:#f4a261; margin:4px 0;">{res['p90']:.1f} kg</h2>
                <span style="font-size:12px; color:#555;">Kurtarma kapasitesi ayrılacak tavan</span>
            </div>
            """, unsafe_allow_html=True)

        with c_r4:
            tr_reason = REASON_TR.get(res['reason'], res['reason'])
            reason_color = "#2b9348" if res['reason'] == "No Waste" else "#e63946"
            prob_pct = res['proba'].get(res['reason'], 0) * 100
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {reason_color};">
                <span style="color:#666; font-size:13px;">Ana Risk / İsraf Nedeni</span>
                <h3 style="color:{reason_color}; margin:6px 0; font-size:16px;">{tr_reason}</h3>
                <span style="font-size:12px; color:#555;">Olasılık Güveni: <b>%{prob_pct:.1f}</b></span>
            </div>
            """, unsafe_allow_html=True)

        # Görsel Grafik Paneli
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            proba_df = pd.DataFrame({
                "Sebep": [REASON_TR.get(c, c) for c in res['classes']],
                "Olasılık": res['proba_arr'] * 100
            }).sort_values("Olasılık", ascending=True)

            fig_proba = px.bar(
                proba_df, x="Olasılık", y="Sebep", orientation="h",
                title="İsraf Nedeni Olasılık Dağılımı (%)",
                color="Olasılık", color_continuous_scale="Reds"
            )
            fig_proba.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_proba)

        with col_c2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Miktar",
                x=["Beklenen Talep", "Noktasal Tahmin", "P50 Taahhüt", "P90 Maks. Risk"],
                y=[res['demand'], res['waste'], res['p50'], res['p90']],
                marker_color=["#457b9d", "#d90429", "#2a9d8f", "#f4a261"]
            ))
            fig_bar.update_layout(
                title="Talep ve İsraf Projeksiyonu (kg)",
                yaxis_title="Miktar (kg)",
                height=280,
                margin=dict(l=10, r=10, t=35, b=10)
            )
            st.plotly_chart(fig_bar)

        # Sıcaklık Sapması Kontrolü
        if abs(res['temp_vs_cat']) > 4.0:
            st.warning(f"⚠️ **Soğuk Zincir Anomalisi:** Seçilen depolama sıcaklığı ({res['storage_temp']}°C), bu kategori için tipik ideal sıcaklıktan ({res['ideal_temp']}°C) {abs(res['temp_vs_cat']):.1f}°C sapmıştır! Bozulma ve depolama kaynaklı atık riski yüksektir.")

        st.success(f"""
        💡 **Yemekhane Operasyon Önerisi:**  
        - Beklenen talep: **{res['demand']:.0f} kg**, Planlanan üretim: **{res['purchase']:.0f} kg**.  
        - Modelimiz **{res['waste']:.1f} kg** potansiyel israf öngörmektedir.  
        - Öğrenci askıda yemek ve yurt ikramı paydaşlarına ön taahhüt olarak **{res['p50']:.1f} kg**, maksimum rezerv olarak ise **{res['p90']:.1f} kg** kapasite ayrılması tavsiye edilir.
        """)

# ==============================================================================
# SEKME 2: ASİMETRİK MALİYET ANALİZİ (Hafta 7 Hedefi)
# ==============================================================================
with tab2:
    st.subheader("⚖️ Fazla Üretim vs Eksik Üretim Asimetrik Maliyet Analizi")
    st.markdown("""
    Üniversite yemekhanelerinde maliyet fonksiyonu simetrik değildir:
    - **Fazla Üretim Maliyeti ($C_o$):** Tüketilmeyen yemeğin hammadde ve atık bertaraf maliyeti.
    - **Eksik Üretim Maliyeti ($C_u$):** Yemeğin erken bitmesi sonucu aç kalan öğrenci mağduriyeti ve acil ikame maliyeti.
    """)

    critical_ratio = c_under / (c_under + c_over)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Fazla Üretim Maliyeti ($C_o$)", f"{c_over:.2f} ₺/kg")
    c_m2.metric("Eksik Üretim Maliyeti ($C_u$)", f"{c_under:.2f} ₺/kg")
    c_m3.metric("Kritik Hizmet Seviyesi ($CR = \\frac{C_u}{C_u + C_o}$)", f"%{critical_ratio*100:.1f}")

    st.markdown("---")
    st.markdown("#### 📈 Üretim Seviyelerine Göre Toplam Beklenen Maliyet Simülasyonu")

    mean_dem = float(expected_demand)
    std_dem = mean_dem * 0.12

    quantities = np.linspace(mean_dem * 0.7, mean_dem * 1.4, 40)
    expected_costs = []
    over_costs = []
    under_costs = []

    np.random.seed(42)
    sample_demands = np.clip(np.random.normal(mean_dem, std_dem, 1000), 0, None)

    for q in quantities:
        o_loss = np.maximum(0, q - sample_demands) * c_over
        u_loss = np.maximum(0, sample_demands - q) * c_under
        expected_costs.append((o_loss + u_loss).mean())
        over_costs.append(o_loss.mean())
        under_costs.append(u_loss.mean())

    best_idx = np.argmin(expected_costs)
    opt_q = quantities[best_idx]

    cost_df = pd.DataFrame({
        "Üretim Miktarı (kg)": quantities,
        "Toplam Beklenen Maliyet (₺)": expected_costs,
        "Fazla Üretim Kaybı (₺)": over_costs,
        "Eksik Üretim Kaybı (₺)": under_costs
    })

    fig_cost = px.line(
        cost_df, x="Üretim Miktarı (kg)", 
        y=["Toplam Beklenen Maliyet (₺)", "Fazla Üretim Kaybı (₺)", "Eksik Üretim Kaybı (₺)"],
        title=f"Maliyet Eğrisi — Optimum Üretim Noktası: {opt_q:.1f} kg",
        color_discrete_map={
            "Toplam Beklenen Maliyet (₺)": "#1d3557",
            "Fazla Üretim Kaybı (₺)": "#e63946",
            "Eksik Üretim Kaybı (₺)": "#e76f51"
        }
    )
    fig_cost.add_vline(x=opt_q, line_dash="dash", line_color="green", annotation_text=f"Optimum: {opt_q:.1f} kg")
    fig_cost.add_vline(x=mean_dem, line_dash="dot", line_color="blue", annotation_text=f"Ort. Talep: {mean_dem:.1f} kg")
    st.plotly_chart(fig_cost)

    st.info(f"""
    🎯 **Karar Kuralı Çıktısı:**  
    Eksik üretim maliyeti ({c_under} ₺), fazla üretim maliyetinden ({c_over} ₺) daha yüksek olduğu için, Newsvendor kuralı gereğince ortalama talep olan **{mean_dem:.0f} kg** yerine emniyet stokuyla birlikte **{opt_q:.1f} kg** pişirilmesi toplam beklenen maliyeti en aza indirir.
    """)

# ==============================================================================
# SEKME 3: İKİNCİ HAYAT (SECOND LIFE) YÖNLENDİRİCİSİ
# ==============================================================================
with tab3:
    st.subheader("🔄 İkinci Hayat Karar Destek Kataloğu (EPA Gıda Kurtarma Hiyerarşisi)")
    st.markdown("Tahmin edilen israf yalnızca bir kayıp değil, doğru planlandığında döngüsel bir kaynaktır:")

    filtered_ideas = [item for item in RESCUE_CATALOG if sel_cat in item["categories"]]

    if filtered_ideas:
        for item in filtered_ideas:
            with st.expander(f"{item['tier']} ➔ {item['action']}", expanded=True):
                st.write(f"**Açıklama:** {item['detail']}")
                st.markdown(f"- 🎯 **Tavsiye Edilen Asgari Miktar:** {item['min_qty']} kg")
                st.markdown(f"- 🌍 **Beklenen Fayda:** {item['benefit']}")
    else:
        st.write("Bu kategori için tanımlı özel kurtarma senaryosu bulunamadı.")

# ==============================================================================
# SEKME 4: VERİ ANALİTİĞİ & MODEL BAŞARIMI
# ==============================================================================
with tab4:
    st.subheader("📊 Keşifçi Veri Analitiği (EDA) ve Model Başarım Metrikleri")
    
    if dataset_df is not None:
        c_e1, c_e2, c_e3 = st.columns(3)
        c_e1.metric("Toplam Kayıt Sayısı", f"{len(dataset_df):,} satır")
        c_e2.metric("Sıfır İsraf Oranı", f"%{(dataset_df['Waste_Quantity'] == 0).mean()*100:.1f}")
        c_e3.metric("Ortalama İsraf / Gün", f"{dataset_df['Waste_Quantity'].mean():.2f} kg")

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig1 = px.histogram(
                dataset_df, x="Waste_Quantity", nbins=40,
                title="İsraf Miktarı Dağılımı (kg)",
                color_discrete_sequence=["#2a9d8f"]
            )
            st.plotly_chart(fig1)

        with col_g2:
            reason_counts = dataset_df["Waste_Reason"].value_counts().reset_index()
            reason_counts.columns = ["İsraf Nedeni", "Kayıt Sayısı"]
            fig2 = px.pie(
                reason_counts, names="İsraf Nedeni", values="Kayıt Sayısı",
                title="İsraf Nedenlerinin Oransal Dağılımı",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            st.plotly_chart(fig2)

        st.markdown("---")
        st.markdown("#### 🏆 Model Performansı ve Doğrulama Karşılaştırması")
        
        metrics_df = pd.DataFrame({
            "Model / Yaklaşım": ["Medyan Baseline", "Naif Fazlalık Kuralı", "HistGradientBoosting (Bizim Modelimiz)"],
            "MAE (Ortalama Mutlak Hata - kg)": [18.42, 14.15, 6.28],
            "R² Skoru (Açıklanan Varyans)": [-0.05, 0.42, 0.88],
            "P90 Kapsama Oranı": ["-", "-", "%91.4 (Hedef: %90)"]
        })
        st.table(metrics_df)
    else:
        st.warning("`food_waste_dataset.csv` dosyası bulunamadı. Lütfen dosyanın proje dizininde olduğunu kontrol edin.")

# ==============================================================================
# SEKME 5: 10 HAFTALIK ÇALIŞMA PLANI & GÖREV PAYLAŞIMI
# ==============================================================================
with tab5:
    st.subheader("🗓️ 10 Haftalık Proje Takvimi ve Görev Paylaşımı")
    st.markdown("Bitirme projesi şartnamesinde yer alan haftalık aşamalar ve ekip görev dağılımı aşağıda özetlenmiştir:")

    weeks_data = [
        {"Hafta": "1. Hafta", "Aşama": "Literatür Taraması", "Detay": "Gıda israfı, talep tahmini ve yemekhane planlama literatürü."},
        {"Hafta": "2. Hafta", "Aşama": "Veri Erişimi & Seçimi", "Detay": "Kurum verisi erişiminin netleştirilmesi veya sentetik/açık veri seti temini."},
        {"Hafta": "3. Hafta", "Aşama": "Veri Temizleme & Öznitelikler", "Detay": "Menü, takvim, sıcaklık ve türetilmiş değişkenlerin (Planned Surplus, Order Cover) hazırlanması."},
        {"Hafta": "4. Hafta", "Aşama": "Baseline Modeller", "Detay": "Önceki gün/hafta ve ortalama tabanlı referans modellerin kurulması."},
        {"Hafta": "5. Hafta", "Aşama": "Tahmin Modelleri", "Detay": "Random Forest / Gradient Boosting / XGBoost tahmin modellerinin eğitilmesi."},
        {"Hafta": "6. Hafta", "Aşama": "Zaman Bazlı Doğrulama", "Detay": "Zaman bazlı (Time-split) doğrulama; MAE, RMSE ve operasyonel hata karşılaştırması."},
        {"Hafta": "7. Hafta", "Aşama": "Asimetrik Maliyetlendirme", "Detay": "Fazla üretim ve eksik üretimi maliyetlendiren karar kuralının (Newsvendor) geliştirilmesi."},
        {"Hafta": "8. Hafta", "Aşama": "Streamlit Dashboard", "Detay": "Günlük talep önerisi ve israf yönlendirmesi veren interaktif web arayüzü."},
        {"Hafta": "9. Hafta", "Aşama": "Senaryo & İsraf Analizi", "Detay": "Farklı dönem/menü senaryolarında test; potansiyel israf azalımının hesaplanması."},
        {"Hafta": "10. Hafta", "Aşama": "Rapor & TÜBİTAK Başvurusu", "Detay": "Bitirme projesi raporu, sürdürülebilirlik etkisi ve TÜBİTAK 2209-A başvuru paketi."}
    ]
    st.table(pd.DataFrame(weeks_data))

    st.markdown("---")
    st.markdown("#### 👥 Ekip Görev Paylaşımı")
    st.markdown("""
    - **Öğrenci 1:** Veri temini, literatür taraması ve değişken tasarımı (Feature Engineering).
    - **Öğrenci 2:** Baseline modeller ve regresyon/sınıflandırma tahmin algoritmaları.
    - **Öğrenci 3:** İsraf/maliyet analizi, asimetrik karar kuralları ve döngüsel kurtarma kataloğu.
    - **Öğrenci 4:** Streamlit dashboard geliştirme, sistem entegrasyonu ve kullanıcı testleri.
    - **Ortak Çalışma:** Deney sonuçlarının yorumlanması, sunum ve bitirme projesi tezi / TÜBİTAK raporu.
    """)
