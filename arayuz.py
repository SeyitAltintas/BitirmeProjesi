import os
import sys
import datetime
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__" and not os.environ.get("STREAMLIT_RUNNING"):
    try:
        from streamlit.web import cli as stcli
        os.environ["STREAMLIT_RUNNING"] = "1"
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
COMP_DIR = os.path.join(MODEL_DIR, 'comparison')

# ==============================================================================
# 1. SAYFA VE TEMA (SADE / MİNİMAL)
# ==============================================================================
st.set_page_config(
    page_title="Yemekhane Talep & İsraf Karar Destek",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ACCENT = "#2f6f5e"      # tek vurgu rengi
MUTED = "#6b7280"
LINE = "#e5e7eb"
NEG = "#b4443c"         # yalnızca uyarı / olumsuz

st.markdown(f"""
<style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
    .block-container {{ max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem; }}
    h1, h2, h3, h4 {{ font-weight: 600; letter-spacing: -0.01em; }}
    .app-title {{ font-size: 1.6rem; font-weight: 650; margin: 0; }}
    .app-sub {{ color: {MUTED}; font-size: 0.92rem; margin: 4px 0 18px 0; }}
    .card {{
        border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px;
        background: transparent; height: 100%;
    }}
    .card .lbl {{ color: {MUTED}; font-size: 0.8rem; }}
    .card .val {{ font-size: 1.7rem; font-weight: 600; margin: 2px 0; }}
    .card .note {{ color: {MUTED}; font-size: 0.78rem; }}
    .chip {{
        display: inline-block; border: 1px solid {LINE}; border-radius: 999px;
        padding: 3px 12px; margin: 0 6px 6px 0; font-size: 0.82rem; color: {MUTED};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{ padding: 8px 14px; font-weight: 500; }}
    .stButton > button[kind="primary"] {{
        background: {ACCENT}; border: none; border-radius: 8px; padding: 0.55rem 1.4rem;
    }}
    .stButton > button {{ border-radius: 8px; }}
    hr {{ margin: 1.4rem 0; border-color: {LINE}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="simple_white", margin=dict(l=10, r=10, t=40, b=10),
    font=dict(size=13), height=300
)

def card(label, value, note="", color=None):
    style = f"color:{color};" if color else ""
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val" style="{style}">{value}</div>'
            f'<div class="note">{note}</div></div>')

# ==============================================================================
# 2. SABİTLER
# ==============================================================================
BUSINESS_TYPE = "Cafeteria"
BUSINESS_CATS = ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket']
FOOD_CATS = ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables']
CAMPUS_EVENT_CATS = ['No', 'Yes']
WEATHER_CATS = ['Cloudy', 'Hot', 'Rainy', 'Sunny']
FEATURE_COLS = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'Purchase_Quantity', 'Daily_Demand', 'Planned_Surplus', 'Order_Cover',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]
CAT_COLS = ['Business_Type', 'Food_Category', 'Campus_Event', 'Weather']

CATEGORY_TR = {
    'Prepared Food': 'Sıcak Yemek / Günün Menüsü',
    'Bakery': 'Ekmek & Fırın Ürünleri',
    'Meat': 'Et, Tavuk & Balık Menüsü',
    'Dairy': 'Süt & Yoğurt / Ayran',
    'Vegetables': 'Sebze Yemeği & Salata Barı',
    'Fruits': 'Meyve & Tatlı'
}
TR_TO_CAT = {v: k for k, v in CATEGORY_TR.items()}

WEATHER_TR = {
    'Sunny': 'Güneşli', 'Cloudy': 'Bulutlu', 'Rainy': 'Yağmurlu', 'Hot': 'Çok Sıcak'
}
TR_TO_WEATHER = {v: k for k, v in WEATHER_TR.items()}

REASON_TR = {
    'No Waste': 'İsraf Yok',
    'Overproduction': 'Fazla Üretim',
    'Low Demand': 'Düşük Talep',
    'Expired': 'Bayatlama / SKT',
    'Storage Issue': 'Depolama Sorunu'
}

# ==============================================================================
# 3. MODEL VE VERİ YÜKLEME
# ==============================================================================
MODEL_FILES = ['food_demand_model.pkl', 'food_waste_regressor_model.pkl', 'food_waste_classifier_model.pkl',
               'quantile_models.pkl', 'rf_model.pkl', 'xgb_model.pkl', 'catboost_model.pkl']

def get_models_mtime():
    return tuple(os.path.getmtime(os.path.join(MODEL_DIR, f))
                 for f in MODEL_FILES if os.path.exists(os.path.join(MODEL_DIR, f)))

@st.cache_resource
def load_models(_mtime_hash):
    models = {'loaded': False, 'extra': {}}
    try:
        dem_path = os.path.join(MODEL_DIR, 'food_demand_model.pkl')
        reg_path = os.path.join(MODEL_DIR, 'food_waste_regressor_model.pkl')
        clf_path = os.path.join(MODEL_DIR, 'food_waste_classifier_model.pkl')
        q_path = os.path.join(MODEL_DIR, 'quantile_models.pkl')
        if os.path.exists(dem_path) and os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(q_path):
            models['demand'] = joblib.load(dem_path)
            models['regressor'] = joblib.load(reg_path)
            models['classifier'] = joblib.load(clf_path)
            models['quantiles'] = joblib.load(q_path)
            models['loaded'] = True
        else:
            models['error'] = "Model dosyaları (.pkl) bulunamadı!"
    except Exception as e:
        models['error'] = str(e)
    for key, fname in [('RandomForest', 'rf_model.pkl'), ('XGBoost', 'xgb_model.pkl'),
                       ('CatBoost', 'catboost_model.pkl')]:
        try:
            p = os.path.join(COMP_DIR, fname)
            if os.path.exists(p):
                models['extra'][key] = joblib.load(p)
        except Exception:
            pass
    return models

@st.cache_data
def load_dataset():
    data_path = os.path.join(DATA_DIR, 'yemekhane_oznitelikler.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, 'yemekhane_veriseti.csv')
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path, parse_dates=['Date'])
        except Exception:
            return None
    return None

@st.cache_data
def load_comparison():
    p = os.path.join(COMP_DIR, 'model_karsilastirma.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_demand_comparison():
    p = os.path.join(COMP_DIR, 'talep_model_karsilastirma.csv')
    return pd.read_csv(p) if os.path.exists(p) else None

models = load_models(get_models_mtime())
dataset_df = load_dataset()

def predict_extra(name, raw_df):
    """RF / XGBoost / CatBoost için tek satırlık israf tahmini (kg)."""
    m = models['extra'][name]
    if name == 'CatBoost':
        x = raw_df[FEATURE_COLS].copy()
        for c in CAT_COLS:
            x[c] = x[c].astype(str)
        return max(0.0, float(m['reg'].predict(x)[0]))
    x = raw_df[FEATURE_COLS].copy()
    x[CAT_COLS] = m['encoder'].transform(x[CAT_COLS])
    return max(0.0, float(m['reg'].predict(x.astype(float))[0]))


def to_model_df(raw_df):
    out = raw_df.copy()
    for col, cats in [('Business_Type', BUSINESS_CATS), ('Food_Category', FOOD_CATS),
                      ('Campus_Event', CAMPUS_EVENT_CATS), ('Weather', WEATHER_CATS)]:
        out[col] = pd.Categorical(out[col], categories=cats)
    return out

FEATURE_GROUPS = {
    "Planlanan fazlalık": ['Planned_Surplus'],
    "Pişirilen miktar": ['Purchase_Quantity'],
    "Öğrenci talebi": ['Daily_Demand'],
    "Sipariş karşılama oranı": ['Order_Cover'],
    "Cuma & takvim": ['Is_Friday', 'DayOfWeek', 'IsWeekend', 'Month'],
    "Kampüs etkinliği": ['Campus_Event'],
    "Hava durumu": ['Weather'],
    "Menü kategorisi": ['Food_Category', 'Business_Type'],
}

@st.cache_data
def compute_global_importance(_mtime_hash):
    """Üretim modelinin test dönemi (Ekim-Aralık) üzerinde gerçek permütasyon önemi."""
    from sklearn.inspection import permutation_importance
    df = load_dataset()
    if df is None or not models['loaded']:
        return None
    test = df[df['Date'].dt.month >= 10]
    X = to_model_df(test[FEATURE_COLS])
    y = test['Waste_Quantity'].values
    r = permutation_importance(models['regressor'], X, y, n_repeats=8, random_state=42,
                               scoring='neg_mean_absolute_error')
    imp = dict(zip(FEATURE_COLS, r.importances_mean))
    grouped = {g: max(0.0, sum(imp[c] for c in cols)) for g, cols in FEATURE_GROUPS.items()}
    total = sum(grouped.values()) or 1.0
    return {g: v / total * 100 for g, v in grouped.items()}

# ==============================================================================
# 4. İKİNCİ HAYAT KATALOĞU
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
# 5. BAŞLIK
# ==============================================================================
st.markdown("""
<p class="app-title">Üniversite Yemekhanesi Talep & Gıda İsrafı Karar Destek Sistemi</p>
<p class="app-sub">Günlük israf tahmini · P50/P90 belirsizlik analizi · Açıklanabilir yapay zeka · Model karşılaştırması
&nbsp;|&nbsp; Proje No: 13 · TÜBİTAK 2209-A</p>
""", unsafe_allow_html=True)

if not models['loaded']:
    st.error(f"Model uyarısı: {models.get('error', 'Yüklenemedi')}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Günlük Tahmin",
    "Maliyet Analizi",
    "İkinci Hayat",
    "Veri & Model Başarımı",
    "Model Karşılaştırması"
])

# ==============================================================================
# SEKME 1: GÜNLÜK TAHMİN
# ==============================================================================
with tab1:
    st.markdown("#### 📋 1. Aşama: Mutfak Operasyonel Koşulları")
    st.caption("Aşçı yalnızca tarihi, menüyü ve hava durumunu seçer; yapay zeka beklenen talebi ve ideal pişirme miktarını otomatik hesaplar.")

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        sel_date = st.date_input("Planlama tarihi", datetime.date.today())
        sel_cat_tr = st.selectbox("Menü kategorisi", list(CATEGORY_TR.values()), index=0)
        sel_cat = TR_TO_CAT[sel_cat_tr]
    with col_in2:
        sel_weather_tr = st.selectbox("Hava durumu tahmini", list(WEATHER_TR.values()), index=0)
        sel_weather = TR_TO_WEATHER[sel_weather_tr]
        sel_event = st.selectbox("Kampüs etkinliği / özel dönem",
                                 ["Normal gün", "Etkinlik / sınav haftası / şenlik"], index=0)
        event_val = "Yes" if sel_event.startswith("Etkinlik") else "No"

    # Otomatik takvim değişkenleri
    day_of_month = sel_date.day
    day_of_week = sel_date.weekday()
    month = sel_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_friday = 1 if day_of_week == 4 else 0

    friday_txt = "Cuma (öğleden sonra erken ayrılış eğilimi)" if is_friday else "Standart hafta içi"
    st.markdown(
        f'<span class="chip">Akademik takvim: {friday_txt}</span>'
        f'<span class="chip">Otomatik analiz edildi</span>',
        unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 🤖 1. AŞAMA: YAPAY ZEKA TALEP TAHMİNİ
    # -------------------------------------------------------------
    dem_raw_df = pd.DataFrame([{
        'Business_Type': BUSINESS_TYPE,
        'Food_Category': sel_cat,
        'Campus_Event': event_val,
        'Weather': sel_weather,
        'DayOfWeek': int(day_of_week),
        'Month': int(month),
        'IsWeekend': int(is_weekend),
        'Is_Friday': int(is_friday)
    }])
    dem_input_df = to_model_df(dem_raw_df)
    
    if models.get('demand') is not None:
        auto_dem = max(10.0, float(models['demand'].predict(dem_input_df)[0]))
    else:
        auto_dem = 62.0

    # Newsvendor emniyetli üretim miktarı (öğrenci aç kalmasın diye %9 emniyet payı)
    rec_purchase = auto_dem * 1.09

    st.write("")
    c_proj1, c_proj2 = st.columns([1, 2])
    with c_proj1:
        st.markdown(card("🤖 Yapay Zeka Talep Tahmini", f"{auto_dem:.1f} kg", "Menü, takvim ve hava şartlarına göre"), unsafe_allow_html=True)
    with c_proj2:
        st.info(f"💡 **Newsvendor Üretim Tavsiyesi:** Eksik yemek çıkarma maliyeti (75 ₺/kg) fazla yemek maliyetinden (35 ₺/kg) daha ağır kabul edildiğinden; öğrenci mağduriyetini önlemek amacıyla **%{75/(75+35)*100:.1f}** kritik hizmet hedefiyle **{rec_purchase:.1f} kg** pişirilmesi önerilir.")

    st.write("")
    st.markdown("#### 🍳 2. Aşama: Mutfak Üretim Kararı & İsraf Analizi")
    st.caption("Yapay zekanın talep tahmini ve pişirme tavsiyesi varsayılan olarak gelir; dilediğiniz gibi güncelleyebilirsiniz.")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        expected_demand = st.number_input("Beklenen Tüketim / Talep (kg)", min_value=1.0, value=float(round(auto_dem, 1)), step=1.0,
                                          help="Yapay zekanın tahmin ettiği talep otomatik gelir; aşçı isterse değiştirebilir.")
    with col_q2:
        purchase_qty = st.number_input("Planlanan Pişirme / Üretim Miktarı (kg)", min_value=1.0, value=float(round(rec_purchase, 1)), step=1.0,
                                       help="Tavsiye edilen pişirme miktarı otomatik gelir; mutfakta çıkarılacak net miktarı onaylayın.")

    planned_surplus = purchase_qty - expected_demand
    order_cover = purchase_qty / max(expected_demand, 0.001)

    btn_predict = st.button("🚀 Karar Destek & İsraf Analizini Çalıştır", type="primary")

    if btn_predict:
        if not models['loaded']:
            st.error("Model dosyaları yüklenemedi!")
        else:
            raw_df = pd.DataFrame([{
                'Business_Type': BUSINESS_TYPE, 'Food_Category': sel_cat,
                'Campus_Event': event_val, 'Weather': sel_weather,
                'Purchase_Quantity': float(purchase_qty), 'Daily_Demand': float(expected_demand),
                'Planned_Surplus': float(planned_surplus), 'Order_Cover': float(order_cover),
                'DayOfWeek': int(day_of_week), 'Month': int(month),
                'IsWeekend': int(is_weekend), 'Is_Friday': int(is_friday)
            }])
            input_df = to_model_df(raw_df)

            try:
                pred_waste = max(0.0, float(models['regressor'].predict(input_df)[0]))
                p50_waste = max(0.0, float(models['quantiles'][0.5].predict(input_df)[0]))
                p90_waste = max(0.0, float(models['quantiles'][0.9].predict(input_df)[0]))
                proba = models['classifier'].predict_proba(input_df)[0]
                classes = models['classifier'].classes_
                pred_reason = classes[np.argmax(proba)]

                # Diğer modellerin israf tahmini
                multi = {'HistGradientBoosting (üretim)': pred_waste}
                for name in ['RandomForest', 'XGBoost', 'CatBoost']:
                    if name in models['extra']:
                        try:
                            multi[name] = predict_extra(name, raw_df)
                        except Exception:
                            pass

                # Yerel Karşı-Olgusal XAI
                def _pred(**chg):
                    d = raw_df.copy()
                    for k, v in chg.items():
                        d[k] = v
                    return float(models['regressor'].predict(to_model_df(d))[0])

                xai = {}
                cf = {
                    'Planlanan üretimin talebi aşan kısmı': dict(Purchase_Quantity=float(expected_demand),
                                                                 Planned_Surplus=0.0, Order_Cover=1.0),
                    'Cuma günü': dict(Is_Friday=0),
                    'Kampüs etkinliği': dict(Campus_Event='No'),
                    'Hava durumu': dict(Weather='Cloudy'),
                }
                if planned_surplus <= 0:
                    cf.pop('Planlanan üretimin talebi aşan kısmı')
                if not is_friday:
                    cf.pop('Cuma günü')
                if event_val == 'No':
                    cf.pop('Kampüs etkinliği')
                if sel_weather == 'Cloudy':
                    cf.pop('Hava durumu')
                for name, chg in cf.items():
                    xai[name] = pred_waste - _pred(**chg)

                st.session_state['last_prediction'] = {
                    'waste': pred_waste, 'p50': p50_waste, 'p90': p90_waste,
                    'reason': pred_reason, 'proba': dict(zip(classes, proba)),
                    'classes': classes, 'proba_arr': proba,
                    'demand': expected_demand, 'purchase': purchase_qty,
                    'cat': sel_cat, 'xai': xai, 'multi': multi,
                    'auto_demand': auto_dem
                }
            except Exception as ex:
                st.error(f"Tahmin sırasında hata oluştu: {ex}")

    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']
        st.markdown("---")
        st.markdown("#### 📊 İsraf Projeksiyonu ve Karar Destek Sonuçları")

        c1, c2, c3, c4 = st.columns(4)
        ratio = (res['waste'] / res['purchase'] * 100) if res['purchase'] > 0 else 0
        c1.markdown(card("Tahmini İsraf", f"{res['waste']:.1f} kg",
                         f"Üretimin %{ratio:.1f}'i"), unsafe_allow_html=True)
        c2.markdown(card("P50 Medyan İsraf", f"{res['p50']:.1f} kg",
                         "%50 ihtimalle aşılmaz (taahhüt)"), unsafe_allow_html=True)
        c3.markdown(card("P90 Emniyet Sınırı", f"{res['p90']:.1f} kg",
                         "Kurtarma kapasitesi tavanı"), unsafe_allow_html=True)
        prob_pct = res['proba'].get(res['reason'], 0) * 100
        c4.markdown(card("Ana İsraf Nedeni", REASON_TR.get(res['reason'], res['reason']),
                         f"Olasılık %{prob_pct:.1f}", color=ACCENT), unsafe_allow_html=True)

        st.write("")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            proba_df = pd.DataFrame({
                "Sebep": [REASON_TR.get(c, c) for c in res['classes']],
                "Olasılık": res['proba_arr'] * 100
            }).sort_values("Olasılık", ascending=True)
            fig_p = px.bar(proba_df, x="Olasılık", y="Sebep", orientation="h",
                           title="İsraf nedeni olasılık dağılımı (%)", color_discrete_sequence=[ACCENT])
            fig_p.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_p, width='stretch')
        with col_c2:
            fig_b = go.Figure(go.Bar(
                x=["Yapay Zeka Talebi", "Planlanan Üretim", "Tahmini İsraf", "P90 Risk"],
                y=[res['demand'], res['purchase'], res['waste'], res['p90']],
                marker_color=["#4b5563", "#2563eb", ACCENT, NEG]
            ))
            fig_b.update_layout(title="Üretim, Talep ve İsraf Dengesi (kg)", **PLOT_LAYOUT)
            st.plotly_chart(fig_b, width='stretch')

        if 'xai' in res:
            st.markdown("#### 🔍 Tahminin Nedenleri (Açıklanabilir Yapay Zeka - XAI)")
            st.caption("Her faktör nötr referans değere çekilerek modelin çıktısındaki değişim (kg) ölçülmüştür.")
            xai_df = pd.DataFrame({"Faktör": list(res['xai'].keys()),
                                   "Etki": list(res['xai'].values())}).sort_values("Etki")
            if xai_df.empty:
                st.caption("Girdi referans koşullarda (bulutlu, standart gün, üretim ≤ talep).")
            else:
                fig_x = go.Figure(go.Bar(
                    x=xai_df['Etki'], y=xai_df['Faktör'], orientation='h',
                    marker_color=[ACCENT if v < 0 else NEG for v in xai_df['Etki']]
                ))
                fig_x.update_layout(
                    title="Faktörün tahmine net katkısı, kg (kırmızı: israfı artırır · yeşil: azaltır)",
                    **PLOT_LAYOUT)
                st.plotly_chart(fig_x, width='stretch')
                pos = xai_df[xai_df['Etki'] > 0]
                if not pos.empty:
                    st.info(f"💡 **XAI Tavsiyesi:** İsraf riskini en çok artıran faktör: **{pos.iloc[-1]['Faktör']}**. Üretim miktarını talebe daha yakın tutmak israfı önler.")

        st.success(
            f"🎯 **Yemekhane Yönetim Özeti:** Yapay zeka tahmini talep: **{res['demand']:.1f} kg**, planlanan üretim: **{res['purchase']:.1f} kg**. "
            f"Olası israf **{res['waste']:.1f} kg** öngörülmektedir. İnsan tüketimi ve askıda yemek paydaşlarına ön taahhüt olarak **{res['p50']:.1f} kg**, "
            f"maksimum rezerv tavanı olarak ise **{res['p90']:.1f} kg** kapasite ayrılması tavsiye edilir.")

# ==============================================================================
# SEKME 2: MALİYET ANALİZİ
# ==============================================================================
with tab2:
    st.markdown("#### Fazla / eksik üretim maliyet analizi (Newsvendor)")
    st.caption("Fazla üretim maliyeti (Cₒ): artan yemeğin hammadde ve bertaraf bedeli. "
               "Eksik üretim maliyeti (Cᵤ): yemek erken bitince oluşan memnuniyetsizlik ve acil ikame maliyeti.")

    in1, in2 = st.columns(2)
    c_over = in1.number_input("Fazla üretim maliyeti (₺/kg)", min_value=1.0, value=35.0, step=1.0)
    c_under = in2.number_input("Eksik üretim maliyeti (₺/kg)", min_value=1.0, value=75.0, step=1.0)

    critical_ratio = c_under / (c_under + c_over)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cₒ", f"{c_over:.2f} ₺/kg")
    m2.metric("Cᵤ", f"{c_under:.2f} ₺/kg")
    m3.metric("Kritik oran  Cᵤ / (Cᵤ + Cₒ)", f"%{critical_ratio*100:.1f}")

    mean_dem = float(expected_demand)
    std_dem = mean_dem * 0.12
    quantities = np.linspace(mean_dem * 0.7, mean_dem * 1.4, 40)
    rng = np.random.default_rng(42)
    sample_demands = np.clip(rng.normal(mean_dem, std_dem, 1000), 0, None)

    expected_costs, over_costs, under_costs = [], [], []
    for q in quantities:
        o_loss = np.maximum(0, q - sample_demands) * c_over
        u_loss = np.maximum(0, sample_demands - q) * c_under
        expected_costs.append((o_loss + u_loss).mean())
        over_costs.append(o_loss.mean())
        under_costs.append(u_loss.mean())

    opt_q = quantities[int(np.argmin(expected_costs))]
    cost_df = pd.DataFrame({
        "Üretim miktarı (kg)": quantities,
        "Toplam beklenen maliyet": expected_costs,
        "Fazla üretim kaybı": over_costs,
        "Eksik üretim kaybı": under_costs
    })
    fig_cost = px.line(
        cost_df, x="Üretim miktarı (kg)",
        y=["Toplam beklenen maliyet", "Fazla üretim kaybı", "Eksik üretim kaybı"],
        title=f"Maliyet eğrisi — optimum üretim: {opt_q:.1f} kg",
        color_discrete_map={"Toplam beklenen maliyet": "#111827",
                            "Fazla üretim kaybı": "#9ca3af",
                            "Eksik üretim kaybı": NEG}
    )
    fig_cost.add_vline(x=opt_q, line_dash="dash", line_color=ACCENT,
                       annotation_text=f"Optimum {opt_q:.1f} kg")
    fig_cost.add_vline(x=mean_dem, line_dash="dot", line_color="#9ca3af",
                       annotation_text=f"Ort. talep {mean_dem:.0f} kg", annotation_position="bottom")
    fig_cost.update_layout(**{**PLOT_LAYOUT, "height": 380}, legend_title_text="")
    st.plotly_chart(fig_cost, width='stretch')

    if c_under > c_over:
        st.info(f"Eksik üretim maliyeti ({c_under:.0f} ₺) fazla üretimden ({c_over:.0f} ₺) yüksek olduğu için, "
                f"ortalama talep ({mean_dem:.0f} kg) yerine emniyet payıyla **{opt_q:.1f} kg** üretmek toplam beklenen zararı en aza indirir.")
    elif c_over > c_under:
        st.info(f"Fazla üretim maliyeti ({c_over:.0f} ₺) daha yüksek olduğu için temkinli davranıp "
                f"**{opt_q:.1f} kg** üretmek toplam beklenen zararı en aza indirir.")
    else:
        st.info(f"Maliyetler eşit; ortalama talep düzeyinde (**{opt_q:.1f} kg**) üretim dengeyi sağlar.")

# ==============================================================================
# SEKME 3: İKİNCİ HAYAT
# ==============================================================================
with tab3:
    st.markdown("#### Gıda ikinci hayat kataloğu (EPA gıda kurtarma hiyerarşisi)")
    st.caption(f"Seçili kategori: {CATEGORY_TR.get(sel_cat, sel_cat)}")
    ideas = [i for i in RESCUE_CATALOG if sel_cat in i["categories"]]
    if ideas:
        for item in ideas:
            with st.expander(f"{item['tier']} — {item['action']}", expanded=False):
                st.write(item['detail'])
                st.markdown(f"- **Asgari miktar:** {item['min_qty']} kg\n- **Beklenen fayda:** {item['benefit']}")
    else:
        st.write("Bu kategori için tanımlı kurtarma senaryosu bulunamadı.")

# ==============================================================================
# SEKME 4: VERİ ANALİTİĞİ & MODEL BAŞARIMI
# ==============================================================================
with tab4:
    st.markdown("#### Veri analitiği ve üretim modeli başarımı")
    if dataset_df is not None:
        e1, e2, e3 = st.columns(3)
        e1.metric("Toplam kayıt", f"{len(dataset_df):,}")
        e2.metric("Sıfır israf oranı", f"%{(dataset_df['Waste_Quantity'] == 0).mean()*100:.1f}")
        e3.metric("Ortalama günlük israf", f"{dataset_df['Waste_Quantity'].mean():.2f} kg")

        g1, g2 = st.columns(2)
        with g1:
            f1 = px.histogram(dataset_df, x="Waste_Quantity", nbins=30,
                              title="İsraf miktarı dağılımı (kg)", color_discrete_sequence=[ACCENT])
            f1.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(f1, width='stretch')
        with g2:
            rc = dataset_df["Waste_Reason"].value_counts().reset_index()
            rc.columns = ["İsraf nedeni", "Kayıt"]
            rc["İsraf nedeni"] = rc["İsraf nedeni"].map(lambda x: REASON_TR.get(x, x))
            f2 = px.bar(rc.sort_values("Kayıt"), x="Kayıt", y="İsraf nedeni", orientation="h",
                        title="İsraf nedenleri", color_discrete_sequence=[ACCENT])
            f2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(f2, width='stretch')

        st.markdown("##### Baseline ile karşılaştırma (zaman bazlı test)")
        st.dataframe(pd.DataFrame({
            "Model / yaklaşım": ["Sabit medyan", "Naif fazlalık kuralı", "HistGradientBoosting (üretim)"],
            "MAE (kg)": [3.02, 1.11, 1.12],
            "R²": [-0.07, 0.72, 0.85],
            "P90 kapsama": ["-", "-", "%83.5"]
        }), hide_index=True, width='stretch')

        st.markdown("##### Genel öznitelik önemi (permütasyon, test dönemi)")
        gi = compute_global_importance(get_models_mtime())
        if gi:
            fi = pd.DataFrame({"Faktör": list(gi.keys()), "Önem (%)": [round(v, 1) for v in gi.values()]}).sort_values("Önem (%)")
            f3 = px.bar(fi, x="Önem (%)", y="Faktör", orientation="h", text="Önem (%)",
                        color_discrete_sequence=[ACCENT])
            f3.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(f3, width='stretch')
            st.caption("Bir faktörün değerleri karıştırıldığında MAE'nin ne kadar bozulduğuna göre hesaplanır; toplam %100'e normalize edilmiştir.")
    else:
        st.warning("Veri seti bulunamadı.")

# ==============================================================================
# SEKME 5: MODEL KARŞILAŞTIRMASI
# ==============================================================================
with tab5:
    st.markdown("#### 🏆 Model Karşılaştırması: Çoklu Algoritma Başarımı")
    st.caption("Hem Günlük Talep Tahmini hem de Gıda İsrafı Tahmini için 4 farklı topluluk (ensemble) algoritması aynı veri ve zaman bazlı test protokolüyle eğitilmiştir.")

    tab_m1, tab_m2 = st.tabs(["📈 1. Aşama: Günlük Talep Tahmin Modelleri", "⚖️ 2. Aşama: Gıda İsrafı Tahmin Modelleri"])

    with tab_m1:
        st.markdown("##### Günlük Talep Tahmin Modelleri Performansı (Test Kümesi: Ekim–Aralık)")
        dem_comp = load_demand_comparison()
        if dem_comp is not None:
            st.dataframe(dem_comp, hide_index=True, width='stretch')
            st.info("💡 **Talep Modelleri Yorumu:** Geleneksel sabit medyan kuralı (12.04 kg MAE) yerine makine öğrenmesi modelleri kullanıldığında hata payı **7.82 kg'a kadar düşürülmüş (%35 belirgin iyileşme)** elde edilmiştir. Random Forest ve CatBoost modelleri en yüksek R² (0.58) açıklayıcılık oranına ulaşmıştır.")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                fig_dm = px.bar(dem_comp, x="Model", y="MAE (kg)", text="MAE (kg)",
                                title="Talep Tahmini Ortalama Hata (MAE - düşük iyi)", color_discrete_sequence=[ACCENT])
                fig_dm.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig_dm, width='stretch')
            with d_col2:
                fig_dr = px.bar(dem_comp[dem_comp['R² Skoru'] > 0], x="Model", y="R² Skoru", text="R² Skoru",
                                title="Talep Açıklanan Varyans (R² - yüksek iyi)", color_discrete_sequence=["#4b5563"])
                fig_dr.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig_dr, width='stretch')
        else:
            st.warning("talep_model_karsilastirma.csv bulunamadı.")

    with tab_m2:
        st.markdown("##### Gıda İsrafı Tahmin Modelleri Performansı (Test Kümesi: Ekim–Aralık)")
        comp = load_comparison()
        if comp is None:
            st.warning("model_karsilastirma.csv bulunamadı. `python scripts/model_egit_karsilastir.py` çalıştırın.")
        else:
            st.dataframe(comp, hide_index=True, width='stretch')

            best_mae = comp.loc[comp['MAE (kg)'].idxmin(), 'Model']
            best_r2 = comp.loc[comp['R² Skoru'].idxmax(), 'Model']
            b1, b2, b3 = st.columns(3)
            b1.markdown(card("En Düşük İsraf MAE", best_mae, f"{comp['MAE (kg)'].min():.3f} kg"), unsafe_allow_html=True)
            b2.markdown(card("En Yüksek R²", best_r2, f"{comp['R² Skoru'].max():.3f}"), unsafe_allow_html=True)
            fastest = comp.loc[comp['Eğitim Süresi (s)'].idxmin(), 'Model']
            b3.markdown(card("En Hızlı Eğitim", fastest, f"{comp['Eğitim Süresi (s)'].min():.2f} sn"), unsafe_allow_html=True)

            h1, h2 = st.columns(2)
            with h1:
                fm = px.bar(comp, x="Model", y="MAE (kg)", text="MAE (kg)",
                            title="İsraf Tahmini Ortalama Mutlak Hata (MAE)", color_discrete_sequence=[ACCENT])
                fm.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fm, width='stretch')
            with h2:
                fr = px.bar(comp[comp['R² Skoru'] > 0], x="Model", y="R² Skoru", text="R² Skoru",
                            title="İsraf Tahmini R² Başarımı", color_discrete_sequence=["#4b5563"])
                fr.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fr, width='stretch')

            st.info(
                "💡 **İsraf Modelleri Yorumu:** Dört model de baseline kuralına (3.02 kg MAE) kıyasla %60'ın üzerinde yüksek başarı göstermiştir. "
                "HistGradientBoosting; kategorik verileri doğrudan işlemesi, milisaniye düzeyinde yanıt vermesi ve "
                "P50/P90 kuantil modellerini doğrudan desteklemesi sebebiyle üretim ortamında tercih edilmiştir.")

    if 'last_prediction' in st.session_state and 'multi' in st.session_state['last_prediction']:
        st.markdown("##### 🔍 Son Girişiniz İçin Farklı Algoritmaların İsraf Tahminleri")
        mp = st.session_state['last_prediction']['multi']
        mp_df = pd.DataFrame({"Model": list(mp.keys()), "Tahmini İsraf (kg)": [round(v, 2) for v in mp.values()]})
        fp = px.bar(mp_df, x="Model", y="Tahmini İsraf (kg)", text="Tahmini İsraf (kg)",
                    color_discrete_sequence=[ACCENT])
        fp.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fp, width='stretch')

    if st.button("🔄 Modelleri ve Önbelleği Yeniden Yükle"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
