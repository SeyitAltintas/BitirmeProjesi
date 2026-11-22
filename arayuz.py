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
MODEL_FILES = ['food_waste_regressor_model.pkl', 'food_waste_classifier_model.pkl',
               'quantile_models.pkl', 'rf_model.pkl', 'xgb_model.pkl', 'catboost_model.pkl']

def get_models_mtime():
    return tuple(os.path.getmtime(os.path.join(MODEL_DIR, f))
                 for f in MODEL_FILES if os.path.exists(os.path.join(MODEL_DIR, f)))

@st.cache_resource
def load_models(_mtime_hash):
    models = {'loaded': False, 'extra': {}}
    try:
        reg_path = os.path.join(MODEL_DIR, 'food_waste_regressor_model.pkl')
        clf_path = os.path.join(MODEL_DIR, 'food_waste_classifier_model.pkl')
        q_path = os.path.join(MODEL_DIR, 'quantile_models.pkl')
        if os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(q_path):
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
    st.markdown("#### Günlük giriş")
    st.caption("Aşçı yalnızca temel mutfak verilerini girer; Cuma etkisi tarihten otomatik hesaplanır.")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        sel_date = st.date_input("Planlama tarihi", datetime.date.today())
        sel_cat_tr = st.selectbox("Menü kategorisi", list(CATEGORY_TR.values()), index=0)
        sel_cat = TR_TO_CAT[sel_cat_tr]
    with col_in2:
        sel_weather_tr = st.selectbox("Hava durumu", list(WEATHER_TR.values()), index=0)
        sel_weather = TR_TO_WEATHER[sel_weather_tr]
        sel_event = st.selectbox("Kampüs etkinliği / özel dönem",
                                 ["Normal gün", "Etkinlik / sınav haftası / şenlik"], index=0)
        event_val = "Yes" if sel_event.startswith("Etkinlik") else "No"
    with col_in3:
        expected_demand = st.number_input("Beklenen talep (kg)", min_value=1.0, value=60.0, step=5.0,
                                          help="Yemekhaneye gelmesi beklenen tahmini tüketim.")
        purchase_qty = st.number_input("Planlanan üretim (kg)", min_value=1.0, value=68.0, step=5.0,
                                       help="Pişirilmesi planlanan miktar.")

    # Otomatik türetilen değişkenler
    day_of_month = sel_date.day
    day_of_week = sel_date.weekday()
    month = sel_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_friday = 1 if day_of_week == 4 else 0

    friday_txt = "Cuma (erken ayrılış eğilimi)" if is_friday else "Standart hafta içi"

    st.markdown(
        f'<span class="chip">Akademik yük: {friday_txt}</span>'
        f'<span class="chip">Otomatik hesaplandı</span>',
        unsafe_allow_html=True)

    planned_surplus = purchase_qty - expected_demand
    order_cover = purchase_qty / max(expected_demand, 0.001)

    btn_predict = st.button("Tahmini çalıştır", type="primary")

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

                # Diğer modellerin tahmini (Model Karşılaştırması sekmesi için)
                multi = {'HistGradientBoosting (üretim)': pred_waste}
                for name in ['RandomForest', 'XGBoost', 'CatBoost']:
                    if name in models['extra']:
                        try:
                            multi[name] = predict_extra(name, raw_df)
                        except Exception:
                            pass

                # Yerel XAI: her faktör "nötr" değere çekilip modelin tahmini yeniden alınır;
                # fark (kg) o faktörün tahmine gerçek katkısıdır.
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
                    'cat': sel_cat, 'xai': xai, 'multi': multi
                }
            except Exception as ex:
                st.error(f"Tahmin sırasında hata oluştu: {ex}")

    if 'last_prediction' in st.session_state:
        res = st.session_state['last_prediction']
        st.markdown("---")
        st.markdown("#### Sonuçlar")

        c1, c2, c3, c4 = st.columns(4)
        ratio = (res['waste'] / res['purchase'] * 100) if res['purchase'] > 0 else 0
        c1.markdown(card("Tahmini israf", f"{res['waste']:.1f} kg",
                         f"Üretimin %{ratio:.1f}'i"), unsafe_allow_html=True)
        c2.markdown(card("P50 (medyan)", f"{res['p50']:.1f} kg",
                         "%50 ihtimalle aşılmaz"), unsafe_allow_html=True)
        c3.markdown(card("P90 (emniyet sınırı)", f"{res['p90']:.1f} kg",
                         "Kurtarma kapasitesi tavanı"), unsafe_allow_html=True)
        prob_pct = res['proba'].get(res['reason'], 0) * 100
        c4.markdown(card("Ana israf nedeni", REASON_TR.get(res['reason'], res['reason']),
                         f"Olasılık %{prob_pct:.1f}", color=ACCENT), unsafe_allow_html=True)

        st.write("")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            proba_df = pd.DataFrame({
                "Sebep": [REASON_TR.get(c, c) for c in res['classes']],
                "Olasılık": res['proba_arr'] * 100
            }).sort_values("Olasılık", ascending=True)
            fig_p = px.bar(proba_df, x="Olasılık", y="Sebep", orientation="h",
                           title="İsraf nedeni olasılıkları (%)", color_discrete_sequence=[ACCENT])
            fig_p.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_p, width='stretch')
        with col_c2:
            fig_b = go.Figure(go.Bar(
                x=["Beklenen talep", "Tahmin", "P50", "P90"],
                y=[res['demand'], res['waste'], res['p50'], res['p90']],
                marker_color=["#9ca3af", ACCENT, ACCENT, ACCENT]
            ))
            fig_b.update_layout(title="Talep ve israf projeksiyonu (kg)", **PLOT_LAYOUT)
            st.plotly_chart(fig_b, width='stretch')

        if 'xai' in res:
            st.markdown("#### Tahminin nedenleri (XAI)")
            st.caption("Her faktör nötr değere çekilip model yeniden çalıştırılır; tahmindeki değişim (kg) o faktörün katkısıdır.")
            xai_df = pd.DataFrame({"Faktör": list(res['xai'].keys()),
                                   "Etki": list(res['xai'].values())}).sort_values("Etki")
            fig_x = go.Figure(go.Bar(
                x=xai_df['Etki'], y=xai_df['Faktör'], orientation='h',
                marker_color=[ACCENT if v < 0 else NEG for v in xai_df['Etki']]
            ))
            fig_x.update_layout(
                title="Faktörün tahmine katkısı, kg (kırmızı: israfı artırır · yeşil: azaltır)",
                **PLOT_LAYOUT)
            st.plotly_chart(fig_x, width='stretch')
            if xai_df.empty:
                st.caption("Girdi referans (nötr) koşullarda: bulut/dengeli dönem/Cuma değil/etkinlik yok ve üretim ≤ talep.")
            pos = xai_df[xai_df['Etki'] > 0]
            if not pos.empty:
                st.info(f"İsraf riskini en çok yükselten faktör: **{pos.iloc[-1]['Faktör']}**. "
                        f"Üretimi talebe yaklaştırmak israfı azaltır.")

        st.success(
            f"Beklenen talep **{res['demand']:.0f} kg**, planlanan üretim **{res['purchase']:.0f} kg**. "
            f"Model **{res['waste']:.1f} kg** israf öngörüyor. Ön taahhüt için **{res['p50']:.1f} kg**, "
            f"maksimum rezerv için **{res['p90']:.1f} kg** kapasite ayırın.")

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
    st.markdown("#### Model karşılaştırması: HistGradientBoosting · RandomForest · XGBoost · CatBoost")
    st.caption("Aynı veri, aynı zaman bazlı bölme (Ocak–Eylül eğitim, Ekim–Aralık test) ve aynı 13 öznitelik ile eğitildi.")

    comp = load_comparison()
    if comp is None:
        st.warning("model_karsilastirma.csv bulunamadı. `python scripts/model_egit_karsilastir.py` komutunu çalıştırın.")
    else:
        comp_show = comp.drop(columns=['P90 Kapsama (%)'], errors='ignore')
        best_mae = comp_show.loc[comp_show['MAE (kg)'].idxmin(), 'Model']
        best_r2 = comp_show.loc[comp_show['R² Skoru'].idxmax(), 'Model']
        b1, b2, b3 = st.columns(3)
        b1.markdown(card("En düşük MAE", best_mae,
                         f"{comp_show['MAE (kg)'].min():.3f} kg"), unsafe_allow_html=True)
        b2.markdown(card("En yüksek R²", best_r2,
                         f"{comp_show['R² Skoru'].max():.3f}"), unsafe_allow_html=True)
        fastest = comp_show.loc[comp_show['Eğitim Süresi (s)'].idxmin(), 'Model']
        b3.markdown(card("En hızlı eğitim", fastest,
                         f"{comp_show['Eğitim Süresi (s)'].min():.1f} sn"), unsafe_allow_html=True)

        st.write("")
        st.dataframe(comp_show, hide_index=True, width='stretch')

        h1, h2 = st.columns(2)
        with h1:
            fm = px.bar(comp_show, x="Model", y="MAE (kg)", text="MAE (kg)",
                        title="Ortalama mutlak hata (düşük iyi)", color_discrete_sequence=[ACCENT])
            fm.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fm, width='stretch')
        with h2:
            fr = px.bar(comp_show, x="Model", y="R² Skoru", text="R² Skoru",
                        title="R² skoru (yüksek iyi)", color_discrete_sequence=["#6b7280"])
            fr.update_layout(**PLOT_LAYOUT)
            fr.update_yaxes(range=[min(0.7, comp_show['R² Skoru'].min() - 0.05), 1])
            st.plotly_chart(fr, width='stretch')

        h3, h4 = st.columns(2)
        with h3:
            fa = px.bar(comp_show, x="Model", y="Sınıflandırıcı Doğruluğu (%)",
                        text="Sınıflandırıcı Doğruluğu (%)", title="İsraf nedeni doğruluğu (%)",
                        color_discrete_sequence=[ACCENT])
            fa.update_layout(**PLOT_LAYOUT)
            fa.update_yaxes(range=[70, 90])
            st.plotly_chart(fa, width='stretch')
        with h4:
            ft = px.bar(comp_show, x="Model", y="Eğitim Süresi (s)", text="Eğitim Süresi (s)",
                        title="Eğitim süresi (sn)", color_discrete_sequence=["#6b7280"])
            ft.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(ft, width='stretch')

        st.info(
            "**Yorum:** Dört model birbirine çok yakın sonuç verdi (MAE farkı ≈ 0.17 kg). Bu, hata payının "
            "model seçiminden çok veri setindeki doğal gürültüden kaynaklandığını gösterir. "
            "Üretim modeli olarak HistGradientBoosting; kategorik değişkenleri doğal işlemesi, ek bağımlılık "
            "gerektirmemesi ve P50/P90 kantil tahminlerini aynı çatı altında vermesi nedeniyle korunmuştur. "
            "Tek test dönemiyle yapılan bu fark istatistiksel olarak kesin değildir; çapraz doğrulama ile "
            "desteklenmesi önerilir.")

    if 'last_prediction' in st.session_state and 'multi' in st.session_state['last_prediction']:
        st.markdown("##### Son girişiniz için model tahminleri")
        mp = st.session_state['last_prediction']['multi']
        mp_df = pd.DataFrame({"Model": list(mp.keys()), "Tahmini israf (kg)": [round(v, 2) for v in mp.values()]})
        fp = px.bar(mp_df, x="Model", y="Tahmini israf (kg)", text="Tahmini israf (kg)",
                    color_discrete_sequence=[ACCENT])
        fp.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fp, width='stretch')
    else:
        st.caption("Aynı girdinin dört modelde nasıl tahmin edildiğini görmek için önce 'Günlük Tahmin' sekmesinde tahmini çalıştırın.")

    if st.button("Modelleri yeniden yükle"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
