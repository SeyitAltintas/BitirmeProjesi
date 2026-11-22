"""
Yemekhane Talep & İsraf Karar Destek Sistemi - Tasarım ve Tema Modülü
=====================================================================
Streamlit arayüzü için CSS stilleri, renk paleti, Plotly grafik temaları 
ve HTML kart/başlık bileşenlerini içerir.
"""

import streamlit as st

# ==============================================================================
# 1. RENK PALETİ (SADE & MİNİMAL TEMA)
# ==============================================================================
ACCENT = "#2f6f5e"      # Ana vurgu rengi (Zümrüt yeşili)
MUTED = "#6b7280"       # İkincil / pasif metin rengi (Gri)
LINE = "#e5e7eb"        # Kenarlık ve ayırıcı çizgi rengi (Açık gri)
NEG = "#b4443c"         # Uyarı / olumsuz etki rengi (Kiremit kırmızısı)

# Grafik renkleri
COLOR_PRIMARY = "#2563eb"
COLOR_DARK = "#111827"
COLOR_GRAY = "#4b5563"
COLOR_LIGHT_GRAY = "#9ca3af"

# ==============================================================================
# 2. GRAFİK TEMA DÜZENİ (PLOTLY)
# ==============================================================================
PLOT_LAYOUT = dict(
    template="simple_white",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(size=13),
    height=300
)

# ==============================================================================
# 3. ÖZEL CSS STİLLERİ
# ==============================================================================
CUSTOM_CSS = f"""
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
"""


def apply_custom_css():
    """Uygulamaya özel CSS stil tanımlarını enjekte eder."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 4. GÖRSEL BİLEŞENLER (HTML YARDIMCILARI)
# ==============================================================================
def card(label: str, value: str, note: str = "", color: str = None) -> str:
    """Özet istatistik veya metrik gösterimi için HTML kart oluşturur."""
    style = f"color:{color};" if color else ""
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val" style="{style}">{value}</div>'
            f'<div class="note">{note}</div></div>')


def chip(text: str) -> str:
    """Etiket (chip / rozet) stili HTML elemanı oluşturur."""
    return f'<span class="chip">{text}</span>'


def render_header(
    title: str = "Üniversite Yemekhanesi Talep & Gıda İsrafı Karar Destek Sistemi",
    subtitle: str = "Günlük israf tahmini · P50/P90 belirsizlik analizi · Açıklanabilir yapay zeka · Model karşılaştırması",
    badge: str = "Proje No: 13 · TÜBİTAK 2209-A"
):
    """Ana sayfa başlık ve alt bilgi bandını çizer."""
    st.markdown(f"""
    <p class="app-title">{title}</p>
    <p class="app-sub">{subtitle} &nbsp;|&nbsp; {badge}</p>
    """, unsafe_allow_html=True)
