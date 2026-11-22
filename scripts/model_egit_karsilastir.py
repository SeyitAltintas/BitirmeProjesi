"""
Model Eğitim ve Karşılaştırma Scripti
======================================
1. Talep Tahmini Modelleri (Daily_Demand): HGB, RF, XGBoost, CatBoost
2. İsraf Tahmini Modelleri (Waste_Quantity): HGB, RF, XGBoost, CatBoost

Her iki görev için modelleri eğitir, test döneminde (Ekim-Aralık) karşılaştırır ve
models/comparison/ klasörüne kaydeder.

Çalıştırma:
  python scripts/model_egit_karsilastir.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
COMP_DIR = os.path.join(MODEL_DIR, 'comparison')
os.makedirs(COMP_DIR, exist_ok=True)

# ============================================================
# 1. VERİ YÜKLEME
# ============================================================
data_path = os.path.join(DATA_DIR, 'yemekhane_oznitelikler.csv')
df = pd.read_csv(data_path, parse_dates=['Date'])
df['Weather'] = df['Weather'].fillna('Cloudy')
print(f"[OK] Veri yüklendi: {len(df)} satır, {df.shape[1]} sütun")

# Kategori listeleri
CAT_CATEGORIES = {
    'Business_Type': ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket'],
    'Food_Category': ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables'],
    'Campus_Event' : ['No', 'Yes'],
    'Weather'      : ['Cloudy', 'Hot', 'Rainy', 'Sunny'],
}
CAT_COLS = list(CAT_CATEGORIES.keys())

DEMAND_FEATURES = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]

WASTE_FEATURES = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'Purchase_Quantity', 'Daily_Demand', 'Planned_Surplus', 'Order_Cover',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]

# Modeller
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return round(mae, 3), round(rmse, 3), round(r2, 3)

def make_hgb_df(df_raw, features):
    out = df_raw[features].copy()
    for col, cats in CAT_CATEGORIES.items():
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=cats)
    return out

# ============================================================
# GÖREV 1: TALEP TAHMİNİ KARŞILAŞTIRMASI (Daily_Demand kg)
# ============================================================
print("\n" + "=" * 60)
print("1. GÖREV: TALEP TAHMİNİ KARŞILAŞTIRMASI (Daily_Demand kg)")
print("=" * 60)

demand_df = df.dropna(subset=['Daily_Demand']).copy()
train_dem = demand_df[demand_df['Date'].dt.month <= 9].copy()
test_dem  = demand_df[demand_df['Date'].dt.month >= 10].copy()

X_tr_dem_raw = train_dem[DEMAND_FEATURES].copy()
X_te_dem_raw = test_dem[DEMAND_FEATURES].copy()
y_tr_dem = train_dem['Daily_Demand'].values
y_te_dem = test_dem['Daily_Demand'].values

# Encoders
enc_dem = OrdinalEncoder(categories=[CAT_CATEGORIES[c] for c in CAT_COLS], handle_unknown='use_encoded_value', unknown_value=-1)
enc_dem.fit(X_tr_dem_raw[CAT_COLS])

X_tr_dem_enc = X_tr_dem_raw.copy()
X_te_dem_enc = X_te_dem_raw.copy()
X_tr_dem_enc[CAT_COLS] = enc_dem.transform(X_tr_dem_raw[CAT_COLS])
X_te_dem_enc[CAT_COLS] = enc_dem.transform(X_te_dem_raw[CAT_COLS])

X_tr_dem_hgb = make_hgb_df(X_tr_dem_raw, DEMAND_FEATURES)
X_te_dem_hgb = make_hgb_df(X_te_dem_raw, DEMAND_FEATURES)

X_tr_dem_cb = X_tr_dem_raw.copy()
X_te_dem_cb = X_te_dem_raw.copy()
for c in CAT_COLS:
    X_tr_dem_cb[c] = X_tr_dem_cb[c].astype(str)
    X_te_dem_cb[c] = X_te_dem_cb[c].astype(str)

demand_results = []

# Baseline
base_med = np.median(y_tr_dem)
b_mae, b_rmse, b_r2 = regression_metrics(y_te_dem, np.full_like(y_te_dem, base_med))
demand_results.append({
    'Model': 'Sabit Medyan (Baseline)', 'MAE (kg)': b_mae, 'RMSE (kg)': b_rmse,
    'R² Skoru': b_r2, 'Eğitim Süresi (s)': 0.0, 'İyileşme (%)': '0.0%'
})
print(f"Sabit Medyan Baseline: MAE={b_mae} kg")

# 1. HGB (Demand)
t0 = time.time()
try:
    hgb_dem = joblib.load(os.path.join(MODEL_DIR, 'food_demand_model.pkl'))
except Exception:
    hgb_dem = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, categorical_features='from_dtype', random_state=42).fit(X_tr_dem_hgb, y_tr_dem)
t_hgb_dem = time.time() - t0
p_hgb_dem = hgb_dem.predict(X_te_dem_hgb)
mae, rmse, r2 = regression_metrics(y_te_dem, p_hgb_dem)
imp = round((b_mae - mae) / b_mae * 100, 1)
demand_results.append({
    'Model': 'HistGradientBoosting (Üretim)', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
    'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_hgb_dem, 2), 'İyileşme (%)': f"%{imp}"
})
print(f"HGB Demand: MAE={mae} kg, R²={r2}, İyileşme=%{imp}")

# 2. RF (Demand)
t0 = time.time()
rf_dem = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_tr_dem_enc, y_tr_dem)
t_rf_dem = time.time() - t0
p_rf_dem = rf_dem.predict(X_te_dem_enc)
mae, rmse, r2 = regression_metrics(y_te_dem, p_rf_dem)
imp = round((b_mae - mae) / b_mae * 100, 1)
demand_results.append({
    'Model': 'Random Forest', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
    'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_rf_dem, 2), 'İyileşme (%)': f"%{imp}"
})
print(f"RF Demand: MAE={mae} kg, R²={r2}, İyileşme=%{imp}")

# 3. XGB (Demand)
if HAS_XGB:
    t0 = time.time()
    xgb_dem = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=5, subsample=0.8, random_state=42, verbosity=0).fit(X_tr_dem_enc, y_tr_dem)
    t_xgb_dem = time.time() - t0
    p_xgb_dem = xgb_dem.predict(X_te_dem_enc)
    mae, rmse, r2 = regression_metrics(y_te_dem, p_xgb_dem)
    imp = round((b_mae - mae) / b_mae * 100, 1)
    demand_results.append({
        'Model': 'XGBoost', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
        'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_xgb_dem, 2), 'İyileşme (%)': f"%{imp}"
    })
    print(f"XGB Demand: MAE={mae} kg, R²={r2}, İyileşme=%{imp}")

# 4. CatBoost (Demand)
if HAS_CAT:
    t0 = time.time()
    cb_dem = CatBoostRegressor(iterations=300, learning_rate=0.08, depth=5, cat_features=CAT_COLS, random_seed=42, verbose=0).fit(X_tr_dem_cb, y_tr_dem)
    t_cb_dem = time.time() - t0
    p_cb_dem = cb_dem.predict(X_te_dem_cb)
    mae, rmse, r2 = regression_metrics(y_te_dem, p_cb_dem)
    imp = round((b_mae - mae) / b_mae * 100, 1)
    demand_results.append({
        'Model': 'CatBoost', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
        'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_cb_dem, 2), 'İyileşme (%)': f"%{imp}"
    })
    print(f"CatBoost Demand: MAE={mae} kg, R²={r2}, İyileşme=%{imp}")

# Save demand models
joblib.dump({
    'hgb': hgb_dem, 'rf': rf_dem,
    'xgb': xgb_dem if HAS_XGB else None,
    'catboost': cb_dem if HAS_CAT else None,
    'encoder': enc_dem
}, os.path.join(COMP_DIR, 'demand_models_comparison.pkl'))

dem_comp_df = pd.DataFrame(demand_results)
dem_comp_df.to_csv(os.path.join(COMP_DIR, 'talep_model_karsilastirma.csv'), index=False)
print(f"[KAYDEDİLDİ] talep_model_karsilastirma.csv")

# ============================================================
# GÖREV 2: İSRAF TAHMİNİ KARŞILAŞTIRMASI (Waste_Quantity kg)
# ============================================================
print("\n" + "=" * 60)
print("2. GÖREV: İSRAF TAHMİNİ KARŞILAŞTIRMASI (Waste_Quantity kg)")
print("=" * 60)

train_wst = df[df['Date'].dt.month <= 9].copy()
test_wst  = df[df['Date'].dt.month >= 10].copy()

X_tr_wst_raw = train_wst[WASTE_FEATURES].copy()
X_te_wst_raw = test_wst[WASTE_FEATURES].copy()
y_tr_wst = train_wst['Waste_Quantity'].values
y_te_wst = test_wst['Waste_Quantity'].values
c_tr_wst = train_wst['Waste_Reason'].astype(str).to_numpy(dtype=object)
c_te_wst = test_wst['Waste_Reason'].astype(str).to_numpy(dtype=object)

enc_wst = OrdinalEncoder(categories=[CAT_CATEGORIES[c] for c in CAT_COLS], handle_unknown='use_encoded_value', unknown_value=-1)
enc_wst.fit(X_tr_wst_raw[CAT_COLS])

X_tr_wst_enc = X_tr_wst_raw.copy()
X_te_wst_enc = X_te_wst_raw.copy()
X_tr_wst_enc[CAT_COLS] = enc_wst.transform(X_tr_wst_raw[CAT_COLS])
X_te_wst_enc[CAT_COLS] = enc_wst.transform(X_te_wst_raw[CAT_COLS])

X_tr_wst_hgb = make_hgb_df(X_tr_wst_raw, WASTE_FEATURES)
X_te_wst_hgb = make_hgb_df(X_te_wst_raw, WASTE_FEATURES)

X_tr_wst_cb = X_tr_wst_raw.copy()
X_te_wst_cb = X_te_wst_raw.copy()
for c in CAT_COLS:
    X_tr_wst_cb[c] = X_tr_wst_cb[c].astype(str)
    X_te_wst_cb[c] = X_te_wst_cb[c].astype(str)

waste_results = []

# Baseline
base_med_wst = np.median(y_tr_wst)
bw_mae, bw_rmse, bw_r2 = regression_metrics(y_te_wst, np.full_like(y_te_wst, base_med_wst))
waste_results.append({
    'Model': 'Sabit Medyan (Baseline)', 'MAE (kg)': bw_mae, 'RMSE (kg)': bw_rmse,
    'R² Skoru': bw_r2, 'Eğitim Süresi (s)': 0.0, 'Sınıflandırıcı Doğruluğu (%)': '-', 'P90 Kapsama (%)': '-'
})

# 1. HGB (Production Waste Model)
t0 = time.time()
try:
    hgb_reg = joblib.load(os.path.join(MODEL_DIR, 'food_waste_regressor_model.pkl'))
    hgb_clf = joblib.load(os.path.join(MODEL_DIR, 'food_waste_classifier_model.pkl'))
    q_mod = joblib.load(os.path.join(MODEL_DIR, 'quantile_models.pkl'))
    hgb_q90 = q_mod[0.9]
except Exception:
    hgb_reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, categorical_features='from_dtype', random_state=42).fit(X_tr_wst_hgb, y_tr_wst)
    hgb_clf = HistGradientBoostingClassifier(max_iter=300, random_state=42, categorical_features='from_dtype').fit(X_tr_wst_hgb, c_tr_wst)
    hgb_q90 = HistGradientBoostingRegressor(loss='quantile', quantile=0.9, max_iter=300, random_state=42, categorical_features='from_dtype').fit(X_tr_wst_hgb, y_tr_wst)
t_hgb = time.time() - t0

p_hgb = np.maximum(0, hgb_reg.predict(X_te_wst_hgb))
mae, rmse, r2 = regression_metrics(y_te_wst, p_hgb)
clf_acc = accuracy_score(c_te_wst, hgb_clf.predict(X_te_wst_hgb)) * 100
cov90 = (y_te_wst <= hgb_q90.predict(X_te_wst_hgb)).mean() * 100
waste_results.append({
    'Model': 'HistGradientBoosting (Üretim)', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
    'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_hgb, 2),
    'Sınıflandırıcı Doğruluğu (%)': round(clf_acc, 1), 'P90 Kapsama (%)': f"%{cov90:.1f}"
})
print(f"HGB Waste: MAE={mae} kg, RMSE={rmse}, R²={r2}, Clf=%{clf_acc:.1f}, P90=%{cov90:.1f}")

# 2. RF
t0 = time.time()
rf_reg = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X_tr_wst_enc, y_tr_wst)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X_tr_wst_enc, c_tr_wst)
t_rf = time.time() - t0
p_rf = np.maximum(0, rf_reg.predict(X_te_wst_enc))
mae, rmse, r2 = regression_metrics(y_te_wst, p_rf)
clf_acc = accuracy_score(c_te_wst, rf_clf.predict(X_te_wst_enc)) * 100
waste_results.append({
    'Model': 'Random Forest', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
    'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_rf, 2),
    'Sınıflandırıcı Doğruluğu (%)': round(clf_acc, 1), 'P90 Kapsama (%)': '-'
})
joblib.dump({'reg': rf_reg, 'clf': rf_clf, 'encoder': enc_wst}, os.path.join(COMP_DIR, 'rf_model.pkl'))
print(f"RF Waste: MAE={mae} kg, RMSE={rmse}, R²={r2}, Clf=%{clf_acc:.1f}")

# 3. XGBoost
if HAS_XGB:
    t0 = time.time()
    xgb_reg = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.8, random_state=42, verbosity=0).fit(X_tr_wst_enc, y_tr_wst)
    xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')
    le_wst = LabelEncoder().fit(c_tr_wst)
    xgb_clf.fit(X_tr_wst_enc, le_wst.transform(c_tr_wst))
    t_xgb = time.time() - t0
    p_xgb = np.maximum(0, xgb_reg.predict(X_te_wst_enc))
    mae, rmse, r2 = regression_metrics(y_te_wst, p_xgb)
    clf_acc = accuracy_score(le_wst.transform(c_te_wst), xgb_clf.predict(X_te_wst_enc)) * 100
    waste_results.append({
        'Model': 'XGBoost', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
        'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_xgb, 2),
        'Sınıflandırıcı Doğruluğu (%)': round(clf_acc, 1), 'P90 Kapsama (%)': '-'
    })
    joblib.dump({'reg': xgb_reg, 'clf': xgb_clf, 'encoder': enc_wst, 'label_encoder': le_wst}, os.path.join(COMP_DIR, 'xgb_model.pkl'))
    print(f"XGB Waste: MAE={mae} kg, RMSE={rmse}, R²={r2}, Clf=%{clf_acc:.1f}")

# 4. CatBoost
if HAS_CAT:
    t0 = time.time()
    cb_reg = CatBoostRegressor(iterations=400, learning_rate=0.08, depth=6, loss_function='RMSE', cat_features=CAT_COLS, random_seed=42, verbose=0).fit(X_tr_wst_cb, y_tr_wst)
    cb_clf = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=5, cat_features=CAT_COLS, random_seed=42, verbose=0).fit(X_tr_wst_cb, c_tr_wst)
    t_cb = time.time() - t0
    p_cb = np.maximum(0, cb_reg.predict(X_te_wst_cb))
    mae, rmse, r2 = regression_metrics(y_te_wst, p_cb)
    clf_acc = accuracy_score(c_te_wst, cb_clf.predict(X_te_wst_cb)) * 100
    waste_results.append({
        'Model': 'CatBoost', 'MAE (kg)': mae, 'RMSE (kg)': rmse,
        'R² Skoru': r2, 'Eğitim Süresi (s)': round(t_cb, 2),
        'Sınıflandırıcı Doğruluğu (%)': round(clf_acc, 1), 'P90 Kapsama (%)': '-'
    })
    joblib.dump({'reg': cb_reg, 'clf': cb_clf}, os.path.join(COMP_DIR, 'catboost_model.pkl'))
    print(f"CatBoost Waste: MAE={mae} kg, RMSE={rmse}, R²={r2}, Clf=%{clf_acc:.1f}")

wst_comp_df = pd.DataFrame(waste_results)
wst_comp_df.to_csv(os.path.join(COMP_DIR, 'model_karsilastirma.csv'), index=False)
print(f"[KAYDEDİLDİ] model_karsilastirma.csv")

print("\n" + "=" * 60)
print("TALEP MODELLERİ KARŞILAŞTIRMASI:")
print(dem_comp_df.to_string(index=False))
print("\nİSRAF MODELLERİ KARŞILAŞTIRMASI:")
print(wst_comp_df.to_string(index=False))
print("=" * 60)
