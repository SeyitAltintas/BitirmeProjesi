"""
Üretim Modellerini Eğitme Scripti (HistGradientBoosting)
=========================================================
1. Talep Tahmin Modeli (Daily_Demand kg)
2. İsraf Miktarı Regresyon Modeli (Waste_Quantity kg)
3. İsraf Belirsizlik Kantil Modelleri (P50 & P90 kg)
4. İsraf Nedeni Sınıflandırma Modeli (Waste_Reason)

Zaman bazlı bölme (Ocak-Eylül eğitim / Ekim-Aralık test) ile eğitilir ve models/ klasörüne kaydedilir.
Çalıştırma:  python scripts/model_egit_uretim.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'yemekhane_oznitelikler.csv')
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

# -------------------------------------------------------------
# 1. ORTAK KATEGORİK DEĞERLER VE ÖN İŞLEME
# -------------------------------------------------------------
CATS = {
    'Business_Type': ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket'],
    'Food_Category': ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables'],
    'Campus_Event': ['No', 'Yes'],
    'Weather': ['Cloudy', 'Hot', 'Rainy', 'Sunny'],
}

df['Weather'] = df['Weather'].fillna('Cloudy')

DEMAND_FEATURES = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]

WASTE_FEATURES = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'Purchase_Quantity', 'Daily_Demand', 'Planned_Surplus', 'Order_Cover',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]

def prep_df(d, feature_cols):
    X = d[feature_cols].copy()
    for c, cats in CATS.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cats)
    return X

train_df = df[df['Date'].dt.month <= 9].copy()
test_df  = df[df['Date'].dt.month >= 10].copy()

# -------------------------------------------------------------
# 2. AŞAMA 1: TALEP TAHMİN MODELİ (Daily_Demand)
# -------------------------------------------------------------
print("=" * 60)
print("1. AŞAMA: TALEP TAHMİN MODELİ EĞİTİMİ (Daily_Demand kg)")
print("=" * 60)

# Demand için NaN satırları ayıkla
train_demand_df = train_df.dropna(subset=['Daily_Demand']).copy()
test_demand_df  = test_df.dropna(subset=['Daily_Demand']).copy()

X_tr_dem = prep_df(train_demand_df, DEMAND_FEATURES)
X_te_dem = prep_df(test_demand_df, DEMAND_FEATURES)
y_tr_dem = train_demand_df['Daily_Demand'].values
y_te_dem = test_demand_df['Daily_Demand'].values

# Baseline
base_median = np.median(y_tr_dem)
base_mae = mean_absolute_error(y_te_dem, np.full_like(y_te_dem, base_median))
print(f"Referans (Sabit Medyan) Talep MAE: {base_mae:.2f} kg")

demand_model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
    min_samples_leaf=15, l2_regularization=0.1,
    categorical_features='from_dtype', random_state=42
)
demand_model.fit(X_tr_dem, y_tr_dem)

pred_dem = demand_model.predict(X_te_dem)
dem_mae = mean_absolute_error(y_te_dem, pred_dem)
dem_rmse = np.sqrt(mean_squared_error(y_te_dem, pred_dem))
dem_r2 = r2_score(y_te_dem, pred_dem)
print(f"HGB Talep Modeli Başarımı:")
print(f"  MAE  : {dem_mae:.2f} kg (Referansa göre %{(base_mae - dem_mae)/base_mae*100:.1f} iyileşme)")
print(f"  RMSE : {dem_rmse:.2f} kg")
print(f"  R²   : {dem_r2:.3f}")

# -------------------------------------------------------------
# 3. AŞAMA 2: İSRAF REGRESYON, KUANTİL VE SINIFLANDIRMA MODELLERİ
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. AŞAMA: İSRAF REGRESYON, KUANTİL VE SINIFLANDIRMA MODELLERİ")
print("=" * 60)

X_tr_wst = prep_df(train_df, WASTE_FEATURES)
X_te_wst = prep_df(test_df, WASTE_FEATURES)
y_tr_wst = train_df['Waste_Quantity'].values
y_te_wst = test_df['Waste_Quantity'].values
c_tr_wst = train_df['Waste_Reason'].astype(str).to_numpy(dtype=object)
c_te_wst = test_df['Waste_Reason'].astype(str).to_numpy(dtype=object)

# Regresyon
waste_reg = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.1,
    categorical_features='from_dtype', random_state=42
).fit(X_tr_wst, y_tr_wst)

# Kuantiller (P50 & P90)
q50 = HistGradientBoostingRegressor(
    loss='quantile', quantile=0.5, max_iter=300, learning_rate=0.08,
    random_state=42, categorical_features='from_dtype'
).fit(X_tr_wst, y_tr_wst)

q90 = HistGradientBoostingRegressor(
    loss='quantile', quantile=0.9, max_iter=300, learning_rate=0.08,
    random_state=42, categorical_features='from_dtype'
).fit(X_tr_wst, y_tr_wst)

# Sınıflandırıcı
waste_clf = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, random_state=42,
    categorical_features='from_dtype'
).fit(X_tr_wst, c_tr_wst)

pred_wst = np.maximum(0, waste_reg.predict(X_te_wst))
wst_mae = mean_absolute_error(y_te_wst, pred_wst)
wst_rmse = np.sqrt(mean_squared_error(y_te_wst, pred_wst))
wst_r2 = r2_score(y_te_wst, pred_wst)
p90_cov = (y_te_wst <= q90.predict(X_te_wst)).mean() * 100
clf_acc = accuracy_score(c_te_wst, waste_clf.predict(X_te_wst)) * 100

print(f"HGB İsraf Modelleri Başarımı:")
print(f"  MAE          : {wst_mae:.3f} kg")
print(f"  RMSE         : {wst_rmse:.3f} kg")
print(f"  R²           : {wst_r2:.3f}")
print(f"  P90 Kapsama  : %{p90_cov:.1f}")
print(f"  Sınıflandırıcı : %{clf_acc:.1f}")

# -------------------------------------------------------------
# 4. MODELLERİ KAYDETME
# -------------------------------------------------------------
out_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(out_dir, exist_ok=True)

joblib.dump(demand_model, os.path.join(out_dir, 'food_demand_model.pkl'))
joblib.dump(waste_reg, os.path.join(out_dir, 'food_waste_regressor_model.pkl'))
joblib.dump({0.5: q50, 0.9: q90}, os.path.join(out_dir, 'quantile_models.pkl'))
joblib.dump(waste_clf, os.path.join(out_dir, 'food_waste_classifier_model.pkl'))

print("\n" + "=" * 60)
print(f"[TAMAMLANDI] 4 model başarıyla kaydedildi -> {out_dir}")
print("  1. food_demand_model.pkl")
print("  2. food_waste_regressor_model.pkl")
print("  3. quantile_models.pkl (P50 & P90)")
print("  4. food_waste_classifier_model.pkl")
print("=" * 60)
