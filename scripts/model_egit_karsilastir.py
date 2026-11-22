"""
Model Eğitim ve Karşılaştırma Scripti
======================================
HistGradientBoosting (HGB), RandomForest, XGBoost, CatBoost
4 modeli eğitir, karşılaştırır ve .pkl olarak kaydeder.

Çalıştırma:
  python model_egit_karsilastir.py
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
print(f"[OK] Veri yüklendi: {len(df)} satır, {df.shape[1]} sütun")

FEATURE_COLS = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'Purchase_Quantity', 'Daily_Demand', 'Planned_Surplus', 'Order_Cover',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]
TARGET_REG = 'Waste_Quantity'
TARGET_CLF = 'Waste_Reason'

CAT_COLS = ['Business_Type', 'Food_Category', 'Campus_Event', 'Weather']
NUM_COLS  = [c for c in FEATURE_COLS if c not in CAT_COLS]

# ============================================================
# 2. ZAMAN BAZLI TRAIN/TEST BÖLME  (Ocak-Eylül / Ekim-Aralık)
# ============================================================
train = df[df['Date'].dt.month <= 9]
test  = df[df['Date'].dt.month >= 10]
print(f"[OK] Train: {len(train)}, Test: {len(test)}")

X_train_raw = train[FEATURE_COLS].copy()
X_test_raw  = test[FEATURE_COLS].copy()
y_train_reg = train[TARGET_REG].values
y_test_reg  = test[TARGET_REG].values
y_train_clf = train[TARGET_CLF].astype(str).to_numpy(dtype=object)
y_test_clf  = test[TARGET_CLF].astype(str).to_numpy(dtype=object)

# ============================================================
# 3. KATEGORİK ENCODER (XGBoost / RandomForest / CatBoost için)
# ============================================================
from sklearn.preprocessing import OrdinalEncoder

CAT_CATEGORIES = {
    'Business_Type': ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket'],
    'Food_Category': ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables'],
    'Campus_Event' : ['No', 'Yes'],
    'Weather'      : ['Cloudy', 'Hot', 'Rainy', 'Sunny'],
}

enc = OrdinalEncoder(
    categories=[CAT_CATEGORIES[c] for c in CAT_COLS],
    handle_unknown='use_encoded_value', unknown_value=-1
)

def encode(df_in):
    out = df_in.copy()
    out[CAT_COLS] = enc.transform(df_in[CAT_COLS])
    return out.astype(float)

enc.fit(X_train_raw[CAT_COLS])
X_train_enc = encode(X_train_raw)
X_test_enc  = encode(X_test_raw)

# HGB için pandas Categorical
def make_hgb_df(df_raw):
    out = df_raw.copy()
    for col, cats in CAT_CATEGORIES.items():
        out[col] = pd.Categorical(out[col], categories=cats)
    return out

X_train_hgb = make_hgb_df(X_train_raw)
X_test_hgb  = make_hgb_df(X_test_raw)

# ============================================================
# 4. METRIK FONKSİYONU
# ============================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ============================================================
# 5. MODEL TANIMLAMALARI
# ============================================================
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
    print("[OK] XGBoost import edildi")
except ImportError:
    HAS_XGB = False
    print("[SKIP] XGBoost yok")

# CatBoost
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
    print("[OK] CatBoost import edildi")
except ImportError:
    HAS_CAT = False
    print("[SKIP] CatBoost yok")

results = {}  # {model_name: {mae, rmse, r2, train_time, clf_acc}}

# ============================================================
# 6a. HistGradientBoosting (HGB) — PRODUCTION MODEL
# ============================================================
print("\n--- HistGradientBoosting (Üretim Modeli) ---")

t0 = time.time()
hgb_reg = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=0.1,
    categorical_features='from_dtype', random_state=42
)
hgb_reg.fit(X_train_hgb, y_train_reg)
train_time_hgb = time.time() - t0

# Quantile modelleri
hgb_q50 = HistGradientBoostingRegressor(
    loss='quantile', quantile=0.5, max_iter=300, learning_rate=0.08, random_state=42,
    categorical_features='from_dtype'
)
hgb_q90 = HistGradientBoostingRegressor(
    loss='quantile', quantile=0.9, max_iter=300, learning_rate=0.08, random_state=42,
    categorical_features='from_dtype'
)
hgb_q50.fit(X_train_hgb, y_train_reg)
hgb_q90.fit(X_train_hgb, y_train_reg)

hgb_clf = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, random_state=42,
    categorical_features='from_dtype'
)
hgb_clf.fit(X_train_hgb, y_train_clf)

# Karşılaştırmada GERÇEK üretim modelleri (models/*.pkl) değerlendirilir; bulunamazsa yeni eğitilenler kullanılır.
try:
    hgb_reg = joblib.load(os.path.join(MODEL_DIR, 'food_waste_regressor_model.pkl'))
    hgb_clf = joblib.load(os.path.join(MODEL_DIR, 'food_waste_classifier_model.pkl'))
    _q = joblib.load(os.path.join(MODEL_DIR, 'quantile_models.pkl'))
    hgb_q90 = _q[0.9]
    print('  [INFO] Üretim modelleri models/ klasöründen yüklendi ve değerlendirildi')
except Exception as _e:
    print('  [WARN] Üretim modeli yüklenemedi, yeni eğitilen kullanılıyor:', _e)

pred_hgb = np.maximum(0, hgb_reg.predict(X_test_hgb))
mae, rmse, r2 = regression_metrics(y_test_reg, pred_hgb)
clf_acc = accuracy_score(y_test_clf, hgb_clf.predict(X_test_hgb))

# P90 coverage
p90_pred = hgb_q90.predict(X_test_hgb)
coverage_90 = (y_test_reg <= p90_pred).mean()

results['HGB'] = dict(mae=mae, rmse=rmse, r2=r2, train_time=train_time_hgb,
                       clf_acc=clf_acc, coverage_90=coverage_90)
print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  Clf=%{clf_acc*100:.1f}  P90-cov=%{coverage_90*100:.1f}")

# KAYDET (mevcut üretim modelleri güncelle)
# Üretim modellerinin (food_waste_*.pkl) üzerine YAZILMAZ; sadece karşılaştırma yapılır.
print("  [INFO] Üretim modelleri korunuyor (üzerine yazılmadı)")

# ============================================================
# 6b. RandomForest
# ============================================================
print("\n--- RandomForest ---")

t0 = time.time()
rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
rf_reg.fit(X_train_enc, y_train_reg)
train_time_rf = time.time() - t0

rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
rf_clf.fit(X_train_enc, y_train_clf)

pred_rf = np.maximum(0, rf_reg.predict(X_test_enc))
mae, rmse, r2 = regression_metrics(y_test_reg, pred_rf)
clf_acc = accuracy_score(y_test_clf, rf_clf.predict(X_test_enc))
results['RandomForest'] = dict(mae=mae, rmse=rmse, r2=r2, train_time=train_time_rf,
                                clf_acc=clf_acc, coverage_90=None)
print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  Clf=%{clf_acc*100:.1f}  t={train_time_rf:.1f}s")

joblib.dump({'reg': rf_reg, 'clf': rf_clf, 'encoder': enc}, os.path.join(COMP_DIR, 'rf_model.pkl'))
print("  [SAVED] rf_model.pkl")

# ============================================================
# 6c. XGBoost
# ============================================================
if HAS_XGB:
    print("\n--- XGBoost ---")
    t0 = time.time()
    xgb_reg = XGBRegressor(
        n_estimators=400, learning_rate=0.08, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbosity=0
    )
    xgb_reg.fit(X_train_enc, y_train_reg)
    train_time_xgb = time.time() - t0

    xgb_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42, verbosity=0,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_train_clf)
    xgb_clf.fit(X_train_enc, le.transform(y_train_clf))

    pred_xgb = np.maximum(0, xgb_reg.predict(X_test_enc))
    mae, rmse, r2 = regression_metrics(y_test_reg, pred_xgb)
    clf_acc = accuracy_score(le.transform(y_test_clf), xgb_clf.predict(X_test_enc))
    results['XGBoost'] = dict(mae=mae, rmse=rmse, r2=r2, train_time=train_time_xgb,
                               clf_acc=clf_acc, coverage_90=None)
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  Clf=%{clf_acc*100:.1f}  t={train_time_xgb:.1f}s")

    joblib.dump({'reg': xgb_reg, 'clf': xgb_clf, 'encoder': enc, 'label_encoder': le}, os.path.join(COMP_DIR, 'xgb_model.pkl'))
    print("  [SAVED] xgb_model.pkl")

# ============================================================
# 6d. CatBoost
# ============================================================
if HAS_CAT:
    print("\n--- CatBoost ---")
    # CatBoost kategorikleri doğal destekler, fakat string olmali
    cat_feature_indices = [FEATURE_COLS.index(c) for c in CAT_COLS]

    X_train_cat = X_train_raw.copy()
    X_test_cat  = X_test_raw.copy()
    # CatBoost requires string categoricals
    for c in CAT_COLS:
        X_train_cat[c] = X_train_cat[c].fillna('Bilinmiyor').astype(str)
        X_test_cat[c]  = X_test_cat[c].fillna('Bilinmiyor').astype(str)

    t0 = time.time()
    cb_reg = CatBoostRegressor(
        iterations=400, learning_rate=0.08, depth=6,
        loss_function='RMSE', cat_features=CAT_COLS,
        random_seed=42, verbose=0
    )
    cb_reg.fit(X_train_cat, y_train_reg)
    train_time_cb = time.time() - t0

    cb_clf = CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=5,
        cat_features=CAT_COLS, random_seed=42, verbose=0
    )
    cb_clf.fit(X_train_cat, y_train_clf)

    pred_cb = np.maximum(0, cb_reg.predict(X_test_cat))
    mae, rmse, r2 = regression_metrics(y_test_reg, pred_cb)
    clf_acc = accuracy_score(y_test_clf, cb_clf.predict(X_test_cat))
    results['CatBoost'] = dict(mae=mae, rmse=rmse, r2=r2, train_time=train_time_cb,
                                clf_acc=clf_acc, coverage_90=None)
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  Clf=%{clf_acc*100:.1f}  t={train_time_cb:.1f}s")

    joblib.dump({'reg': cb_reg, 'clf': cb_clf}, os.path.join(COMP_DIR, 'catboost_model.pkl'))
    print("  [SAVED] catboost_model.pkl")

# ============================================================
# 7. KARŞILAŞTIRMA TABLOSU (CSV)
# ============================================================
rows = []
for name, r in results.items():
    rows.append({
        'Model': name,
        'MAE (kg)': round(r['mae'], 3),
        'RMSE (kg)': round(r['rmse'], 3),
        'R² Skoru': round(r['r2'], 3),
        'Eğitim Süresi (s)': round(r['train_time'], 1),
        'Sınıflandırıcı Doğruluğu (%)': round(r['clf_acc'] * 100, 1),
        'P90 Kapsama (%)': round(r.get('coverage_90', 0) * 100, 1) if r.get('coverage_90') else '-'
    })

comp_df = pd.DataFrame(rows)
comp_path = os.path.join(COMP_DIR, 'model_karsilastirma.csv')
comp_df.to_csv(comp_path, index=False)
print(f"\n[DONE] Karşılaştırma tablosu: {comp_path}")
print(comp_df.to_string(index=False))
