"""
Üretim Modellerini Eğitme Scripti (HistGradientBoosting)
=========================================================
Regresyon (israf kg), Kantil (P50/P90) ve Sınıflandırma (israf nedeni) modellerini
zaman bazlı bölme (Ocak-Eylül eğitim / Ekim-Aralık test) ile eğitir ve models/ klasörüne kaydeder.

Çalıştırma:  python scripts/model_egit_uretim.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'yemekhane_oznitelikler.csv'), parse_dates=['Date'])

FEATURE_COLS = [
    'Business_Type', 'Food_Category', 'Campus_Event', 'Weather',
    'Purchase_Quantity', 'Daily_Demand', 'Planned_Surplus', 'Order_Cover',
    'DayOfWeek', 'Month', 'IsWeekend', 'Is_Friday'
]
CATS = {
    'Business_Type': ['Cafeteria', 'Cloud Kitchen', 'Restaurant', 'Supermarket'],
    'Food_Category': ['Bakery', 'Dairy', 'Fruits', 'Meat', 'Prepared Food', 'Vegetables'],
    'Campus_Event': ['No', 'Yes'],
    'Weather': ['Cloudy', 'Hot', 'Rainy', 'Sunny'],
}

def prep(d):
    X = d[FEATURE_COLS].copy()
    for c, cats in CATS.items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return X

train, test = df[df['Date'].dt.month <= 9], df[df['Date'].dt.month >= 10]
Xtr, Xte = prep(train), prep(test)
ytr, yte = train['Waste_Quantity'].values, test['Waste_Quantity'].values
ctr = train['Waste_Reason'].astype(str).to_numpy(dtype=object)
cte = test['Waste_Reason'].astype(str).to_numpy(dtype=object)
print(f"Train: {len(train)}  Test: {len(test)}  Öznitelik: {len(FEATURE_COLS)}")

reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1, categorical_features='from_dtype', random_state=42).fit(Xtr, ytr)
q50 = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=300, learning_rate=0.08,
        random_state=42, categorical_features='from_dtype').fit(Xtr, ytr)
q90 = HistGradientBoostingRegressor(loss='quantile', quantile=0.9, max_iter=300, learning_rate=0.08,
        random_state=42, categorical_features='from_dtype').fit(Xtr, ytr)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42,
        categorical_features='from_dtype').fit(Xtr, ctr)

p = np.maximum(0, reg.predict(Xte))
print(f"MAE={mean_absolute_error(yte, p):.3f}  R2={r2_score(yte, p):.3f}  "
      f"P90 kapsama=%{(yte <= q90.predict(Xte)).mean()*100:.1f}  Sınıflandırma=%{accuracy_score(cte, clf.predict(Xte))*100:.1f}")

out = os.path.join(BASE_DIR, 'models')
joblib.dump(reg, os.path.join(out, 'food_waste_regressor_model.pkl'))
joblib.dump({0.5: q50, 0.9: q90}, os.path.join(out, 'quantile_models.pkl'))
joblib.dump(clf, os.path.join(out, 'food_waste_classifier_model.pkl'))
print("Kaydedildi: models/")
