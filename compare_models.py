import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. Load and prepare data (same preprocessing logic as model.py)
df = pd.read_csv(r"D:\flip.csv")
df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['resolved_dt'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
df['end_dt'] = df['resolved_dt'].fillna(df['closed_dt'])
df['duration_minutes'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60.0

df_clean = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] <= 10080)].copy()
df_clean['target'] = np.log1p(df_clean['duration_minutes'])

# Extract temporal features
df_clean['hour'] = df_clean['start_dt'].dt.hour.fillna(12)
df_clean['day_of_week'] = df_clean['start_dt'].dt.dayofweek.fillna(0)

# Categorical and numerical columns
cat_cols = ['event_type', 'event_cause', 'requires_road_closure', 'priority', 'police_station', 'corridor', 'zone', 'junction']
num_cols = ['latitude', 'longitude', 'hour', 'day_of_week']

# Fill nulls
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).fillna('Missing').str.lower().str.strip()
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Label encode for LightGBM
df_lgb = df_clean.copy()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_lgb[col] = le.fit_transform(df_lgb[col])
    encoders[col] = le

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(
    df_lgb[cat_cols + num_cols], df_lgb['target'], test_size=0.2, random_state=42
)

# Convert targets back to minutes for evaluation
y_val_actual = np.expm1(y_val)

# ─────────────────────────────────────────────────────────
# Model 1: LightGBM Regressor
# ─────────────────────────────────────────────────────────
print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

lgb_preds_log = lgb_model.predict(X_val)
lgb_preds_actual = np.expm1(lgb_preds_log)

lgb_mae = mean_absolute_error(y_val_actual, lgb_preds_actual)
lgb_medae = median_absolute_error(y_val_actual, lgb_preds_actual)
lgb_r2_log = r2_score(y_val, lgb_preds_log)

# Typical errors (actual duration <= 180 min)
typical_idx = y_val_actual <= 180
lgb_mae_typ = mean_absolute_error(y_val_actual[typical_idx], lgb_preds_actual[typical_idx])
lgb_medae_typ = median_absolute_error(y_val_actual[typical_idx], lgb_preds_actual[typical_idx])

# ─────────────────────────────────────────────────────────
# Model 2: CatBoost Regressor
# ─────────────────────────────────────────────────────────
print("Training CatBoost...")
# Split using raw categoricals for CatBoost native support
X_train_raw, X_val_raw, _, _ = train_test_split(
    df_clean[cat_cols + num_cols], df_clean['target'], test_size=0.2, random_state=42
)

cat_features_idx = [i for i, col in enumerate(cat_cols + num_cols) if col in cat_cols]

cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
    cat_features=cat_features_idx
)
cb_model.fit(
    X_train_raw, y_train,
    eval_set=(X_val_raw, y_val),
    early_stopping_rounds=20
)

cb_preds_log = cb_model.predict(X_val_raw)
cb_preds_actual = np.expm1(cb_preds_log)

cb_mae = mean_absolute_error(y_val_actual, cb_preds_actual)
cb_medae = median_absolute_error(y_val_actual, cb_preds_actual)
cb_r2_log = r2_score(y_val, cb_preds_log)

cb_mae_typ = mean_absolute_error(y_val_actual[typical_idx], cb_preds_actual[typical_idx])
cb_medae_typ = median_absolute_error(y_val_actual[typical_idx], cb_preds_actual[typical_idx])

# Output comparison table
print("\n" + "="*50)
print("             MODEL COMPARISON RESULTS")
print("="*50)
print(f"{'Metric':<25} | {'LightGBM':<10} | {'CatBoost':<10}")
print("-"*50)
print(f"{'Val R² (Log-Scale)':<25} | {lgb_r2_log:<10.4f} | {cb_r2_log:<10.4f}")
print(f"{'Global Median Error (Min)':<25} | {lgb_medae:<10.2f} | {cb_medae:<10.2f}")
print(f"{'Global MAE (Min)':<25} | {lgb_mae:<10.2f} | {cb_mae:<10.2f}")
print(f"{'Typical MedAE (<=3h)':<25} | {lgb_medae_typ:<10.2f} | {cb_medae_typ:<10.2f}")
print(f"{'Typical MAE (<=3h)':<25} | {lgb_mae_typ:<10.2f} | {cb_mae_typ:<10.2f}")
print("="*50)
