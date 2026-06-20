import pandas as pd
import numpy as np
import pickle
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import TabularResNet
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def generate_plots():
    print("Generating model validation and calibration charts...")
    base_file = r"D:\flip.csv"
    if not os.path.exists(base_file):
        print(f"Base file {base_file} not found.")
        return False
        
    try:
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
            
        model_pt = TabularResNet(
            cat_dims=preprocessor['cat_dims'],
            num_features=len(preprocessor['num_cols_pt']),
            hidden_dim=128,
            num_blocks=2,
            dropout=0.15
        )
        model_pt.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model_pt.eval()
        
        model_cb = CatBoostRegressor()
        model_cb.load_model('catboost_model.cbm')
    except Exception as e:
        print(f"Error loading models for plotting: {e}")
        return False
        
    df = pd.read_csv(base_file)
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['resolved_dt'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    df['end_dt'] = df['resolved_dt'].fillna(df['closed_dt'])
    df['duration_minutes'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60.0
    df_clean = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] <= 10080)].copy()
    
    np.random.seed(42)
    val_sample = df_clean.sample(min(400, len(df_clean)), random_state=42)
    
    cat_cols = preprocessor['cat_cols']
    num_cols = preprocessor['num_cols']
    num_cols_pt = preprocessor['num_cols_pt']
    
    actuals = []
    preds = []
    
    for _, row in val_sample.iterrows():
        try:
            start_dt = pd.to_datetime(row['start_datetime'])
            hour = start_dt.hour
            day_of_week = start_dt.weekday()
        except:
            hour = 12
            day_of_week = 0
            
        event_type = str(row['event_type']).lower().strip()
        event_cause = str(row['event_cause']).lower().strip()
        priority = str(row['priority']).lower().strip()
        requires_road_closure = str(row['requires_road_closure']).lower().strip()
        police_station = str(row['police_station']).lower().strip()
        corridor = str(row['corridor']).lower().strip()
        zone = str(row['zone']).lower().strip()
        junction = str(row['junction']).lower().strip()
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        
        actual = float(row['duration_minutes'])
        
        input_df = pd.DataFrame([{
            'event_type': event_type,
            'event_cause': event_cause,
            'requires_road_closure': requires_road_closure,
            'priority': priority,
            'police_station': police_station,
            'corridor': corridor,
            'zone': zone,
            'junction': junction,
            'latitude': lat,
            'longitude': lon,
            'hour': hour,
            'day_of_week': day_of_week
        }])
        pred_log_cb = model_cb.predict(input_df)[0]
        
        cat_vals = [event_type, event_cause, requires_road_closure, priority, police_station, corridor, zone, junction]
        X_cat = []
        for i, col in enumerate(cat_cols):
            val = cat_vals[i]
            le = preprocessor['encoders'][col]
            if val not in le.classes_:
                val = 'unknown'
            X_cat.append(le.transform([val])[0])
        X_cat = np.array([X_cat], dtype=np.int64)
        
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        sin_day = np.sin(2 * np.pi * day_of_week / 7.0)
        cos_day = np.cos(2 * np.pi * day_of_week / 7.0)
        
        num_vals = [lat, lon, sin_hour, cos_hour, sin_day, cos_day]
        X_num = preprocessor['scaler'].transform([num_vals])
        
        t_cat = torch.tensor(X_cat, dtype=torch.long)
        t_num = torch.tensor(X_num, dtype=torch.float32)
        with torch.no_grad():
            pred_log_pt = model_pt(t_cat, t_num).item()
            
        w_pt = preprocessor['pytorch_weight']
        w_cb = preprocessor['catboost_weight']
        pred_log = w_pt * pred_log_pt + w_cb * pred_log_cb
        pred = max(1.0, np.expm1(pred_log))
        
        actuals.append(actual)
        preds.append(pred)
        
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    # PLOT 1: Calibration Scatter (Light Theme)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    plot_act = np.clip(actuals, 0, 360)
    plot_prd = np.clip(preds, 0, 360)
    
    ax.scatter(plot_act, plot_prd, alpha=0.6, color='#1d4ed8', edgecolors='none', s=25, label='Events')
    ax.plot([0, 360], [0, 360], color='#d97706', linestyle='--', linewidth=1.5, label='Ideal Predictor')
    
    ax.set_xlabel('Actual Duration (Minutes)', color='#475569', fontsize=10)
    ax.set_ylabel('Predicted Duration (Minutes)', color='#475569', fontsize=10)
    ax.set_title('Model Calibration: Actual vs Predicted', color='#0a2240', fontsize=11, fontweight='bold')
    ax.grid(True, color='#cbd5e1', alpha=0.5)
    ax.legend(facecolor='#ffffff', edgecolor='#cbd5e1')
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        
    r2 = r2_score(np.log1p(actuals), np.log1p(preds))
    mae = mean_absolute_error(actuals, preds)
    stats_text = f"Log R²: {r2:.4f}\nMAE: {mae:.1f}m"
    ax.text(20, 300, stats_text, color='#0a2240', fontsize=9,
            bbox=dict(facecolor='#ffffff', alpha=0.9, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
            
    plt.tight_layout()
    plt.savefig('validation_scatter.png', dpi=150, facecolor='#ffffff')
    plt.close()
    
    # PLOT 2: Residuals Plot (Light Theme)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    residuals = np.log1p(actuals) - np.log1p(preds)
    ax.scatter(np.log1p(preds), residuals, alpha=0.5, color='#16a34a', edgecolors='none', s=20)
    ax.axhline(y=0, color='#dc2626', linestyle='-', linewidth=1.2)
    
    ax.set_xlabel('Predicted Log-Duration', color='#475569', fontsize=10)
    ax.set_ylabel('Residual Error (Actual - Predicted)', color='#475569', fontsize=10)
    ax.set_title('Residuals Analysis: Error Distribution', color='#0a2240', fontsize=11, fontweight='bold')
    ax.grid(True, color='#cbd5e1', alpha=0.5)
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=150, facecolor='#ffffff')
    plt.close()
    
    print("Calibration charts saved successfully!")
    return True

if __name__ == "__main__":
    generate_plots()
