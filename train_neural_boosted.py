import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from catboost import CatBoostRegressor

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────────────────────
# 1. MODELS AND ARCHITECTURE
# ─────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.swish1 = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.swish2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.swish1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.swish2(out)
        out = self.dropout(out)
        return residual + out

class TabularResNet(nn.Module):
    def __init__(self, cat_dims, num_features, hidden_dim=128, num_blocks=2, dropout=0.15):
        super().__init__()
        embedding_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim)
            for num_classes, emb_dim in zip(cat_dims, embedding_dims)
        ])
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + num_features
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.swish_input = nn.SiLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_cat, x_num, return_features=False):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([embedded, x_num], dim=1)
        out = self.input_layer(x)
        out = self.ln_input(out)
        out = self.swish_input(out)
        out = self.blocks(out)
        if return_features:
            return out  # Return 128-dim hidden representation
        return self.output_layer(out)

# ─────────────────────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(r"flip.csv")

df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['resolved_dt'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
df['end_dt'] = df['resolved_dt'].fillna(df['closed_dt'])
df['duration_minutes'] = (df['end_dt'] - df['start_dt']).dt.total_seconds() / 60.0

df_clean = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] <= 10080)].copy()
df_clean['target'] = np.log1p(df_clean['duration_minutes'])

df_clean['hour'] = df_clean['start_dt'].dt.hour.fillna(12).astype(int)
df_clean['day_of_week'] = df_clean['start_dt'].dt.dayofweek.fillna(0).astype(int)

cat_cols = ['event_type', 'event_cause', 'requires_road_closure', 'priority', 'police_station', 'corridor', 'zone', 'junction']
num_cols = ['latitude', 'longitude', 'hour', 'day_of_week']

# Fill nulls
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).fillna('missing').str.lower().str.strip()
    
medians = {}
for col in num_cols:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
    medians[col] = median_val

# Encoders for PyTorch
encoders = {}
cat_dims = []
X_cat_encoded = np.zeros((len(df_clean), len(cat_cols)), dtype=np.int64)
for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    unique_vals = list(df_clean[col].unique())
    if 'unknown' not in unique_vals:
        unique_vals.append('unknown')
    le.fit(unique_vals)
    encoders[col] = le
    X_cat_encoded[:, i] = le.transform(df_clean[col])
    cat_dims.append(len(le.classes_))

# Scale cyclical times for PyTorch
df_clean['sin_hour'] = np.sin(2 * np.pi * df_clean['hour'] / 24.0)
df_clean['cos_hour'] = np.cos(2 * np.pi * df_clean['hour'] / 24.0)
df_clean['sin_day'] = np.sin(2 * np.pi * df_clean['day_of_week'] / 7.0)
df_clean['cos_day'] = np.cos(2 * np.pi * df_clean['day_of_week'] / 7.0)
num_cols_pt = ['latitude', 'longitude', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day']

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df_clean[num_cols_pt])

y = df_clean['target'].values

# Split data
X_train_cat, X_val_cat, X_train_num, X_val_num, y_train, y_val = train_test_split(
    X_cat_encoded, X_num_scaled, y, test_size=0.2, random_state=42
)

X_train_raw, X_val_raw, _, _ = train_test_split(
    df_clean[cat_cols + num_cols], df_clean['target'], test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────
# 3. TRAINING STEP 1: PYTORCH REPRESENTATION LEARNING
# ─────────────────────────────────────────────────────────
print("\n--- Step 1: Training PyTorch Tabular ResNet Extractor ---")
train_dataset = TabularDataset(X_train_cat, X_train_num, y_train)
val_dataset = TabularDataset(X_val_cat, X_val_num, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

pt_model = TabularResNet(cat_dims, len(num_cols_pt)).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(pt_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

best_val_loss = float('inf')
early_stop_patience = 12
early_stop_counter = 0

for epoch in range(80):
    pt_model.train()
    for batch_cat, batch_num, batch_y in train_loader:
        batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(pt_model(batch_cat, batch_num), batch_y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    pt_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_cat, batch_num, batch_y in val_loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            loss = criterion(pt_model(batch_cat, batch_num), batch_y)
            val_loss += loss.item() * len(batch_y)
    val_loss /= len(val_dataset)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(pt_model.state_dict(), 'model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            break

pt_model.load_state_dict(torch.load('model.pth'))
pt_model.eval()

# ─────────────────────────────────────────────────────────
# 4. STEP 2: EXTRACT NEURAL REPRESENTATION FEATURES
# ─────────────────────────────────────────────────────────
print("\n--- Step 2: Extracting Latent Neural Features ---")
# Extract for the entire dataset to preserve splits
full_dataset = TabularDataset(X_cat_encoded, X_num_scaled)
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)

neural_features = []
with torch.no_grad():
    for batch_cat, batch_num in full_loader:
        batch_cat, batch_num = batch_cat.to(device), batch_num.to(device)
        features = pt_model(batch_cat, batch_num, return_features=True)
        neural_features.append(features.cpu().numpy())
neural_features = np.vstack(neural_features)

# Append neural features (128 dimensions) to CatBoost dataset
neural_cols = [f"neural_feat_{i}" for i in range(neural_features.shape[1])]
df_neural = pd.DataFrame(neural_features, columns=neural_cols, index=df_clean.index)
df_joint = pd.concat([df_clean[cat_cols + num_cols], df_neural], axis=1)

# Split joint dataset
X_train_joint, X_val_joint, _, _ = train_test_split(
    df_joint, df_clean['target'], test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────
# 5. STEP 3: TRAIN NEURAL-BOOSTED CATBOOST
# ─────────────────────────────────────────────────────────
print("\n--- Step 3: Training Neural-Boosted CatBoost Regressor ---")
cat_features_idx = [i for i, col in enumerate(df_joint.columns) if col in cat_cols]

cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=100,
    cat_features=cat_features_idx
)
cb_model.fit(
    X_train_joint, y_train,
    eval_set=(X_val_joint, y_val),
    early_stopping_rounds=35,
    use_best_model=True
)

cb_preds_log = cb_model.predict(X_val_joint)
cb_preds_actual = np.expm1(cb_preds_log)
y_val_actual = np.expm1(y_val)

# Compute metrics
mae = mean_absolute_error(y_val_actual, cb_preds_actual)
medae = median_absolute_error(y_val_actual, cb_preds_actual)
r2_log = r2_score(y_val, cb_preds_log)

typical_idx = y_val_actual <= 180
mae_typ = mean_absolute_error(y_val_actual[typical_idx], cb_preds_actual[typical_idx])
medae_typ = median_absolute_error(y_val_actual[typical_idx], cb_preds_actual[typical_idx])

print("\n" + "="*50)
print("          NEURAL-BOOSTED CATBOOST EVALUATION")
print("="*50)
print(f"  Ensemble R² (Log-Scale):        {r2_log:.4f} (Previous best: 0.3527)")
print(f"  Ensemble Median Error (MedAE): {medae:.2f} minutes")
print(f"  Ensemble Mean Error (MAE):     {mae:.2f} minutes")
print(f"  Typical Incident MedAE (<=3h): {medae_typ:.2f} minutes")
print(f"  Typical Incident MAE (<=3h):   {mae_typ:.2f} minutes")
print("="*50)

# Save best configurations
preprocessor = {
    'encoders': encoders,
    'scaler': scaler,
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'num_cols_pt': num_cols_pt,
    'cat_dims': cat_dims,
    'medians': medians,
    'neural_cols': neural_cols,
    'is_neural_boosted': True
}

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Ensemble config saved to preprocessor.pkl.")

cb_model.save_model('catboost_model.cbm')
print("CatBoost model saved to catboost_model.cbm.")
