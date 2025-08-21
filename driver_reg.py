import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as md
import torch.utils.data as data
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTEENN

from sklearn.metrics import precision_recall_curve
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    auc
)

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ss = StandardScaler()
mm = MinMaxScaler()

# 1) NEW: global scalers for static features
global_scalers = {
    "minmax": MinMaxScaler(),
    "std":    StandardScaler()
}

# 2) per-depth scalers container
depth_scalers = {}

import torch
import torch.nn as nn

def preprocess(df):
    # keep original coords for diagnostics
    df = add_time_features(df)
    df = encode_depth(df)
    
    # drop columns we don’t want to scale / feed directly
    drop_cols = ['ocean_date_time', 'ocean_time', 'depth', 'depth_cat']  # keep 'oxyg' for regression
    df_model = df.drop(columns=drop_cols)
    
    # split features into two scaler groups
    latlon_cols = ['lat_rho', 'lon_rho']
    phys_cols   = ['SOCalt', 'PEA', 'DCPtemp']
    time_cols   = ['doy_sin','doy_cos','mon_sin','mon_cos','hour_sin','hour_cos']
    depth_ohe_cols = [c for c in df_model.columns if c.startswith('depth_')]
    
    # fit scalers on full train set (you can refactor to do per‐depth if desired)
    scaler_ll = StandardScaler()
    scaler_phys = MinMaxScaler(feature_range=(0,1))
    
    df_model[latlon_cols] = scaler_ll.fit_transform(df_model[latlon_cols])
    df_model[phys_cols]   = scaler_phys.fit_transform(df_model[phys_cols])
    
    # everything else (time and depth_ohe) is already ~[-1,1] or {0,1}
    final_cols = latlon_cols + phys_cols + time_cols + depth_ohe_cols
    return df_model[final_cols]

def add_time_features(df):
    dt = df['ocean_date_time']
    # day of year
    df['doy_sin'] = np.sin(2*np.pi * dt.dt.dayofyear/365)
    df['doy_cos'] = np.cos(2*np.pi * dt.dt.dayofyear/365)
    # month
    df['mon_sin'] = np.sin(2*np.pi * (dt.dt.month-1)/12)
    df['mon_cos'] = np.cos(2*np.pi * (dt.dt.month-1)/12)
    # hour
    df['hour_sin'] = np.sin(2*np.pi * dt.dt.hour/24)
    df['hour_cos'] = np.cos(2*np.pi * dt.dt.hour/24)
    return df

# 2) OPTIONAL: ONE-HOT ENCODE DEPTH AS CATEGORICAL BUCKETS
def encode_depth(df):
    # example: bin depths into 5 buckets
    df['depth_cat'] = pd.qcut(df['depth'], q=5, labels=False)
    depth_ohe = pd.get_dummies(df['depth_cat'], prefix='depth')
    return pd.concat([df, depth_ohe], axis=1)


def make_dataloader(X_tensor, y_tensor, batch_size=1024):
    # compute sample weights to balance classes on the fly
    counts = np.bincount(y_tensor.cpu().numpy().astype(int))
    class_weights = 1. / counts
    sample_weights = class_weights[y_tensor.cpu().numpy().astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


# LSTM Classifier (unchanged)
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)  # regression output
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers*(2 if self.lstm.bidirectional else 1),
                         x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers*(2 if self.lstm.bidirectional else 1),
                         x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1])  # shape (batch, 1)

# STT (Spatio-Temporal Transformer) Definition
class STT(nn.Module):
    def __init__(self, input_dim, nhead=16, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                    batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.regressor = nn.Linear(dim_feedforward, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.regressor(x.mean(dim=1))  # shape (batch, 1)

# Medformer-style Transformer (simplified)
class Medformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_linear = nn.Linear(embed_dim, 1)  # attention weights
        self.regressor = nn.Linear(embed_dim, 1)
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        attn = torch.softmax(self.attn_linear(x), dim=1)  # shape (B, T, 1)
        pooled = (attn * x).sum(dim=1)
        return self.regressor(pooled)

# Focal Loss implementation for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    def forward(self, input, target):
        BCE_loss = self.ce(input, target)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() if self.reduction=='mean' else F_loss

def get_model(model_type, input_dim):
    if model_type == 'lstm':
        return LSTMClassifier(input_dim, 120, 2, dropout=0.3, bidirectional=True)
    elif model_type == 'stt':
        return STT(input_dim)
    elif model_type == 'medformer':
        return Medformer(input_dim)
    else:
        raise ValueError("Unknown model type")

def apply_smote(X_tensor, y_tensor):
    """
    Apply SMOTE to sequence data.
    X_tensor: (N, T, F)
    y_tensor: (N,)
    """
    X_flat = X_tensor.view(X_tensor.size(0), -1).cpu().numpy()
    y_np = y_tensor.cpu().numpy()

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_flat, y_np)

    X_res_tensor = torch.tensor(X_res, dtype=torch.float32).view(-1, X_tensor.shape[1], X_tensor.shape[2])
    y_res_tensor = torch.tensor(y_res, dtype=torch.float32)

    return X_res_tensor, y_res_tensor

def create_dataset(X, y, lookback):
    """
    Converts sequences into sliding window format for LSTM
    X: [n_samples, n_features]
    y: [n_samples, 1]
    Returns: tensors of shape (n_windows, lookback, n_features) and (n_windows, 1)
    """
    X1, y1 = [], []
    for i in range(len(X) - lookback):
        X1.append(X[i:i + lookback])
        y1.append(y[i + lookback])
    return torch.tensor(np.array(X1), dtype=torch.float32), torch.tensor(np.array(y1), dtype=torch.float32)

# def prepare_dataset(df, predictor="oxy_class", lookback=30, test_year=2020, test_month=8, month_list=[5,6,7,8]):
#     train_list_x, train_list_y = [], []
#     test_list_x, test_list_y = [], []
#     train_list_lat_lon, test_list_lat_lon = [], []

#     for value in tqdm(df.depth.unique()):
#         df_depth = df[(df['depth'] == value) & (df['ocean_date_time'].dt.month.isin(month_list))]
#         df_test = df_depth[(df_depth['ocean_date_time'].dt.year == test_year) &
#                            (df_depth['ocean_date_time'].dt.month == test_month)]
#         df_train = df_depth[~df_depth.index.isin(df_test.index)]

#         train_x_raw = df_train[['SOCalt', 'PEA', 'DCPtemp', 'depth']]
#         test_x_raw = df_test[['SOCalt', 'PEA', 'DCPtemp', 'depth']]
#         train_y_raw = df_train[[predictor]]
#         test_y_raw = df_test[[predictor]]

#         X_train_scaled = mm.fit_transform(train_x_raw)
#         X_test_scaled = mm.transform(test_x_raw)

#         if predictor != "oxy_class":
#             y_train_scaled = mm.fit_transform(train_y_raw)
#             y_test_scaled = mm.transform(test_y_raw)
#         else:
#             y_train_scaled = train_y_raw.values
#             y_test_scaled = test_y_raw.values

#         train_x, train_y = create_dataset(X_train_scaled, y_train_scaled, lookback)
#         test_x, test_y = create_dataset(X_test_scaled, y_test_scaled, lookback)

#         df_train_lat_lon = df_train[['lat_rho', 'lon_rho']].iloc[:-lookback]
#         df_test_lat_lon = df_test[['lat_rho', 'lon_rho']].iloc[:-lookback]

#         train_list_x.append(train_x)
#         train_list_y.append(train_y)
#         test_list_x.append(test_x)
#         test_list_y.append(test_y)
#         train_list_lat_lon.append(df_train_lat_lon)
#         test_list_lat_lon.append(df_test_lat_lon)

#     # Stack tensors
#     X_train = torch.vstack(train_list_x)
#     y_train = torch.vstack(train_list_y).squeeze(-1)
#     X_test = torch.vstack(test_list_x)
#     y_test = torch.vstack(test_list_y).squeeze(-1)

#     # Lat/lon as DataFrames
#     df_train_lat_lon = pd.concat(train_list_lat_lon, ignore_index=True)
#     df_test_lat_lon = pd.concat(test_list_lat_lon, ignore_index=True)

#     return X_train, y_train, X_test, y_test, df_train_lat_lon, df_test_lat_lon


def prepare_dataset_2(
    df,
    predictor="oxyg",
    lookback=7,
    test_year=2020,
    test_month=8,
    month_list=[5,6,7,8]
):
    # 0) List out the columns you DON'T want to scale / feed to the network:
    exclude = {predictor, 'depth', 'ocean_date_time', 'ocean_date'}
    feature_cols = [c for c in df.columns if c not in exclude]

    train_x_list, train_y_list = [], []
    test_x_list,  test_y_list  = [], []

    for depth_val in tqdm(df.depth.unique()):
        # 1) filter by depth & month
        df_depth = df[
            (df.depth == depth_val) &
            (df.ocean_date_time.dt.month.isin(month_list))
        ]

        # 2) train/test split by year & month
        df_test  = df_depth[
            (df_depth.ocean_date_time.dt.year  == test_year) &
            (df_depth.ocean_date_time.dt.month == test_month)
        ]
        df_train = df_depth.drop(df_test.index)

        # 3) inside the loop, make a *local* copy of feature_cols so we don't
        #    accidentally modify the master list:
        feats = list(feature_cols)

        # (just in case) re‑ensure no datetime sneaks in:
        if 'ocean_date_time' in feats:
            feats.remove('ocean_date_time')

        # 4) pull out numpy arrays
        X_tr_raw = df_train[feats].to_numpy(dtype=float)
        X_te_raw = df_test[feats].to_numpy(dtype=float)
        y_tr_raw = df_train[[predictor]].to_numpy(dtype=float)
        y_te_raw = df_test[[predictor]].to_numpy(dtype=float)

        # 5) scale (you can also move mm.fit_transform *outside* the loop
        #    if you want a global scaler instead of per‑depth)
        X_tr = mm.fit_transform(X_tr_raw)
        X_te = mm.fit_transform(X_te_raw)

        # 6) build sliding windows
        X_tr_win, y_tr_win = create_dataset(X_tr, y_tr_raw, lookback)
        X_te_win, y_te_win = create_dataset(X_te, y_te_raw, lookback)

        train_x_list.append(X_tr_win)
        train_y_list.append(y_tr_win)
        test_x_list.append(X_te_win)
        test_y_list.append(y_te_win)

    # 7) stack them all back together
    X_train = torch.vstack(train_x_list)
    y_train = torch.vstack(train_y_list).squeeze(-1)
    X_test = torch.vstack(test_x_list)
    y_test = torch.vstack(test_y_list).squeeze(-1)

    return X_train, y_train, X_test, y_test

def prepare_dataset_regression(
    df,
    predictor="oxyg",
    lookback=7,
    test_year=2020,
    test_month=8,
    month_list=[5,6,7,8]
):
    # 1) keep only the columns we need
    feats = ['SOCalt', 'PEA', 'DCPtemp', 'depth']
    cols  = feats + ['ocean_date_time', predictor]
    df2   = df[cols].dropna().sort_values('ocean_date_time')

    # 2) filter to the months of interest
    df2 = df2[df2['ocean_date_time'].dt.month.isin(month_list)]

    # 3) split train vs test on year/month
    is_test  = (df2['ocean_date_time'].dt.year  == test_year) & \
               (df2['ocean_date_time'].dt.month == test_month)
    df_train = df2[~is_test]
    df_test  = df2[ is_test]

    # 4) extract raw numpy arrays
    X_train_raw = df_train[feats].to_numpy(dtype=float)
    X_test_raw  = df_test [feats].to_numpy(dtype=float)
    y_train_raw = df_train[[predictor]].to_numpy(dtype=float)
    y_test_raw  = df_test [[predictor]].to_numpy(dtype=float)

    # 5) global scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform  (X_test_raw)

    # 6) build sliding windows
    X_tr_win, y_tr_win = create_dataset(X_train, y_train_raw, lookback)
    X_te_win, y_te_win = create_dataset(X_test,  y_test_raw,  lookback)

    # 7) wrap into torch tensors
    X_train_tensor, y_train_tensor = X_tr_win.detach(), y_tr_win.detach().squeeze(-1)
    X_test_tensor,  y_test_tensor  = X_te_win.detach(), y_te_win.detach().squeeze(-1)

    print(f"X_train_tensor shape: {X_train_tensor.shape}, y_train_tensor shape: {y_train_tensor.shape}")
    print(f"X_test_tensor shape: {X_test_tensor.shape}, y_test_tensor shape: {y_test_tensor.shape}")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def compute_class_weight(y_train):
    return None  # Not needed for regression

def train_model(model, loss_fn, optimizer, loader, device, n_epochs=50):
    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze(-1)
            loss = loss_fn(y_pred, y_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(loader):.4f}")

def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).squeeze(-1).cpu().numpy()
    y_true = y_test.cpu().numpy().flatten()
    # Metrics
    results = {
        'MSE': mse(y_true, y_pred),
        'RMSE': np.sqrt(mse(y_true, y_pred)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'R2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }
    # Plot true vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    plt.savefig('regression_true_vs_pred.png')
    return results, (y_true, y_pred)

def main():
    ds = xr.open_dataset('PEA_SOCalt_DCPtemp_hypxia_oxygen_depth_3D_2018_2020_exp14_2.nc', decode_times=False)
    df = ds.to_dataframe()

    #df = df[(df.index > pd.to_datetime('2013-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
    #df = df[(df.index > pd.to_datetime('2018-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
    #df_test = df[(df.index > pd.to_datetime('2019-01-01')) & (df.index < pd.to_datetime('2019-12-31'))]

    df.reset_index(inplace=True)

    df = df.dropna(axis=0)

    df = df[(df['SOCalt'] > 0.0) & (df['PEA'] > 0.0) & (df['DCPtemp'] > 0.0)]

    df['ocean_time'] = df['ocean_time'].astype(int) - 719529
    df['ocean_date_time'] = df['ocean_time'].apply(lambda x: md.num2date(x) if pd.notnull(x) else pd.NaT)
    df['ocean_time'] = df['ocean_time']

    df['ocean_date'] = pd.to_datetime(df['ocean_date_time']).dt.date
    df = df.set_index('ocean_date')

    df = df.sort_index()

    df_scale_vector_rbf = df[['lat_rho', 'lon_rho', 'ocean_date_time', 'ocean_time', 'SOCalt', 'PEA', 'DCPtemp', 'depth', 'oxyg']]

    df_hyp = df_scale_vector_rbf.copy()
    # df_hyp = df_hyp.head(10000)

    # For regression, do not create oxy_class
    print(f"Oxygen value stats: min={df_hyp['oxyg'].min()}, max={df_hyp['oxyg'].max()}, mean={df_hyp['oxyg'].mean()}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # preprocess all the columns
    df_time = df_hyp[['depth','ocean_date_time', 'oxyg']].copy()
    df_feat = preprocess(df_hyp)
    df_proc = pd.concat([df_time, df_feat], axis=1)
    # For regression, do not attach oxy_class
    print(f"Processed DataFrame shape: {df_proc.columns}, {df_proc.shape}")
    
    # now group by depth and build your sliding windows exactly as before,
    # but use df_proc instead of the raw df
    # X_train, y_train, X_test, y_test = prepare_dataset_2(
    #     df_proc.rename_axis('ocean_date').reset_index(), 
    #     predictor="oxyg", lookback=7
    # )

    X_train, y_train, X_test, y_test = prepare_dataset_regression(
        df_proc.rename_axis('ocean_date').reset_index(), 
        predictor="oxyg", lookback=30
    )

    
    # For regression, use standard DataLoader
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
    
    # # predictor = 'oxyg'
    # predictor = 'oxy_class'

    # X_train, y_train, X_test, y_test, train_latlon, test_latlon = prepare_dataset(df_1, predictor="oxy_class", lookback=7)
    # print(f"Train shape: {X_train.shape}, {y_train.shape}")
    # print(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Apply SMOTE
    # X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    # X_train_smote, y_train_smote = X_train_smote.to(device), y_train_smote.to(device)
    # print(f"SMOTE applied: {X_train_smote.shape}, {y_train_smote.shape}")
    # Number of classes & counts for each class in y_train_smote: {len(torch.unique(y_train_smote))}")
    # print(f"Number of classes and counts for each class in y_train_smote: {torch.unique(y_train_smote, return_counts=True)}")

   
    input_dim = X_train.shape[2]
    # CHOOSE MODEL TYPE: 'lstm', 'stt', or 'medformer'
    model_type = 'lstm'
    model = get_model(model_type, input_dim)
    model.to(device)

    # Regression loss
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training with {len(loader)} batches...")
    train_model(model, loss_fn, optimizer, loader, device, n_epochs=50)
    print(f"Finsihed training...")

    results, (y_true, y_pred) = evaluate(model, X_test, y_test, device)
    print('Results:', results)

    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

    y_true_hyp = (y_true <= 2.0).astype(int)
    y_pred_hyp = (y_pred <= 2.0).astype(int)

    print(confusion_matrix(y_true_hyp, y_pred_hyp))
    print(classification_report(y_true_hyp, y_pred_hyp))
    print("ROC AUC:", roc_auc_score(y_true_hyp, y_pred))

    import matplotlib.pyplot as plt
    import numpy as np

    residuals = y_true - y_pred

    plt.figure(figsize=(10,4))
    plt.scatter(y_pred, residuals, alpha=0.2)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Oxygen Value")
    plt.ylabel("Residual (True - Predicted)")
    plt.title("Residuals vs Predicted Oxygen")
    plt.savefig('regression_residuals.png')

    df_eval = pd.DataFrame({
        "depth": X_test[:, 0, 3],  # depth of only test values
        "y_true": y_true,
        "y_pred": y_pred
    })

    depth_metrics = df_eval.groupby('depth').apply(
        lambda g: pd.Series({
            'MAE': np.mean(np.abs(g['y_true'] - g['y_pred'])),
            'RMSE': np.sqrt(np.mean((g['y_true'] - g['y_pred'])**2))
        })
    ).reset_index()

    # Optionally plot RMSE/MAE by depth
    plt.figure(figsize=(10,4))
    plt.plot(depth_metrics['depth'], depth_metrics['RMSE'])
    plt.xlabel('Depth')
    plt.ylabel('RMSE')
    plt.title('RMSE by Depth')
    plt.savefig('regression_rmse_by_depth.png')

if __name__ == "__main__":
    main()

