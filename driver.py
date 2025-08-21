import os
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
from models import LSTMClassifier, STT, Medformer, FocalLoss, TCNClassifier

from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
import seaborn as sns


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
    print(f"df shape: {df.head()}")
    # df = encode_depth(df)
    
    # drop columns we don’t want to scale / feed directly
    drop_cols = ['ocean_date_time', 'ocean_time', 'oxyg']
    df_model = df.drop(columns=drop_cols)
    
    # split features into two scaler groups
    latlon_cols = ['lat_rho', 'lon_rho']
    phys_cols   = ['SOCalt', 'PEA', 'DCPtemp', 'depth']
    time_cols   = ['doy_sin','doy_cos','mon_sin','mon_cos','hour_sin','hour_cos']
    # depth_ohe_cols = [c for c in df_model.columns if c.startswith('depth_')]
    
    # fit scalers on full train set (you can refactor to do per‐depth if desired)
    # scaler_ll = StandardScaler()
    # scaler_phys = MinMaxScaler(feature_range=(0,1))
    
    # df_model[latlon_cols] = scaler_ll.fit_transform(df_model[latlon_cols])
    # df_model[phys_cols]   = scaler_phys.fit_transform(df_model[phys_cols])
    
    # everything else (time and depth_ohe) is already ~[-1,1] or {0,1}
    # final_cols = latlon_cols + phys_cols + time_cols + depth_ohe_cols
    final_cols = latlon_cols + phys_cols + time_cols
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


def get_model(model_type, input_dim, num_classes):
    if model_type == 'lstm':
        return LSTMClassifier(input_dim, 120, 2, num_classes, dropout=0.3, bidirectional=True)
    elif model_type == 'stt':
        return STT(input_dim, num_classes)
    elif model_type == 'medformer':
        return Medformer(input_dim, num_classes)
    elif model_type == 'tcn':
        # e.g. three layers of 128 channels each
        return TCNClassifier(input_dim, num_classes, num_channels=[128,128,128], kernel_size=3, dropout=0.1)
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

def create_dataset(X, y, lookback, mode):
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

# t1 # feature [1, 1, 0 , 3]
# t2 # feature [1, 1, 0 , 3]
# t3 # feature [1, 1, 0 , 3] & y = 0

# lookback = 3
# feature = [1, 1, 0, 3], [1, 1,0,3], [1, 1, 0, 3] & y = 0

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
    predictor="oxy_class",
    lookback=30,
    test_year=2020,
    test_month=8,
    month_list=[5,6,7,8]
):
    if 'depth' not in df.columns:
        df = df.reset_index()
    
    # 0) precompute feature columns (exclude only the things you don't want in X)
    exclude = {predictor, 'ocean_date_time', 'ocean_date'}
    feature_cols = [c for c in df.columns if c not in exclude]

    train_x_list, train_y_list = [], []
    test_x_list,  test_y_list  = [], []

    # 1) iterate by depth
    for depth_val in tqdm(df['depth'].unique()):
        # keep only the months we care about
        df_depth = df[
            (df['depth'] == depth_val) &
            (df['ocean_date_time'].dt.month.isin(month_list))
        ]

        # 2) split train vs test by year/month
        is_test = (
            (df_depth['ocean_date_time'].dt.year == test_year) &
            (df_depth['ocean_date_time'].dt.month == test_month)
        )
        df_train = df_depth[~is_test]
        df_test = df_depth[is_test]

        # 3) pull out raw numpy arrays
        X_tr_raw = df_train[feature_cols].to_numpy(dtype=float)
        y_tr_raw = df_train[[predictor]].to_numpy(dtype=float)
        X_te_raw = df_test[feature_cols].to_numpy(dtype=float)
        y_te_raw = df_test[[predictor]].to_numpy(dtype=float)

        scaler = MinMaxScaler()
        X_tr_raw = scaler.fit_transform(X_tr_raw)
        X_te_raw = scaler.transform(X_te_raw)
        # y_tr_raw = scaler.fit_transform(y_tr_raw)
        # y_te_raw = scaler.transform(y_te_raw)

        # 4) build sliding windows on that group
        X_tr_win, y_tr_win = create_dataset(X_tr_raw, y_tr_raw, lookback, "train")
        X_te_win, y_te_win = create_dataset(X_te_raw, y_te_raw, lookback, "test")
        

        # 5) collect
        if len(X_tr_win) > 0:
            train_x_list.append(X_tr_win)
            train_y_list.append(y_tr_win)
        if len(X_te_win) > 0:
            test_x_list.append(X_te_win)
            test_y_list.append(y_te_win)
    # 6) finally, stack *all* depths together
    if not train_x_list or not test_x_list:
        print("No valid train or test data was found after grouping by depth. Check your data and parameters.")
        return None, None, None, None
    X_train = torch.vstack(train_x_list)
    y_train = torch.vstack(train_y_list).squeeze(-1)
    X_test = torch.vstack(test_x_list)
    y_test = torch.vstack(test_y_list).squeeze(-1)

    return X_train, y_train, X_test, y_test

def compute_class_weight(y_train):
    bincount = np.bincount(y_train.astype('int'))
    weight = 1. / np.clip(bincount, 1, None)
    return torch.tensor(weight, dtype=torch.float32)

def train_model(model, loss_fn, optimizer, loader, device, n_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.flatten().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(loader):.5f}")

def evaluate_and_save(model, X_test, y_test, device, save_prefix="model", out_root="results"):
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        y_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y_true = y_test.cpu().numpy().flatten()

    # Default predictions
    y_preds_default = (y_probs >= 0.5).astype(int)
    accuracy = (y_preds_default == y_true).mean()

    # Metrics
    auroc = roc_auc_score(y_true, y_probs)
    aupr  = average_precision_score(y_true, y_probs)
    lloss = log_loss(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    cm_default = confusion_matrix(y_true, y_preds_default)

    # Precision–Recall + Best F1 threshold
    prec, rec, thresh = precision_recall_curve(y_true, y_probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1_valid = f1[:-1]                        # ignore last entry
    best_i = np.argmax(f1_valid)
    best_thresh = thresh[best_i]
    best_f1 = f1_valid[best_i]

    # Predictions at best threshold
    y_preds_best = (y_probs >= best_thresh).astype(int)
    cm_best = confusion_matrix(y_true, y_preds_best)

    # ROC
    fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
    idx_default = np.argmin(np.abs(roc_thresh - 0.5))
    fpr_default, tpr_default = fpr[idx_default], tpr[idx_default]
    idx_best = np.argmin(np.abs(roc_thresh - best_thresh))
    fpr_best, tpr_best = fpr[idx_best], tpr[idx_best]

    # PR markers
    idx_pr_default = np.argmin(np.abs(thresh - 0.5))
    prec_default, rec_default = prec[idx_pr_default], rec[idx_pr_default]
    prec_best, rec_best = prec[best_i], rec[best_i]

    print(f"[{save_prefix}] Best F1 threshold: {best_thresh:.4f}  F1: {best_f1:.4f}")

    # Create output directory
    out_dir = os.path.join(out_root, save_prefix)
    os.makedirs(out_dir, exist_ok=True)

    # --- Plot 1: ROC ---
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f} | Accuracy = {accuracy:.4f}")
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.scatter(fpr_default, tpr_default, s=40, label="thr=0.50")
    plt.scatter(fpr_best, tpr_best, marker='*', s=140, label=f"best thr={best_thresh:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{save_prefix}_roc_curve.png', dpi=150)
    plt.close()

    # --- Plot 2: Precision–Recall ---
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AUPR = {aupr:.4f}")
    plt.scatter(rec_default, prec_default, s=40, label="thr=0.50")
    plt.scatter(rec_best, prec_best, marker='*', s=140,
                label=f"best thr={best_thresh:.2f}\nF1={best_f1:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{save_prefix}_pr_curve.png', dpi=150)
    plt.close()

    # --- Plot 3: Calibration metrics (bar) ---
    plt.figure(figsize=(4,5))
    plt.bar(["LogLoss","Brier"], [lloss, brier], color=["tab:blue","tab:orange"])
    for i,val in enumerate([lloss,brier]):
        plt.text(i, val + 0.001, f"{val:.3f}", ha="center", va="bottom")
    plt.title("Calibration Metrics")
    plt.ylim(0, max(lloss, brier)*1.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{save_prefix}_calibration_metrics.png', dpi=150)
    plt.close()

    # --- Plot 4: Confusion matrices (two separate PNGs) ---
    # Default threshold
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (thr=0.50)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{save_prefix}_confusion_matrix_default.png', dpi=150)
    plt.close()

    # Best threshold
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f"Confusion Matrix (best thr={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{save_prefix}_confusion_matrix_best.png', dpi=150)
    plt.close()

    # Collect numeric results
    results = {
        'Log Loss': lloss,
        'Brier Score': brier,
        'AUC-ROC': auroc,
        'AUC-PR': aupr,
        'Best F1 Threshold': best_thresh,
        'Best F1': best_f1,
        'Confusion Matrix (thr=0.5)': cm_default,
        'Confusion Matrix (best thr)': cm_best
    }
    return results, y_probs

def main():
    df_scale_vector_rbf = pd.read_pickle('df_hyp_input.pkl')

    df_hyp = df_scale_vector_rbf.copy()

    # # 1 if Hypoxia and Non - Hypoxia 0
    df_hyp['oxy_class'] = np.where(df_hyp['oxyg'] > 2.0, 0, 1)
    print("Number of hypoxia cases: ", df_hyp['oxy_class'].value_counts())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # preprocess all the columns
    df_time = df_hyp[['ocean_date_time', 'oxyg']].copy()
    df_feat = preprocess(df_hyp)
    df_proc = pd.concat([df_time, df_feat], axis=1)
    df_proc['oxy_class'] = df_hyp['oxy_class']  # reattach target

    print(f"Processed DataFrame shape: {df_proc.columns}, {df_proc.shape}")
    
    # now group by depth and build your sliding windows exactly as before,
    # but use df_proc instead of the raw df
    X_train, y_train, X_test, y_test, *_ = prepare_dataset_2(
        df_proc.rename_axis('ocean_date').reset_index(), 
        predictor="oxy_class", lookback=7
    )

    if X_train is None or X_test is None or X_train.shape[1] == 0 or X_test.shape[1] == 0:
        print("No valid data available for training or testing. Exiting.")
        return
    
    # # # make the balanced loader
    loader = make_dataloader(X_train, y_train, batch_size=1024)
    
    # # # predictor = 'oxyg'
    # predictor = 'oxy_class'

    # # X_train, y_train, X_test, y_test, train_latlon, test_latlon = prepare_dataset(df_1, predictor="oxy_class", lookback=7)
    # # print(f"Train shape: {X_train.shape}, {y_train.shape}")
    # # print(f"Test shape: {X_test.shape}, {y_test.shape}")

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_train, y_train = X_train_smote.to(device), y_train_smote.to(device)
    print(f"SMOTE applied: {X_train_smote.shape}, {y_train_smote.shape}")
    print(f"Number of classes & counts for each class in y_train_smote: {len(torch.unique(y_train))}")
    print(f"Number of classes and counts for each class in y_train_smote: {torch.unique(y_train, return_counts=True)}")

   
    # input_dim = X_train_smote.shape[2]
    input_dim = X_train.shape[2]
    output_dim = 2
    

    # CHOOSE MODEL TYPE: 'lstm', 'stt', or 'medformer'
    model_list = ['lstm', 'stt', 'medformer', 'tcn']
    # model_list = ['tcn']
    for i in model_list:
        model_type = i
        model = get_model(model_type, input_dim, output_dim)
        model.to(device)

        # # Class weights for loss
        # weights = compute_class_weight(y_train.cpu().numpy())
        loss_fn = FocalLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f" X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f" X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # # loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=2048)
        
        print(f"Starting training with {len(loader)} batches...")
        
        train_model(model, loss_fn, optimizer, loader, device, n_epochs=20)

        print(f"Finsihed training for {model_type}...")

        results, _ = evaluate_and_save(model, X_test, y_test, device, save_prefix=model_type, out_root="results")
        
        print('Results:', results)

        # save the model
        torch.save(model.state_dict(), f"{model_type}_model.pth")

if __name__ == "__main__":
    main()

