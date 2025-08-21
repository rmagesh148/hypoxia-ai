import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as md
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE


from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()
mm = MinMaxScaler()

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

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

    X_res_tensor = th.tensor(X_res, dtype=th.float32).view(-1, X_tensor.shape[1], X_tensor.shape[2])
    y_res_tensor = th.tensor(y_res, dtype=th.float32)

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
    return th.tensor(np.array(X1), dtype=th.float32), th.tensor(np.array(y1), dtype=th.float32)

def prepare_dataset(df, predictor="oxy_class", lookback=7, test_year=2020, test_month=8, month_list=[5,6,7,8]):
    train_list_x, train_list_y = [], []
    test_list_x, test_list_y = [], []
    train_list_lat_lon, test_list_lat_lon = [], []

    for value in tqdm(df.depth.unique()):
        df_depth = df[(df['depth'] == value) & (df['ocean_date_time'].dt.month.isin(month_list))]
        df_test = df_depth[(df_depth['ocean_date_time'].dt.year == test_year) &
                           (df_depth['ocean_date_time'].dt.month == test_month)]
        df_train = df_depth[~df_depth.index.isin(df_test.index)]

        train_x_raw = df_train[['SOCalt', 'PEA', 'DCPtemp', 'depth']]
        test_x_raw = df_test[['SOCalt', 'PEA', 'DCPtemp', 'depth']]
        train_y_raw = df_train[[predictor]]
        test_y_raw = df_test[[predictor]]

        X_train_scaled = mm.fit_transform(train_x_raw)
        X_test_scaled = mm.transform(test_x_raw)

        if predictor != "oxy_class":
            y_train_scaled = mm.fit_transform(train_y_raw)
            y_test_scaled = mm.transform(test_y_raw)
        else:
            y_train_scaled = train_y_raw.values
            y_test_scaled = test_y_raw.values

        train_x, train_y = create_dataset(X_train_scaled, y_train_scaled, lookback)
        test_x, test_y = create_dataset(X_test_scaled, y_test_scaled, lookback)

        df_train_lat_lon = df_train[['lat_rho', 'lon_rho']].iloc[:-lookback]
        df_test_lat_lon = df_test[['lat_rho', 'lon_rho']].iloc[:-lookback]

        train_list_x.append(train_x)
        train_list_y.append(train_y)
        test_list_x.append(test_x)
        test_list_y.append(test_y)
        train_list_lat_lon.append(df_train_lat_lon)
        test_list_lat_lon.append(df_test_lat_lon)

    # Stack tensors
    X_train = th.vstack(train_list_x)
    y_train = th.vstack(train_list_y).squeeze(-1)
    X_test = th.vstack(test_list_x)
    y_test = th.vstack(test_list_y).squeeze(-1)

    # Lat/lon as DataFrames
    df_train_lat_lon = pd.concat(train_list_lat_lon, ignore_index=True)
    df_test_lat_lon = pd.concat(test_list_lat_lon, ignore_index=True)

    return X_train, y_train, X_test, y_test, df_train_lat_lon, df_test_lat_lon
    

ds = xr.open_dataset('PEA_SOCalt_DCPtemp_hypxia_oxygen_depth_3D_2018_2020_exp14_2.nc', decode_times=False)
df = ds.to_dataframe()

#df = df[(df.index > pd.to_datetime('2013-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
#df = df[(df.index > pd.to_datetime('2018-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
#df_test = df[(df.index > pd.to_datetime('2019-01-01')) & (df.index < pd.to_datetime('2019-12-31'))]

df.reset_index(inplace=True)

df = df.dropna(axis=0)
df.head()

df = df[(df['SOCalt'] > 0.0) & (df['PEA'] > 0.0) & (df['DCPtemp'] > 0.0)]
df.shape
print("Shape of the dataframe: ", df.shape[0])

df['ocean_time'] = df['ocean_time'].astype(int) - 719529
df['ocean_date_time'] = df['ocean_time'].apply(lambda x: md.num2date(x) if pd.notnull(x) else pd.NaT)
df['ocean_time'] = df['ocean_time']

df['ocean_date'] = pd.to_datetime(df['ocean_date_time']).dt.date
df = df.set_index('ocean_date')

df = df.sort_index()
df.head()

#df['day_of_year'] = df.ocean_date_time.dt.day_of_year

#df_scale_vector = df[['SOCalt', 'PEA', 'DCPtemp', 'depth', 'D Sin', 'D Cos', 'Y Sin', 'Y Cos', 'M Sin', 'M Cos', 'oxyg']]
#df_scale_vector_rbf = df[['SOCalt', 'PEA', 'DCPtemp', 'depth']]
df_scale_vector_rbf = df[['lat_rho', 'lon_rho', 'ocean_date_time', 'ocean_time', 'SOCalt', 'PEA', 'DCPtemp', 'depth', 'oxyg']]
#df_y = df[['oxyg']]

df_hyp = df_scale_vector_rbf.copy()

# 1 if Hypoxia and Non - Hypoxia 0
df_hyp['oxy_class'] = np.where(df_hyp['oxyg'] > 2.0, 0, 1)

df_1 = df_hyp.groupby('depth', group_keys=True).apply(lambda x: x)
# predictor = 'oxyg'
predictor = 'oxy_class'

X_train, y_train, X_test, y_test, train_latlon, test_latlon = prepare_dataset(df_1, predictor="oxy_class", lookback=7)

# Apply SMOTE
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

#num_classes = 1
# input_dim = 4
input_dim = 4
hidden_dim = 120

# change this to 1 or 2
output_dim = 2

# Layder dim =2
layer_dim = 2

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Running on:", device)

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim, dropout=0.3, bidirectional=True)
# model = LSTMClassfier(input_dim, hidden_dim, layer_dim, output_dim, X_train.shape[2])
# # model = LSTMReg(input_dim, hidden_dim, layer_dim, output_dim, X_train.shape[2])

# model = LSTMBinaryClassifier(input_dim, hidden_dim, layer_dim)
model = model.to(device)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

loader = data.DataLoader(data.TensorDataset(X_train_smote, y_train_smote), shuffle=False, batch_size=2048)

# Driver for Classification
X_test = X_test.to(th.float32)
n_epochs = 10
for epoch in tqdm(range(n_epochs)):
    model.train()
    for X_batch, y_batch in (loader):
        # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        # y_batch = y_batch.float().view(-1)
        y_batch = y_batch.flatten().long()
        # print(y_pred.squeeze().shape, y_batch.shape)
        loss = loss_fn(y_pred.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with th.no_grad():
    y_pred_test = model(X_test)
    predicted_labels = th.argmax(y_pred_test, dim=1)
    y_true = y_test.numpy().flatten()
    y_scores = predicted_labels.numpy().flatten()


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

# Assuming y_pred_test contains raw logits
y_prob = th.softmax(y_pred_test, dim=1)[:, 1].cpu().numpy()  # predicted probability for class 1
y_pred = predicted_labels.cpu().numpy()
y_true = y_test.cpu().numpy().flatten()  # ensure compatibility

# Metrics
print("Evaluation Metrics")
print(f"Log Loss: {log_loss(y_true, y_prob):.4f}")
print(f"Brier Score Loss: {brier_score_loss(y_true, y_prob):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")
print(f"AUC-PR (average_precision_score): {average_precision_score(y_true, y_prob):.4f}")

# Precision-Recall AUC manually (for plotting)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
aupr = auc(recall, precision)
print(f"AUC-PR (manual): {aupr:.4f}")

# McNemar's Test
cm = confusion_matrix(y_true, y_pred)
# if cm.shape == (2, 2):
#     result = mcnemar(cm, exact=False, correction=True)
#     print(f"McNemar’s Test p-value: {result.pvalue:.4f}")
# else:
#     print("McNemar's Test requires binary classification with a 2x2 confusion matrix.")

# Plot ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AUPR = {aupr:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("pr-auc.png")
plt.legend()

plt.tight_layout()