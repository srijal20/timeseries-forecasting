"""
Multivariate Time-Series Forecasting on the Jena Climate Dataset
==================================================================

Dataset: Jena Climate 2009-2016 (Max Planck Institute for Biogeochemistry)
14 weather variables recorded every 10 minutes.
Source: https://www.bgc-jena.mpg.de/wetter/

This script:
  1. Downloads and preprocesses the dataset
  2. Builds a chronological train/val/test split (no shuffling - critical for time series)
  3. Creates windowed sequences for supervised multi-step forecasting
  4. Trains and compares three models:
       - Naive persistence baseline (last observed value repeated)
       - LSTM
       - A lightweight Transformer encoder
  5. Reports RMSE / MAE per model, per forecast horizon
  6. Saves training/validation loss curves and prediction-vs-actual plots

Run:
    pip install -r requirements.txt
    python forecast.py
"""

import os
import math
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "jena_climate.zip")
CSV_PATH = os.path.join(DATA_DIR, "jena_climate_2009_2016.csv")

LOOKBACK = 144       # 144 * 10min = 24 hours of history
HORIZON = 24         # predict 24 steps ahead = next 4 hours
STRIDE = 1           # step size when generating windows (1 = maximum overlap)
SAMPLING_RATE = 1    # use every point (set >1 to subsample, e.g. hourly)

BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
TARGET_COL = "T (degC)"  # the variable we forecast (temperature)
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# 1. Data download + loading
# ----------------------------------------------------------------------------
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        print("Downloading Jena Climate dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("Download complete.")
    else:
        print("Dataset already present, skipping download.")


def load_dataframe():
    df = pd.read_csv(CSV_PATH)
    # The raw file has a row every 10 minutes; drop duplicate timestamp rows if any
    df = df.drop_duplicates(subset="Date Time")
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Wind direction is an angle (degrees) - encoding it directly is misleading
    # (0 and 360 are the same direction but numerically far apart).
    # Convert wind speed + direction into (wx, wy) vector components instead.
    wv = df.pop("wv (m/s)")
    max_wv = df.pop("max. wv (m/s)")
    wd_rad = df.pop("wd (deg)") * np.pi / 180

    # Some recorded wind speeds are -9999 (sensor error); clip to zero
    wv = wv.clip(lower=0)
    max_wv = max_wv.clip(lower=0)

    df["Wx"] = wv * np.cos(wd_rad)
    df["Wy"] = wv * np.sin(wd_rad)
    df["max Wx"] = max_wv * np.cos(wd_rad)
    df["max Wy"] = max_wv * np.sin(wd_rad)

    # Time-of-day and time-of-year are cyclical - encode as sin/cos so the model
    # sees "midnight" and "just before midnight" as close together, not far apart.
    date_time = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = 365.2425 * day
    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


# ----------------------------------------------------------------------------
# 3. Chronological split + normalization
# ----------------------------------------------------------------------------
def split_and_normalize(df: pd.DataFrame):
    n = len(df)
    train_df = df[: int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) :]

    # Fit normalization stats on TRAIN ONLY to avoid leaking future information
    mean = train_df.mean()
    std = train_df.std()

    train_df = (train_df - mean) / std
    val_df = (val_df - mean) / std
    test_df = (test_df - mean) / std

    return train_df, val_df, test_df, mean, std


# ----------------------------------------------------------------------------
# 4. Windowed dataset
# ----------------------------------------------------------------------------
class WindowedDataset(Dataset):
    """
    Produces (X, y) pairs where:
      X = [lookback, n_features]   -> past LOOKBACK steps, all features
      y = [horizon]                -> future HORIZON steps of TARGET_COL only
    """

    def __init__(self, data: np.ndarray, target_idx: int,
                 lookback: int, horizon: int, stride: int = 1):
        self.data = data.astype(np.float32)
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride

        max_start = len(data) - lookback - horizon
        self.starts = list(range(0, max_start, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        x = self.data[i : i + self.lookback]
        y = self.data[i + self.lookback : i + self.lookback + self.horizon, self.target_idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------------------------------------------------------------
# 5. Models
# ----------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=2,
                             batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: [batch, lookback, n_features]
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]              # [batch, hidden_size]
        return self.head(last_hidden)      # [batch, horizon]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model, horizon, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: [batch, lookback, n_features]
        z = self.input_proj(x)
        z = self.pos_enc(z)
        z = self.encoder(z)
        pooled = z[:, -1, :]               # use last time step's representation
        return self.head(pooled)


# ----------------------------------------------------------------------------
# 6. Training / evaluation
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, name):
    """Trains the model and returns (best model, history).

    history is a list with one dict per epoch:
    {"epoch", "train_mse", "val_mse"} (MSE in normalized units).
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    history = []
    n_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        iterator = tqdm(train_loader, desc=f"[{name}] epoch {epoch}", total=n_batches) \
            if HAS_TQDM else train_loader
        for batch_idx, (x, y) in enumerate(iterator):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

            if not HAS_TQDM and batch_idx % 50 == 0:
                elapsed = time.time() - epoch_start
                print(f"  [{name}] epoch {epoch} batch {batch_idx}/{n_batches} "
                      f"({elapsed:.0f}s elapsed, batch_loss={loss.item():.4f})")
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{name}] epoch {epoch:02d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_{name}.pt")

    print(f"[{name}] best epoch: {best_epoch} (val_mse={best_val:.4f})")
    model.load_state_dict(torch.load(f"best_{name}.pt"))
    return model, history


def save_history(history, name, std_target):
    """Saves the per-epoch losses to results/loss_history_<name>.csv.

    Adds RMSE in degC next to the normalized MSE, so the numbers can be
    compared with the test RMSE: rmse_degC = sqrt(mse) * std of temperature.
    """
    df = pd.DataFrame(history)
    df["train_rmse_degC"] = np.sqrt(df["train_mse"]) * std_target
    df["val_rmse_degC"] = np.sqrt(df["val_mse"]) * std_target
    df["is_best"] = df["val_mse"] == df["val_mse"].min()
    path = os.path.join(RESULTS_DIR, f"loss_history_{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def evaluate(model, loader, std_target, mean_target, name):
    """Returns RMSE and MAE in ORIGINAL units (degC), per forecast step."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_true.append(y.numpy())
    preds = np.concatenate(all_preds, axis=0) * std_target + mean_target
    trues = np.concatenate(all_true, axis=0) * std_target + mean_target

    rmse_per_step = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
    mae_per_step = np.mean(np.abs(preds - trues), axis=0)

    print(f"\n[{name}] Overall RMSE: {rmse_per_step.mean():.3f} degC | "
          f"Overall MAE: {mae_per_step.mean():.3f} degC")
    return preds, trues, rmse_per_step, mae_per_step


def naive_persistence_eval(test_windowed: WindowedDataset, std_target, mean_target):
    """Baseline: predict that the future equals the last observed value."""
    all_preds, all_true = [], []
    target_idx = test_windowed.target_idx
    for i in range(len(test_windowed)):
        x, y = test_windowed[i]
        last_val = x[-1, target_idx].item()
        pred = np.full(test_windowed.horizon, last_val)
        all_preds.append(pred)
        all_true.append(y.numpy())
    preds = np.array(all_preds) * std_target + mean_target
    trues = np.array(all_true) * std_target + mean_target
    rmse_per_step = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
    mae_per_step = np.mean(np.abs(preds - trues), axis=0)
    print(f"\n[naive_persistence] Overall RMSE: {rmse_per_step.mean():.3f} degC | "
          f"Overall MAE: {mae_per_step.mean():.3f} degC")
    return preds, trues, rmse_per_step, mae_per_step


# ----------------------------------------------------------------------------
# 7. Plotting
# ----------------------------------------------------------------------------
def plot_horizon_error(results: dict, out_path="rmse_vs_horizon.png"):
    plt.figure(figsize=(8, 5))
    for name, rmse in results.items():
        plt.plot(range(1, len(rmse) + 1), rmse, marker="o", label=name)
    plt.xlabel("Forecast step (x10 min into the future)")
    plt.ylabel("RMSE (degC)")
    plt.title("Forecast error vs. horizon")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_loss_curves(histories: dict, out_path="loss_curves.png"):
    """One panel per model: training and validation MSE per epoch.

    The panels share the y-axis so the two models can be compared directly,
    and the epoch with the lowest validation loss (the one kept) is marked.
    """
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4.5),
                             sharey=True, squeeze=False)
    for ax, (name, history) in zip(axes[0], histories.items()):
        epochs = [h["epoch"] for h in history]
        train = [h["train_mse"] for h in history]
        val = [h["val_mse"] for h in history]
        best = epochs[int(np.argmin(val))]

        ax.plot(epochs, train, color="#1f77b4", linestyle="-", marker="o", label="Training")
        ax.plot(epochs, val, color="#ff7f0e", linestyle="--", marker="s", label="Validation")
        ax.axvline(best, color="gray", linestyle=":", linewidth=1)
        ax.annotate(f"best epoch {best}", xy=(best, min(val)), xytext=(6, 14),
                    textcoords="offset points", fontsize=9, color="dimgray")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0][0].set_ylabel("MSE (normalized units)")
    fig.suptitle("Training and validation loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sample_forecast(preds, trues, sample_idx=0, out_path="sample_forecast.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(trues[sample_idx], label="Actual", marker="o")
    plt.plot(preds[sample_idx], label="Predicted", marker="x")
    plt.xlabel("Steps ahead")
    plt.ylabel("Temperature (degC)")
    plt.title("Sample forecast vs actual")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    download_data()
    df = load_dataframe()
    df = engineer_features(df)

    train_df, val_df, test_df, mean, std = split_and_normalize(df)
    feature_cols = list(train_df.columns)
    target_idx = feature_cols.index(TARGET_COL)
    mean_target, std_target = mean[TARGET_COL], std[TARGET_COL]

    train_ds = WindowedDataset(train_df.values, target_idx, LOOKBACK, HORIZON, STRIDE)
    val_ds = WindowedDataset(val_df.values, target_idx, LOOKBACK, HORIZON, STRIDE)
    test_ds = WindowedDataset(test_df.values, target_idx, LOOKBACK, HORIZON, STRIDE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = len(feature_cols)
    print(f"\nFeatures: {n_features} | Lookback: {LOOKBACK} | Horizon: {HORIZON}")
    print(f"Train windows: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    results_rmse = {}

    # --- Baseline ---
    _, _, rmse_naive, _ = naive_persistence_eval(test_ds, std_target, mean_target)
    results_rmse["Naive persistence"] = rmse_naive

    # --- LSTM ---
    lstm = LSTMForecaster(n_features, HIDDEN_SIZE, HORIZON)
    lstm, hist_lstm = train_model(lstm, train_loader, val_loader, EPOCHS, LEARNING_RATE, "lstm")
    save_history(hist_lstm, "lstm", std_target)
    preds_lstm, trues_lstm, rmse_lstm, _ = evaluate(lstm, test_loader, std_target, mean_target, "LSTM")
    results_rmse["LSTM"] = rmse_lstm

    # --- Transformer ---
    transformer = TransformerForecaster(n_features, d_model=HIDDEN_SIZE, horizon=HORIZON)
    transformer, hist_tf = train_model(transformer, train_loader, val_loader, EPOCHS, LEARNING_RATE, "transformer")
    save_history(hist_tf, "transformer", std_target)
    preds_tf, trues_tf, rmse_tf, _ = evaluate(transformer, test_loader, std_target, mean_target, "Transformer")
    results_rmse["Transformer"] = rmse_tf

    # --- Plots ---
    plot_loss_curves({"LSTM": hist_lstm, "Transformer": hist_tf},
                     out_path=os.path.join(RESULTS_DIR, "loss_curves.png"))
    plot_horizon_error(results_rmse, out_path=os.path.join(RESULTS_DIR, "rmse_vs_horizon.png"))
    plot_sample_forecast(preds_lstm, trues_lstm, sample_idx=100, out_path=os.path.join(RESULTS_DIR, "sample_forecast_lstm.png"))
    plot_sample_forecast(preds_tf, trues_tf, sample_idx=100, out_path=os.path.join(RESULTS_DIR, "sample_forecast_transformer.png"))

    print("\nDone. Check the saved .png files for results.")


if __name__ == "__main__":
    main()
