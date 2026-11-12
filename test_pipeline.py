"""
Sanity test: runs the full pipeline (windowing, models, training loop, eval, plots)
on small SYNTHETIC data with the same shape as the real Jena Climate dataset,
to catch bugs before running on the real ~420k-row dataset (which needs internet
access to download).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import forecast as fc

# Shrink settings for a fast smoke test
fc.LOOKBACK = 24
fc.HORIZON = 6
fc.BATCH_SIZE = 16
fc.EPOCHS = 2
fc.HIDDEN_SIZE = 8

# --- Build synthetic dataframe mimicking Jena Climate's raw columns ---
n = 2000
rng = np.random.default_rng(0)
dates = pd.date_range("2020-01-01", periods=n, freq="10min")
date_strs = dates.strftime("%d.%m.%Y %H:%M:%S")

df = pd.DataFrame({
    "Date Time": date_strs,
    "p (mbar)": rng.normal(1000, 5, n),
    "T (degC)": 10 + 10 * np.sin(np.linspace(0, 40 * np.pi, n)) + rng.normal(0, 0.5, n),
    "Tpot (K)": rng.normal(283, 5, n),
    "Tdew (degC)": rng.normal(5, 3, n),
    "rh (%)": rng.uniform(30, 90, n),
    "VPmax (mbar)": rng.uniform(5, 20, n),
    "VPact (mbar)": rng.uniform(3, 15, n),
    "VPdef (mbar)": rng.uniform(0, 5, n),
    "sh (g/kg)": rng.uniform(2, 10, n),
    "H2OC (mmol/mol)": rng.uniform(3, 15, n),
    "rho (g/m**3)": rng.normal(1200, 20, n),
    "wv (m/s)": rng.uniform(0, 5, n),
    "max. wv (m/s)": rng.uniform(0, 8, n),
    "wd (deg)": rng.uniform(0, 360, n),
})

print("Synthetic dataframe shape:", df.shape)

# --- Run through the real pipeline functions ---
df_eng = fc.engineer_features(df)
print("After feature engineering, columns:", list(df_eng.columns))

train_df, val_df, test_df, mean, std = fc.split_and_normalize(df_eng)
print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

feature_cols = list(train_df.columns)
target_idx = feature_cols.index(fc.TARGET_COL)
mean_target, std_target = mean[fc.TARGET_COL], std[fc.TARGET_COL]

train_ds = fc.WindowedDataset(train_df.values, target_idx, fc.LOOKBACK, fc.HORIZON, fc.STRIDE)
val_ds = fc.WindowedDataset(val_df.values, target_idx, fc.LOOKBACK, fc.HORIZON, fc.STRIDE)
test_ds = fc.WindowedDataset(test_df.values, target_idx, fc.LOOKBACK, fc.HORIZON, fc.STRIDE)
print("Windowed dataset sizes:", len(train_ds), len(val_ds), len(test_ds))

x0, y0 = train_ds[0]
print("Sample X shape:", x0.shape, "Sample y shape:", y0.shape)
assert x0.shape == (fc.LOOKBACK, len(feature_cols))
assert y0.shape == (fc.HORIZON,)

train_loader = DataLoader(train_ds, batch_size=fc.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=fc.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=fc.BATCH_SIZE, shuffle=False)

n_features = len(feature_cols)

# --- Naive baseline ---
_, _, rmse_naive, _ = fc.naive_persistence_eval(test_ds, std_target, mean_target)
print("Naive RMSE per step:", rmse_naive)

# --- LSTM: forward pass + 2-epoch train ---
lstm = fc.LSTMForecaster(n_features, fc.HIDDEN_SIZE, fc.HORIZON)
out = lstm(x0.unsqueeze(0))
print("LSTM forward output shape:", out.shape)
assert out.shape == (1, fc.HORIZON)
lstm = fc.train_model(lstm, train_loader, val_loader, fc.EPOCHS, fc.LEARNING_RATE, "lstm_test")
preds_lstm, trues_lstm, rmse_lstm, _ = fc.evaluate(lstm, test_loader, std_target, mean_target, "LSTM_test")
print("LSTM RMSE per step:", rmse_lstm)

# --- Transformer: forward pass + 2-epoch train ---
transformer = fc.TransformerForecaster(n_features, d_model=fc.HIDDEN_SIZE, horizon=fc.HORIZON)
out_t = transformer(x0.unsqueeze(0))
print("Transformer forward output shape:", out_t.shape)
assert out_t.shape == (1, fc.HORIZON)
transformer = fc.train_model(transformer, train_loader, val_loader, fc.EPOCHS, fc.LEARNING_RATE, "transformer_test")
preds_tf, trues_tf, rmse_tf, _ = fc.evaluate(transformer, test_loader, std_target, mean_target, "Transformer_test")
print("Transformer RMSE per step:", rmse_tf)

# --- Plots ---
fc.plot_horizon_error({"Naive": rmse_naive, "LSTM": rmse_lstm, "Transformer": rmse_tf},
                       out_path="test_rmse_vs_horizon.png")
fc.plot_sample_forecast(preds_lstm, trues_lstm, sample_idx=5, out_path="test_sample_forecast.png")

print("\nALL SMOKE TESTS PASSED.")
