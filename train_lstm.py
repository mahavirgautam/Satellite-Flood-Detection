import os
import numpy as np
import pandas as pd
from detection_app.utils.deep_learning import RainfallLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ================= PATHS =================
DATA_PATH = "dataset/lstm_training/rainfall_timeseries.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_rainfall.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# ================= FEATURES & TARGET =================
FEATURE_COLS = [
    "vv_mean", "vv_std",
    "vh_mean", "vh_std",
    "ratio_mean",
    "ndvi_mean",
    "flood_pct"
]

TARGET_COL = "rainfall_mm"

X_raw = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values.reshape(-1, 1)

# ================= NORMALIZATION =================
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# ================= CREATE SEQUENCES =================
SEQUENCE_LENGTH = 10

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

print("Sequence X shape:", X_seq.shape)
print("Sequence y shape:", y_seq.shape)

# ================= TRAIN / VALIDATION SPLIT =================
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

# ================= MODEL =================
lstm = RainfallLSTM(
    sequence_length=SEQUENCE_LENGTH,
    n_features=X_seq.shape[2]
)

# ================= TRAIN =================
history = lstm.train(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)

# ================= SAVE MODEL =================
lstm.model.save(MODEL_PATH)
print("✅ LSTM model saved at:", MODEL_PATH)

# ================= SAVE SCALERS =================
import joblib
joblib.dump(x_scaler, os.path.join(MODEL_DIR, "lstm_x_scaler.save"))
joblib.dump(y_scaler, os.path.join(MODEL_DIR, "lstm_y_scaler.save"))

print("✅ Scalers saved")
