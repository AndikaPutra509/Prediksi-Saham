# Prediksi Saham
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, Attention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SYMBOL = "BBCA.JK"
START = "2018-01-01"
END = "2026-02-24"
WINDOW = 60
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
HOLD_BAND = 0.05

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ensure_1d(arr) -> np.ndarray:
    return np.asarray(arr).reshape(-1)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = ensure_1d(df["Close"])
    high = ensure_1d(df["High"])
    low = ensure_1d(df["Low"])
    volume = ensure_1d(df["Volume"])
    out = df.copy()
    out["RSI"] = ta.momentum.RSIIndicator(pd.Series(close, index=df.index)).rsi()
    out["MA20"] = pd.Series(close, index=df.index).rolling(20).mean()
    out["MA50"] = pd.Series(close, index=df.index).rolling(50).mean()
    out["MACD"] = ta.trend.MACD(pd.Series(close, index=df.index)).macd()
    bb = ta.volatility.BollingerBands(pd.Series(close, index=df.index))
    out["BB_high"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["ATR"] = ta.volatility.AverageTrueRange(
        pd.Series(high, index=df.index),
        pd.Series(low, index=df.index),
        pd.Series(close, index=df.index),
    ).average_true_range()
    out["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        pd.Series(close, index=df.index),
        pd.Series(volume, index=df.index),
    ).on_balance_volume()
    out["Return"] = pd.Series(close, index=df.index).pct_change()
    out["Target"] = (out["Return"].shift(-1) > 0).astype(int)
    return out.dropna().copy()

def make_sequences(scaled_array: np.ndarray, targets: np.ndarray, window: int):
    X, y, idx = [], [], []
    for i in range(window, len(scaled_array)):
        X.append(scaled_array[i - window : i])
        y.append(targets[i])
        idx.append(i)
    return np.array(X), np.array(y), np.array(idx)

def get_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.arange(0.30, 0.71, 0.01)
    best_t, best_bacc, best_f1 = 0.5, -1.0, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        bacc = balanced_accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        if (bacc > best_bacc) or (np.isclose(bacc, best_bacc) and f1 > best_f1):
            best_t, best_bacc, best_f1 = float(t), float(bacc), float(f1)
    return best_t

def decide_signal(prob_up: float, threshold: float, hold_band: float = HOLD_BAND) -> str:
    if prob_up >= threshold + hold_band:
        return "BELI"
    if prob_up <= threshold - hold_band:
        return "JUAL"
    return "TAHAN"

def main():
    raw = yf.download(SYMBOL, start=START, end=END, auto_adjust=True, progress=True)
    raw = normalize_columns(raw)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = build_features(df) 
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "RSI", "MA20", "MA50", "MACD", "BB_high", "BB_low", "ATR", "OBV",
    ]
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df[features])
    scaled_test = scaler.transform(test_df[features])
    X_train_all, y_train_all, _ = make_sequences(scaled_train, train_df["Target"].to_numpy(), WINDOW)
    X_test, y_test, idx_test = make_sequences(scaled_test, test_df["Target"].to_numpy(), WINDOW)
    val_cut = int(len(X_train_all) * (1 - VAL_RATIO))
    X_train, y_train = X_train_all[:val_cut], y_train_all[:val_cut]
    X_val, y_val = X_train_all[val_cut:], y_train_all[val_cut:]
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Conv1D(filters=64, kernel_size=3, activation="relu")(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    attention = Attention()([x, x])
    x = LSTM(32)(attention)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    class_weights = get_class_weights(y_train)
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=False,
        verbose=1,
    )
    val_probs = model.predict(X_val, verbose=0).ravel()
    best_threshold = find_best_threshold(y_val, val_probs)
    test_loss, test_acc_at_05 = model.evaluate(X_test, y_test, verbose=0)
    probs_up = model.predict(X_test, verbose=0).ravel()
    probs_down = 1 - probs_up
    preds = (probs_up >= best_threshold).astype(int)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy (threshold 0.50): {test_acc_at_05:.4f}")
    print(f"Test accuracy (best threshold {best_threshold:.2f}): {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs_up):.4f}")
    print(f"Best threshold (validation, balanced-accuracy): {best_threshold:.2f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=4, zero_division=0))
    result = pd.DataFrame(
        {
            "Date": test_df.index[idx_test],
            "Close": ensure_1d(test_df["Close"].iloc[idx_test]),
            "Prob_Naik": probs_up,
            "Prob_Turun": probs_down,
        }
    )
    result["Signal"] = result["Prob_Naik"].apply(lambda p: decide_signal(float(p), best_threshold))
    print("\nContoh output (10 hari terakhir):")
    print(result.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
