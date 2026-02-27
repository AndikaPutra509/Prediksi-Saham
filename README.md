!pip install ta
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

SEED = 42
np.random.seed(SEED)

# Pilih mode: "harian" (intraday/hourly) atau "mingguan" (daily swing)
TRADING_MODE = "mingguan"

SYMBOL = "WMUU.JK"
INTERVAL = "1h"
START = "2013-01-01"
END = "2026-02-28"
INTRADAY_PERIOD = "365d"
INTRADAY_FALLBACK_PERIODS = ["365d", "180d", "90d", "60d", "30d", "10d", "5d", "3d"]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
HOLD_BAND = 0.05
TARGET_MODE = "three_class"  # "binary" atau "three_class"
RETURN_THRESHOLD = 0.01
USE_PROB_CALIBRATION = True
MIN_ROWS_AFTER_FEATURES = 40
MIN_ROWS_INTRADAY = 18
MIN_RAW_ROWS_FOR_FEATURES = 20
AS_OF_DATE = None  # contoh: "2026-02-18" atau "2026-02-18 15:00:00"
IHSG_TICKER = "^JKSE"
INDO_VIX_CANDIDATES = ["^JKVIX", "JKVIX", "VIX.JK", "^VIX", "VIX"]
GLOBAL_INDEX_TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOWJONES": "^DJI",
    "NIKKEI": "^N225",
    "SHANGHAI": "000001.SS",
}
COMMODITY_TICKERS = {
    "OIL": "CL=F",
    "GOLD": "GC=F",
}
VIX_SPIKE_THRESHOLD = 0.04  # lonjakan 1-periode (4%) dianggap fear naik
VIX_HIGH_LEVEL = 25.0
USDIDR_TICKER = "IDR=X"
JKT_TZ = "Asia/Jakarta"
SESSION1_START = "09:00:00"
SESSION1_END = "12:00:00"
SESSION2_START = "13:30:00"
SESSION2_END = "16:00:00"
SR_LOOKBACK_DAILY = 20
SR_LOOKBACK_INTRADAY = 8

def resolve_trading_mode(mode: str) -> dict:
    normalized = str(mode).strip().lower()
    if normalized not in {"harian", "mingguan"}:
        raise ValueError('TRADING_MODE harus "harian" atau "mingguan".')
    if normalized == "harian":
        return {
            "name": "harian",
            "interval": "1h",
            "forecast_periods": 24,
            "forecast_label": "Prediksi harga 24 jam ke depan (hourly)",
            "horizon_note": "intraday/day-trading",
        }
    return {
        "name": "mingguan",
        "interval": "1d",
        "forecast_periods": 5,
        "forecast_label": "Prediksi harga 1 minggu ke depan (5 hari bursa)",
        "horizon_note": "weekly swing-trading",
    }
    
def get_runtime_mode_config() -> dict:
    """Mode dikunci ke mingguan: trading mingguan dengan data train harian (1d)."""
    return resolve_trading_mode("mingguan")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out.loc[:, num_cols] = out.loc[:, num_cols].replace([np.inf, -np.inf], np.nan)
    return out

def download_data(symbol: str, interval: str, start: str, end: str, intraday_period: str) -> pd.DataFrame:
    if interval in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        raw = yf.download(symbol, period=intraday_period, interval=interval, auto_adjust=True, progress=True)
    else:
        raw = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=True)
    raw = normalize_columns(raw)
    raw = filter_jakarta_sessions(raw, interval)
    return raw

def apply_as_of_cutoff(df: pd.DataFrame, as_of_date: str | None) -> pd.DataFrame:
    if as_of_date is None:
        return df
    cutoff = pd.to_datetime(as_of_date)
    # Jika user isi tanggal tanpa jam, anggap sampai akhir hari itu
    if len(str(as_of_date)) <= 10:
        cutoff = cutoff + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(idx.tz)
    return df.loc[df.index <= cutoff].copy()

def apply_idx_to_jakarta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out
    if out.index.tz is None:
        out.index = out.index.tz_localize(JKT_TZ)
    else:
        out.index = out.index.tz_convert(JKT_TZ)
    return out

def filter_jakarta_sessions(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Filter data agar hanya jam perdagangan BEI sesi 1 & 2."""
    if interval not in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        return df
    out = apply_idx_to_jakarta(df)
    sess1 = out.between_time(SESSION1_START, SESSION1_END, inclusive="both")
    sess2 = out.between_time(SESSION2_START, SESSION2_END, inclusive="both")
    out = pd.concat([sess1, sess2]).sort_index()
    # buang akhir pekan bila ada
    if isinstance(out.index, pd.DatetimeIndex):
        out = out[out.index.dayofweek < 5]
    return out

def try_download_close_series(symbol: str, interval: str, start: str, end: str, intraday_period: str, as_of_date: str | None) -> pd.Series | None:
    try:
        raw = download_data(symbol, interval, start, end, intraday_period)
        raw = apply_as_of_cutoff(raw, as_of_date)
        if raw.empty or "Close" not in raw.columns:
            return None
        ser = raw["Close"].copy()
        ser.name = symbol
        return ser
    except Exception:
        return None

def add_market_context_features(df: pd.DataFrame, interval: str, start: str, end: str, intraday_period: str, as_of_date: str | None):
    out = df.copy()
    info = {
        "ihsg_used": False,
        "indovix_symbol": None,
        "currency_used": False,
        "global_markets_used": 0,
        "commodities_used": 0,
        "timestamps_aligned": True,
    }
    is_intraday = interval in {"1h", "60m", "30m", "15m", "5m", "1m"}
    ctx_roll = 12 if is_intraday else 20
    ihsg_close = try_download_close_series(IHSG_TICKER, interval, start, end, intraday_period, as_of_date)
    if ihsg_close is not None:
        ihsg_close = ihsg_close.reindex(out.index).ffill().bfill()
        ihsg_ret = ihsg_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ihsg_ma20 = ihsg_close.rolling(ctx_roll).mean().replace([np.inf, -np.inf], np.nan)
        out["IHSG_Return_1"] = ihsg_ret
        out["IHSG_Return_5"] = ihsg_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["IHSG_Volatility_10"] = ihsg_ret.rolling(10).std().fillna(0.0)
        out["IHSG_MA20"] = ihsg_ma20.ffill().bfill()
        out["IHSG_Trend"] = (ihsg_close / ihsg_ma20).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        out["Market_Risk_Regime"] = (out["IHSG_Trend"] >= 1.0).astype(int)
        info["ihsg_used"] = True
    else:
        out["IHSG_Return_1"] = 0.0
        out["IHSG_Return_5"] = 0.0
        out["IHSG_Volatility_10"] = 0.0
        out["IHSG_MA20"] = 0.0
        out["IHSG_Trend"] = 1.0
        out["Market_Risk_Regime"] = 0
    vix_close = None
    for ticker in INDO_VIX_CANDIDATES:
        candidate = try_download_close_series(ticker, interval, start, end, intraday_period, as_of_date)
        if candidate is not None:
            vix_close = candidate
            info["indovix_symbol"] = ticker
            break
    if vix_close is not None:
        vix_close = vix_close.reindex(out.index).ffill().bfill()
        vix_change = vix_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["INDOVIX_Level"] = vix_close.ffill().bfill()
        out["INDOVIX_Change_1"] = vix_change
        out["INDOVIX_Change_5"] = vix_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["VIX_Change"] = out["INDOVIX_Change_1"]
        out["VIX_Spike"] = (out["VIX_Change"] > VIX_SPIKE_THRESHOLD).astype(int)
        out["VIX_Level"] = out["INDOVIX_Level"]
    else:
        out["INDOVIX_Level"] = 0.0
        out["INDOVIX_Change_1"] = 0.0
        out["INDOVIX_Change_5"] = 0.0
        out["VIX_Change"] = 0.0
        out["VIX_Spike"] = 0
        out["VIX_Level"] = 0.0
    # Global market + commodity context
    market_return_cols = []
    for market_name, market_ticker in GLOBAL_INDEX_TICKERS.items():
        market_close = try_download_close_series(market_ticker, interval, start, end, intraday_period, as_of_date)
        prefix = f"GM_{market_name}"
        if market_close is not None:
            market_close = market_close.reindex(out.index).ffill().bfill()
            market_ret_1 = market_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            market_ret_5 = market_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            market_trend = (market_close / market_close.rolling(ctx_roll).mean()).replace([np.inf, -np.inf], np.nan).ffill().bfill()
            out[f"{prefix}_Return_1"] = market_ret_1
            out[f"{prefix}_Return_5"] = market_ret_5
            out[f"{prefix}_Trend"] = market_trend
            market_return_cols.append(f"{prefix}_Return_1")
            info["global_markets_used"] += 1
        else:
            out[f"{prefix}_Return_1"] = 0.0
            out[f"{prefix}_Return_5"] = 0.0
            out[f"{prefix}_Trend"] = 1.0
    commodity_return_cols = []
    for commodity_name, commodity_ticker in COMMODITY_TICKERS.items():
        commodity_close = try_download_close_series(commodity_ticker, interval, start, end, intraday_period, as_of_date)
        prefix = f"CMDTY_{commodity_name}"
        if commodity_close is not None:
            commodity_close = commodity_close.reindex(out.index).ffill().bfill()
            commodity_ret_1 = commodity_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            commodity_ret_5 = commodity_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            commodity_trend = (commodity_close / commodity_close.rolling(ctx_roll).mean()).replace([np.inf, -np.inf], np.nan).ffill().bfill()
            out[f"{prefix}_Return_1"] = commodity_ret_1
            out[f"{prefix}_Return_5"] = commodity_ret_5
            out[f"{prefix}_Trend"] = commodity_trend
            commodity_return_cols.append(f"{prefix}_Return_1")
            info["commodities_used"] += 1
        else:
            out[f"{prefix}_Return_1"] = 0.0
            out[f"{prefix}_Return_5"] = 0.0
            out[f"{prefix}_Trend"] = 1.0
    if market_return_cols:
        out["Global_Market_Return_Composite"] = out[market_return_cols].mean(axis=1)
        out["Global_Market_Stress"] = out[market_return_cols].std(axis=1).fillna(0.0)
    else:
        out["Global_Market_Return_Composite"] = 0.0
        out["Global_Market_Stress"] = 0.0
    if commodity_return_cols:
        out["Commodity_Return_Composite"] = out[commodity_return_cols].mean(axis=1)
    else:
        out["Commodity_Return_Composite"] = 0.0
    # Interaction: kombinasikan global risk proxy dengan INDO VIX
    vix_proxy = out["INDOVIX_Change_1"].clip(-0.3, 0.3)
    out["Global_VIX_Interaction"] = out["Global_Market_Return_Composite"] * vix_proxy
    out["Commodity_VIX_Interaction"] = out["Commodity_Return_Composite"] * vix_proxy
    # Currency context (USD/IDR)
    usdidr_close = try_download_close_series(USDIDR_TICKER, interval, start, end, intraday_period, as_of_date)
    if usdidr_close is not None:
        usdidr_close = usdidr_close.reindex(out.index).ffill().bfill()
        usdidr_ret = usdidr_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["USDIDR_Level"] = usdidr_close
        out["USDIDR_Return_1"] = usdidr_ret
        out["USDIDR_Return_5"] = usdidr_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["USDIDR_Trend"] = (usdidr_close / usdidr_close.rolling(ctx_roll).mean()).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        info["currency_used"] = True
    else:
        out["USDIDR_Level"] = 0.0
        out["USDIDR_Return_1"] = 0.0
        out["USDIDR_Return_5"] = 0.0
        out["USDIDR_Trend"] = 1.0
    # Hindari drop semua baris saat beberapa sumber eksternal kosong/tidak sinkron.
    out = replace_inf_with_nan(out)
    out = out.ffill().bfill()
    for col in out.columns:
        if out[col].isna().all():
            out[col] = 1.0 if ("Trend" in col or col.endswith("_Level")) else 0.0
    num_cols = out.select_dtypes(include=[np.number]).columns
    out.loc[:, num_cols] = out.loc[:, num_cols].fillna(0.0)
    # Pastikan target tetap valid; jika NaN lebih baik dibuang spesifik di target saja.
    if "Target" in out.columns:
        out = out[out["Target"].notna()].copy()
    return out, info

def build_features(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    out = df.copy()
    if len(out) < MIN_RAW_ROWS_FOR_FEATURES:
        return pd.DataFrame()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]
    if interval in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        ma_short, ma_long = 8, 20
        ret_w1, ret_w2 = 3, 6
        vol_w1, vol_w2 = 6, 12
        lag_list = [1, 2, 3, 4, 6]
        roll_ctx = 12
    else:
        ma_short, ma_long = 20, 50
        ret_w1, ret_w2 = 5, 10
        vol_w1, vol_w2 = 10, 20
        lag_list = [1, 2, 3, 5, 10]
        roll_ctx = 20
    out["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    out["MA20"] = close.rolling(ma_short).mean()
    out["MA50"] = close.rolling(ma_long).mean()
    out["Trend_Filter"] = (out["MA20"] > out["MA50"]).astype(int)
    out["MACD"] = ta.trend.MACD(close).macd()
    bb = ta.volatility.BollingerBands(close)
    out["BB_high"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_width"] = out["BB_high"] - out["BB_low"]
    atr_window = max(2, min(14, len(out)))
    out["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=atr_window).average_true_range()
    out["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out["Return"] = close.pct_change()
    out["Return_5"] = close.pct_change(ret_w1)
    out["Return_10"] = close.pct_change(ret_w2)
    out["Volatility_10"] = out["Return"].rolling(vol_w1).std()
    out["Volatility_20"] = out["Return"].rolling(vol_w2).std()
    out["Volume_Change"] = volume.pct_change()
    for lag in lag_list:
        out[f"Lag_Return_{lag}"] = out["Return"].shift(lag)
        out[f"Lag_RSI_{lag}"] = out["RSI"].shift(lag)
    # Volume anomaly + buyer/seller pressure proxy
    out["Candle_Direction"] = np.sign(out["Close"] - out["Open"])
    out["Buy_Volume_Proxy"] = np.where(out["Candle_Direction"] > 0, out["Volume"], 0.0)
    out["Sell_Volume_Proxy"] = np.where(out["Candle_Direction"] < 0, out["Volume"], 0.0)
    vol_mean_20 = out["Volume"].rolling(roll_ctx).mean()
    vol_std_20 = out["Volume"].rolling(roll_ctx).std().replace(0, np.nan)
    out["Volume_ZScore"] = ((out["Volume"] - vol_mean_20) / vol_std_20).replace([np.inf, -np.inf], np.nan)
    buy_mean_20 = out["Buy_Volume_Proxy"].rolling(roll_ctx).mean()
    buy_std_20 = out["Buy_Volume_Proxy"].rolling(roll_ctx).std().replace(0, np.nan)
    sell_mean_20 = out["Sell_Volume_Proxy"].rolling(roll_ctx).mean()
    sell_std_20 = out["Sell_Volume_Proxy"].rolling(roll_ctx).std().replace(0, np.nan)
    out["Buy_Volume_Anomaly"] = ((out["Buy_Volume_Proxy"] - buy_mean_20) / buy_std_20).replace([np.inf, -np.inf], np.nan)
    out["Sell_Volume_Anomaly"] = ((out["Sell_Volume_Proxy"] - sell_mean_20) / sell_std_20).replace([np.inf, -np.inf], np.nan)
    out["Net_Volume_Anomaly"] = out["Buy_Volume_Anomaly"] - out["Sell_Volume_Anomaly"]
    out["Volume_Anomaly_Spike"] = (out["Volume_ZScore"] > 2.0).astype(int)
    # Support / Resistance features (lebih realistis untuk intraday/daily)
    if interval in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        sr_base = SR_LOOKBACK_INTRADAY
    else:
        sr_base = SR_LOOKBACK_DAILY
    sr_lookback = max(5, min(sr_base, max(5, len(out) - 1)))
    out["Support_Level"] = out["Low"].rolling(sr_lookback).min()
    out["Resistance_Level"] = out["High"].rolling(sr_lookback).max()
    out["Distance_To_Support"] = (out["Close"] - out["Support_Level"]) / out["Close"]
    out["Distance_To_Resistance"] = (out["Resistance_Level"] - out["Close"]) / out["Close"]
    out["Support_Break"] = (out["Close"] < out["Support_Level"] * 0.999).astype(int)
    out["Resistance_Break"] = (out["Close"] > out["Resistance_Level"] * 1.001).astype(int)
    out["Future_Return"] = out["Return"].shift(-1)
    if TARGET_MODE == "three_class":
        conditions = [
            out["Future_Return"] < -RETURN_THRESHOLD,
            out["Future_Return"].between(-RETURN_THRESHOLD, RETURN_THRESHOLD, inclusive="both"),
            out["Future_Return"] > RETURN_THRESHOLD,
        ]
        out["Target"] = np.select(conditions, [0, 1, 2], default=1).astype(int)
    else:
        out["Target"] = (out["Future_Return"] > RETURN_THRESHOLD).astype(int)
    out = replace_inf_with_nan(out)
    return out.dropna().copy()

def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, probs)

def split_time_series(df: pd.DataFrame):
    n = len(df)
    if n < 12:
        raise ValueError(f"Data terlalu sedikit setelah feature engineering: {n} baris. Coba perbesar rentang START/END, ganti TRADING_MODE, atau gunakan saham dengan histori lebih panjang.")
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    # pastikan masing-masing split minimal 1
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Split menghasilkan data kosong (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})."
        )
    return train_df, val_df, test_df

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    unique_classes = np.unique(y_true)
    if len(y_true) == 0 or len(unique_classes) < 2:
        return 0.5
    # Threshold search hanya relevan untuk kasus biner (0/1).
    # Untuk multiclass (mis. three_class), pakai decision default.
    if len(unique_classes) > 2:
        return 0.5
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
    upper = threshold + hold_band
    lower = threshold - hold_band
    if prob_up >= upper and prob_up > 0.5:
        return "BELI"
    if prob_up <= lower and prob_up < 0.5:
        return "JUAL"
    return "TAHAN"






def interpret_volume_flow(net_anomaly: float, buy_anomaly: float, sell_anomaly: float) -> str:
    if any(np.isnan(x) for x in [net_anomaly, buy_anomaly, sell_anomaly]):
        return "Volume flow tidak tersedia"
    if net_anomaly >= 1.0 and buy_anomaly > sell_anomaly:
        return "BUY PRESSURE DOMINAN (anomali beli kuat)"
    if net_anomaly <= -1.0 and sell_anomaly > buy_anomaly:
        return "SELL PRESSURE DOMINAN (anomali jual kuat)"
    return "Volume flow netral/campuran"




def interpret_sr_breakout(close_price: float, support: float, resistance: float, tol: float = 0.001):
    if any(np.isnan(x) for x in [close_price, support, resistance]):
        return "SR tidak tersedia", 0
    if close_price < support * (1 - tol):
        return "SUPPORT JEBOL (bearish breakout)", -1
    if close_price > resistance * (1 + tol):
        return "RESISTANCE JEBOL (bullish breakout)", 1
    return "Masih di dalam range support-resistance", 0


def adjust_signal_with_vix_fear(signal: str, prob_up: float, vix_level: float | None, vix_change_1: float | None):
    """
    Menyesuaikan signal dengan konteks fear dari Indonesia Volatility Index.
    - Jika VIX melonjak/tinggi, sinyal BELI dibuat lebih defensif.
    """
    if signal == "BELI":
        spike = (vix_change_1 is not None) and (not np.isnan(vix_change_1)) and (vix_change_1 >= VIX_SPIKE_THRESHOLD)
        high_level = (vix_level is not None) and (not np.isnan(vix_level)) and (vix_level >= VIX_HIGH_LEVEL)
        if spike and high_level:
            return "JUAL", "Fear tinggi (VIX spike + level tinggi): BELI diturunkan jadi JUAL defensif"
        if spike or high_level:
            return "TAHAN", "Fear meningkat dari INDO VIX: BELI diturunkan jadi TAHAN"
    if signal == "TAHAN":
        low_fear = (vix_level is not None) and (not np.isnan(vix_level)) and (vix_level < (VIX_HIGH_LEVEL * 0.8))
        fear_drop = (vix_change_1 is not None) and (not np.isnan(vix_change_1)) and (vix_change_1 <= -0.03)
        if low_fear and fear_drop and prob_up >= 0.58:
            return "BELI", "Fear menurun signifikan + probabilitas naik kuat: TAHAN dinaikkan jadi BELI"
    return signal, "Tidak ada penyesuaian signal dari INDO VIX"




class AdaptiveQuantEnsemble:
    """Ensemble quant adaptif: pembobotan model berbasis performa walk-forward + regularisasi probabilitas."""
    def __init__(self, models: list[tuple[str, object]], temperature: float = 0.85):
        self.models = models
        self.temperature = temperature
        self.model_weights_: dict[str, float] = {}
        self.fitted_models_: list[tuple[str, object]] = []
    def _time_split(self, X, y):
        n = len(y)
        split_idx = max(20, int(n * 0.8))
        split_idx = min(split_idx, n - 5) if n > 25 else max(1, n - 1)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        # kalibrasi bobot antar model dengan walk-forward holdout sederhana
        X_sub, y_sub, X_hold, y_hold = self._time_split(X_arr, y_arr)
        raw_scores = []
        temp_models = []
        for name, model in self.models:
            model.fit(X_sub, y_sub)
            p_hold = np.nan_to_num(model.predict_proba(X_hold)[:, 1], nan=0.5, posinf=1.0, neginf=0.0)
            th = find_best_threshold(y_hold, p_hold)
            pred_hold = (p_hold >= th).astype(int)
            score = balanced_accuracy_score(y_hold, pred_hold) if len(np.unique(y_hold)) > 1 else 0.5
            raw_scores.append(max(0.01, float(score)))
            temp_models.append((name, model))
        # softmax-like weighting agar stabil
        score_arr = np.array(raw_scores, dtype=float)
        score_arr = np.exp((score_arr - np.max(score_arr)) * 8.0)
        score_arr = score_arr / np.sum(score_arr)
        self.model_weights_ = {name: float(w) for (name, _), w in zip(temp_models, score_arr)}
        # fit ulang seluruh model dengan full training
        self.fitted_models_ = []
        for name, model in self.models:
            model.fit(X_arr, y_arr)
            self.fitted_models_.append((name, model))
        return self
    def predict_proba(self, X):
        if not self.fitted_models_:
            raise ValueError("AdaptiveQuantEnsemble belum di-fit.")
        probs = []
        weights = []
        for name, model in self.fitted_models_:
            p = np.nan_to_num(model.predict_proba(X), nan=0.5, posinf=1.0, neginf=0.0)
            probs.append(p)
            weights.append(float(self.model_weights_.get(name, 1.0 / max(1, len(self.fitted_models_)))))
        w = np.array(weights, dtype=float)
        w = w / np.sum(w)
        blended = np.zeros_like(probs[0], dtype=float)
        for wi, pi in zip(w, probs):
            blended += wi * pi
        # realism: shrink probabilitas saat antar-model tidak sepakat (kurangi ekstrem bullish/bearish)
        p_up_each = np.column_stack([p[:, 1] for p in probs])
        disagreement = np.std(p_up_each, axis=1)
        shrink = np.clip(disagreement * self.temperature, 0.0, 0.35)
        p_up = blended[:, 1]
        p_up = (1 - shrink) * p_up + shrink * 0.5
        p_down = 1 - p_up
        return np.column_stack([p_down, p_up])


def build_adaptive_quant_ensemble(include_xgb: bool):
    base_models = [
        (
            "hgb",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=4,
                max_iter=450,
                min_samples_leaf=20,
                random_state=SEED,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=10,
                min_samples_leaf=6,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "lr",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
                ]
            ),
        ),
    ]
    if include_xgb and TARGET_MODE != "three_class":
        try:
            from xgboost import XGBClassifier
            base_models.append(
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                )
            )
        except Exception:
            pass
    return AdaptiveQuantEnsemble(models=base_models, temperature=0.85)




class RegimeAwareAdaptiveQuant:
    """Model quant regime-aware: gabungkan 2 ensemble (low-vol & high-vol) dengan gating volatilitas."""
    def __init__(self, include_xgb: bool = True):
        self.include_xgb = include_xgb
        self.low_model = build_adaptive_quant_ensemble(include_xgb=include_xgb)
        self.high_model = build_adaptive_quant_ensemble(include_xgb=include_xgb)
        self.regime_center_ = 0.0
        self.regime_scale_ = 1.0
        self.regime_col_ = "Volatility_20"
    def _extract_regime_series(self, X):
        if hasattr(X, "columns") and self.regime_col_ in X.columns:
            ser = X[self.regime_col_].to_numpy(dtype=float)
        else:
            # fallback jika kolom tidak ada / input ndarray
            arr = np.asarray(X, dtype=float)
            ser = arr[:, -1] if arr.ndim == 2 and arr.shape[1] > 0 else np.zeros(len(arr), dtype=float)
        ser = np.nan_to_num(ser, nan=np.nanmedian(ser) if len(ser) else 0.0, posinf=0.0, neginf=0.0)
        return ser
    def fit(self, X, y):
        y_arr = np.asarray(y)
        regime = self._extract_regime_series(X)
        self.regime_center_ = float(np.nanmedian(regime))
        self.regime_scale_ = float(np.nanstd(regime)) if np.nanstd(regime) > 1e-9 else 1.0
        high_mask = regime >= self.regime_center_
        low_mask = ~high_mask
        # fallback jika salah satu regime terlalu sedikit
        if np.sum(low_mask) < 20 or np.sum(high_mask) < 20:
            self.low_model.fit(X, y_arr)
            self.high_model.fit(X, y_arr)
            return self
        if hasattr(X, "iloc"):
            self.low_model.fit(X.iloc[low_mask], y_arr[low_mask])
            self.high_model.fit(X.iloc[high_mask], y_arr[high_mask])
        else:
            X_arr = np.asarray(X)
            self.low_model.fit(X_arr[low_mask], y_arr[low_mask])
            self.high_model.fit(X_arr[high_mask], y_arr[high_mask])
        return self
    def predict_proba(self, X):
        p_low = self.low_model.predict_proba(X)
        p_high = self.high_model.predict_proba(X)
        regime = self._extract_regime_series(X)
        z = (regime - self.regime_center_) / self.regime_scale_
        gate_high = 1 / (1 + np.exp(-z))
        gate_high = np.clip(gate_high, 0.05, 0.95)
        p_up = (1 - gate_high) * p_low[:, 1] + gate_high * p_high[:, 1]
        p_up = np.clip(p_up, 0.01, 0.99)
        return np.column_stack([1 - p_up, p_up])


def get_model_candidates():
    candidates = {
        "RegimeAwareAdaptiveQuant": RegimeAwareAdaptiveQuant(include_xgb=True),
        "AdaptiveQuantEnsemble": build_adaptive_quant_ensemble(include_xgb=True),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=4,
            max_iter=400,
            min_samples_leaf=20,
            random_state=SEED,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        ),
    }
    if TARGET_MODE != "three_class":
        try:
            from xgboost import XGBClassifier
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
            )
        except Exception:
            pass
    return candidates


def map_target_label_to_text(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y)
    if TARGET_MODE == "three_class":
        mapping = {0: "TURUN", 1: "SIDEWAYS", 2: "NAIK"}
        return np.array([mapping.get(int(v), str(v)) for v in y_arr])
    return np.where(y_arr == 1, "NAIK", "TURUN")


def extract_direction_probs(model, proba: np.ndarray):
    classes = list(getattr(model, "classes_", []))
    if TARGET_MODE == "three_class" and len(classes) >= 3:
        p_down = proba[:, classes.index(0)] if 0 in classes else np.zeros(len(proba))
        p_side = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(proba))
        p_up = proba[:, classes.index(2)] if 2 in classes else np.zeros(len(proba))
        return p_down, p_side, p_up
    # binary fallback
    p_up = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    p_down = 1 - p_up
    p_side = np.zeros(len(p_up))
    return p_down, p_side, p_up


def decide_signal_three_class(prob_down: float, prob_side: float, prob_up: float, threshold: float) -> str:
    if prob_up >= max(threshold, prob_down, prob_side):
        return "BELI"
    if prob_down >= max(threshold, prob_up, prob_side):
        return "JUAL"
    return "TAHAN"


def analyze_chart_bias(trend_filter: float, rsi: float, macd: float, close_price: float, ma20: float, ma50: float) -> str:
    """Analisis chart realistis (Bullish/Bearish) berbasis konfluensi trend+momentum."""
    score = 0
    if not np.isnan(trend_filter):
        score += 1 if trend_filter >= 0.5 else -1
    if not np.isnan(close_price) and not np.isnan(ma20):
        score += 1 if close_price >= ma20 else -1
    if not np.isnan(ma20) and not np.isnan(ma50):
        score += 1 if ma20 >= ma50 else -1
    if not np.isnan(macd):
        score += 1 if macd >= 0 else -1
    if not np.isnan(rsi):
        if rsi >= 55:
            score += 1
        elif rsi <= 45:
            score -= 1
    return "Bullish" if score >= 1 else "Bearish"


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    fitted = model
    fitted.fit(X_train, y_train)
    if USE_PROB_CALIBRATION:
        try:
            calibrated = CalibratedClassifierCV(estimator=fitted, method="sigmoid", cv=3)
            calibrated.fit(X_train, y_train)
            fitted = calibrated
        except Exception:
            pass
    proba = np.nan_to_num(fitted.predict_proba(X_val), nan=0.5, posinf=1.0, neginf=0.0)
    p_down, p_side, p_up = extract_direction_probs(fitted, proba)
    if TARGET_MODE == "three_class":
        val_preds = np.asarray(getattr(fitted, "classes_", np.array([0, 1, 2])))[np.argmax(proba, axis=1)]
        threshold = 0.5
        val_auc = float("nan")
    else:
        threshold = find_best_threshold(y_val, p_up)
        val_preds = (p_up >= threshold).astype(int)
        val_auc = safe_auc(y_val, p_up)
    return {
        "name": name,
        "model": fitted,
        "threshold": threshold,
        "val_auc": val_auc,
        "val_bacc": balanced_accuracy_score(y_val, val_preds),
        "val_acc": accuracy_score(y_val, val_preds),
    }


def estimate_expected_return(prob_up: float, return_series: pd.Series) -> float:
    up_returns = return_series[return_series > 0]
    down_returns = return_series[return_series <= 0]
    mean_up = float(up_returns.mean()) if len(up_returns) else 0.0
    mean_down = float(down_returns.mean()) if len(down_returns) else 0.0
    return (prob_up * mean_up) + ((1 - prob_up) * mean_down)


def suggest_stoploss(signal: str, last_close: float, atr_value: float, prob_up: float, base_mult: float = 1.5):
    if signal == "TAHAN":
        return None, None, "Tidak ada stop-loss karena sinyal TAHAN"
    confidence = abs(prob_up - 0.5) * 2
    multiplier = base_mult + (0.7 * confidence)
    if atr_value is None or np.isnan(atr_value) or atr_value <= 0:
        fallback_pct = 0.03
        if signal == "BELI":
            stop = last_close * (1 - fallback_pct)
            return stop, fallback_pct * 100, "Fallback 3% (ATR tidak valid)"
        stop = last_close * (1 + fallback_pct)
        return stop, fallback_pct * 100, "Fallback 3% (ATR tidak valid, skenario short)"
    if signal == "BELI":
        stop = last_close - (multiplier * atr_value)
        stop_pct = ((last_close - stop) / last_close) * 100
        return stop, stop_pct, f"ATR x {multiplier:.2f} di bawah harga masuk"
    stop = last_close + (multiplier * atr_value)
    stop_pct = ((stop - last_close) / last_close) * 100
    return stop, stop_pct, f"ATR x {multiplier:.2f} di atas harga referensi"




def generate_next_bej_session_timestamps(start_ts: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    ts = pd.Timestamp(start_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(JKT_TZ)
    else:
        ts = ts.tz_convert(JKT_TZ)
    session_hours = [9, 10, 11, 12, 14, 15, 16]
    out = []
    cur = ts
    while len(out) < periods:
        cur = cur + pd.Timedelta(hours=1)
        # normalisasi menit/detik ke jam bulat
        cur = cur.replace(minute=0, second=0, microsecond=0)
        # jika weekend, lompat ke senin jam 09:00
        while cur.dayofweek >= 5:
            cur = (cur + pd.Timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        if cur.hour in session_hours and cur.dayofweek < 5:
            out.append(cur)
    return pd.DatetimeIndex(out)


def estimate_ihsg_influence_on_latest(model, latest_row: pd.DataFrame) -> tuple[float, float]:
    """Bandingkan probabilitas dengan vs tanpa fitur IHSG pada baris terakhir."""
    base_proba = np.nan_to_num(model.predict_proba(latest_row), nan=0.5, posinf=1.0, neginf=0.0)
    _, _, base_up = extract_direction_probs(model, base_proba)
    base_prob = float(base_up[0])
    no_ihsg = latest_row.copy()
    for c in ["IHSG_Return_1", "IHSG_Return_5", "IHSG_Volatility_10"]:
        if c in no_ihsg.columns:
            no_ihsg[c] = 0.0
    no_ihsg_proba = np.nan_to_num(model.predict_proba(no_ihsg), nan=0.5, posinf=1.0, neginf=0.0)
    _, _, no_ihsg_up = extract_direction_probs(model, no_ihsg_proba)
    no_ihsg_prob = float(no_ihsg_up[0])
    return base_prob, (base_prob - no_ihsg_prob)




def suggest_tp_sl_from_sr(signal: str, entry_price: float, support: float, resistance: float, atr_value: float | None, interval: str = "1d"):
    """
    Menentukan take-profit dan stop-loss realistis berbasis level support/resistance.
    - BELI: SL di bawah support (atau ATR fallback), TP mendekati resistance.
    - JUAL: SL di atas resistance, TP mendekati support.
    """
    if signal == "TAHAN":
        return None, None, "Tidak ada TP/SL karena sinyal TAHAN"
    atr_buffer = 0.5 * atr_value if atr_value is not None and not np.isnan(atr_value) and atr_value > 0 else entry_price * 0.005
    tp_atr_mult = 1.8 if interval in {"1h", "60m", "30m", "15m", "5m", "1m"} else 2.5
    sl_atr_mult = 1.2 if interval in {"1h", "60m", "30m", "15m", "5m", "1m"} else 1.8
    if signal == "BELI":
        sl_floor = entry_price - (sl_atr_mult * atr_buffer)
        if not np.isnan(support):
            sl = max(support - atr_buffer, sl_floor)
        else:
            sl = sl_floor
        tp_cap = entry_price + (tp_atr_mult * atr_buffer)
        if not np.isnan(resistance):
            tp = min(resistance - (0.2 * atr_buffer), tp_cap)
        else:
            tp = tp_cap
        if tp <= entry_price:
            tp = entry_price * 1.005
        if sl >= entry_price:
            sl = entry_price * 0.995
        return tp, sl, "TP/SL dari resistance-support + buffer ATR (skenario long)"
    # signal == JUAL (short/defensive)
    sl_cap = entry_price + (sl_atr_mult * atr_buffer)
    if not np.isnan(resistance):
        sl = min(resistance + atr_buffer, sl_cap)
    else:
        sl = sl_cap
    tp_floor = entry_price - (tp_atr_mult * atr_buffer)
    if not np.isnan(support):
        tp = max(support + (0.2 * atr_buffer), tp_floor)
    else:
        tp = tp_floor
    if tp >= entry_price:
        tp = entry_price * 0.995
    if sl <= entry_price:
        sl = entry_price * 1.005
    return tp, sl, "TP/SL dari resistance-support + buffer ATR (skenario short)"


def suggest_entry_range(signal: str, last_close: float, support: float, resistance: float, atr_value: float | None, interval: str = "1d"):
    """Saran range entry realistis berbasis ATR + support/resistance + konteks sinyal."""
    atr_buffer = 0.35 * atr_value if atr_value is not None and not np.isnan(atr_value) and atr_value > 0 else last_close * 0.004
    intraday = interval in {"1h", "60m", "30m", "15m", "5m", "1m"}
    pullback_mult = 0.8 if intraday else 1.1
    breakout_mult = 0.6 if intraday else 0.9
    if signal == "BELI":
        # Entry ideal saat pullback mendekati support atau area diskon dari harga terakhir.
        base_low = last_close - (pullback_mult * atr_buffer)
        base_high = last_close + (0.25 * atr_buffer)
        if not np.isnan(support):
            low = max(support + (0.05 * atr_buffer), base_low)
        else:
            low = base_low
        if not np.isnan(resistance):
            high_cap = resistance - (0.2 * atr_buffer)
            high = min(base_high, high_cap)
        else:
            high = base_high
        if low >= high:
            low = last_close * 0.997
            high = last_close * 1.003
        return low, high, "Entry BUY disarankan saat pullback (dekat support/area diskon ATR)"
    if signal == "JUAL":
        # Entry ideal saat rebound mendekati resistance atau area premium dari harga terakhir.
        base_low = last_close - (0.25 * atr_buffer)
        base_high = last_close + (breakout_mult * atr_buffer)
        if not np.isnan(support):
            low_floor = support + (0.2 * atr_buffer)
            low = max(base_low, low_floor)
        else:
            low = base_low
        if not np.isnan(resistance):
            high = min(resistance - (0.05 * atr_buffer), base_high)
        else:
            high = base_high
        if low >= high:
            low = last_close * 0.997
            high = last_close * 1.003
        return low, high, "Entry SELL disarankan saat rebound (dekat resistance/area premium ATR)"
    neutral_low = last_close - (0.5 * atr_buffer)
    neutral_high = last_close + (0.5 * atr_buffer)
    return neutral_low, neutral_high, "Mode TAHAN: entry agresif tidak disarankan, hanya range observasi"


def forecast_next_periods(last_close: float, expected_return: float, start_date: pd.Timestamp, periods: int, interval: str,
                          return_history: pd.Series, prob_up: float) -> pd.DataFrame:
    """Forecast path yang lebih realistis: tidak pakai return konstan, tapi bootstrap return historis kondisional."""
    if interval in {"1h", "60m"}:
        future_index = generate_next_bej_session_timestamps(start_ts=start_date, periods=periods)
        price_col = "Predicted_Close_1h"
        ret_col = "Simulated_Return"
    else:
        future_index = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=periods)
        price_col = "Predicted_Close_1d"
        ret_col = "Simulated_Return"
    hist = return_history.replace([np.inf, -np.inf], np.nan).dropna()
    if len(hist) < 10:
        simulated_returns = np.full(periods, expected_return)
    else:
        up_hist = hist[hist > 0]
        down_hist = hist[hist <= 0]
        if len(up_hist) == 0:
            up_hist = hist
        if len(down_hist) == 0:
            down_hist = hist
        rng = np.random.default_rng(SEED)
        simulated_returns = []
        local_prob_up = min(max(prob_up, 0.05), 0.95)
        for i in range(periods):
            pick_up = rng.random() < local_prob_up
            if pick_up:
                r = float(rng.choice(up_hist.values))
            else:
                r = float(rng.choice(down_hist.values))
            # clamp agar tidak ekstrem dan tambah sedikit mean-reversion ke expected return
            r = float(np.clip(r, -0.04, 0.04))
            r = 0.7 * r + 0.3 * expected_return
            simulated_returns.append(r)
            # update probabilitas secara ringan supaya jalur tidak monoton
            local_prob_up = 0.8 * local_prob_up + 0.2 * (0.5 + np.tanh(r * 20) * 0.15)
            local_prob_up = min(max(local_prob_up, 0.2), 0.8)
        simulated_returns = np.array(simulated_returns)
    prices, price = [], float(last_close)
    for r in simulated_returns:
        price = price * (1 + r)
        prices.append(price)
    out = pd.DataFrame({"Date": future_index, price_col: prices, ret_col: simulated_returns})
    out["Expected_Return_Base"] = expected_return
    return out


def prepare_dataset(interval: str) -> tuple[pd.DataFrame, str, dict]:
    if interval in {"1h", "60m", "30m", "15m", "5m", "1m"}:
        periods_to_try = [INTRADAY_PERIOD] + [p for p in INTRADAY_FALLBACK_PERIODS if p != INTRADAY_PERIOD]
        for period in periods_to_try:
            raw = download_data(SYMBOL, interval, START, END, period)
            raw = apply_as_of_cutoff(raw, AS_OF_DATE)
            if raw.empty:
                continue
            base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            feat = build_features(base, interval)
            if feat.empty:
                continue
            min_required = MIN_ROWS_INTRADAY if interval in {"1h", "60m", "30m", "15m", "5m", "1m"} else MIN_ROWS_AFTER_FEATURES
            if len(feat) >= min_required:
                feat, info = add_market_context_features(feat, interval, START, END, period, AS_OF_DATE)
                return feat, period, info
        # fallback: pakai dataset terbaik yang masih memungkinkan split
        best_feat, best_period = None, None
        for period in periods_to_try:
            raw = download_data(SYMBOL, interval, START, END, period)
            raw = apply_as_of_cutoff(raw, AS_OF_DATE)
            if raw.empty:
                continue
            base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            feat = build_features(base, interval)
            if feat.empty:
                continue
            if best_feat is None or len(feat) > len(best_feat):
                best_feat, best_period = feat, period
        if best_feat is not None and len(best_feat) >= 12:
            best_feat, info = add_market_context_features(best_feat, interval, START, END, best_period, AS_OF_DATE)
            return best_feat, f"{best_period} (best-effort)", info
        raise ValueError(
            f"Data intraday terlalu sedikit setelah feature engineering. Coba period lebih besar. Tried={periods_to_try}"
        )
    raw = download_data(SYMBOL, interval, START, END, INTRADAY_PERIOD)
    raw = apply_as_of_cutoff(raw, AS_OF_DATE)
    if raw.empty:
        raise ValueError(
            "Data harian kosong dari Yahoo Finance. Cek SYMBOL/INTERVAL, atau perlebar START/END."
        )
    base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    feat = build_features(base, interval)
    if feat.empty or len(feat) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(
            f"Data harian terlalu sedikit setelah feature engineering: {len(feat)} baris. "
            "Coba perlebar START/END atau ganti saham dengan histori lebih lengkap."
        )
    feat, info = add_market_context_features(feat, interval, START, END, INTRADAY_PERIOD, AS_OF_DATE)
    if feat.empty:
        raise ValueError(
            "Semua baris hilang setelah penggabungan fitur konteks market. "
            "Coba nonaktifkan sumber eksternal yang bermasalah atau ganti simbol."
        )
    return feat, "start/end", info


def main():
    mode_config = get_runtime_mode_config()
    interval = mode_config["interval"]
    df, data_window_used, market_info = prepare_dataset(interval)
    train_df, val_df, test_df = split_time_series(df)
    feature_cols = [c for c in df.columns if c != "Target"]
    X_train, y_train = train_df[feature_cols], train_df["Target"].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df["Target"].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df["Target"].to_numpy()
    X_train = replace_inf_with_nan(X_train)
    X_val = replace_inf_with_nan(X_val)
    X_test = replace_inf_with_nan(X_test)
    candidates = get_model_candidates()
    evaluations = [evaluate_model(name, model, X_train, y_train, X_val, y_val) for name, model in candidates.items()]
    best = max(evaluations, key=lambda x: (x["val_bacc"], np.nan_to_num(x["val_auc"], nan=-1.0)))
    best_model = best["model"]
    best_threshold = best["threshold"]
    test_proba = np.nan_to_num(best_model.predict_proba(X_test), nan=0.5, posinf=1.0, neginf=0.0)
    probs_down, probs_side, probs_up = extract_direction_probs(best_model, test_proba)
    if TARGET_MODE == "three_class":
        preds = np.asarray(getattr(best_model, "classes_", np.array([0, 1, 2])))[np.argmax(test_proba, axis=1)]
    else:
        preds = (probs_up >= best_threshold).astype(int)
    print(f"Mode trading : {mode_config['name']} ({mode_config['horizon_note']})")
    print(f"Mode interval: {interval}")
    print(f"Data window used: {data_window_used}")
    print(f"As-of cutoff: {AS_OF_DATE if AS_OF_DATE else 'latest available'}")
    print(f"IHSG context used: {market_info.get('ihsg_used')}")
    print(f"Global markets used: {market_info.get('global_markets_used', 0)}/{len(GLOBAL_INDEX_TICKERS)}")
    print(f"Commodities used  : {market_info.get('commodities_used', 0)}/{len(COMMODITY_TICKERS)}")
    print(f"INDO VIX source : {market_info.get('indovix_symbol') if market_info.get('indovix_symbol') else 'not found (filled neutral)'}")
    print(f"USD/IDR context: {market_info.get('currency_used')}")
    print(f"Timestamp aligned: {market_info.get('timestamps_aligned')}")
    print(f"VIX fear rules  : spike>={VIX_SPIKE_THRESHOLD:.2%}, high_level>={VIX_HIGH_LEVEL}")
    print(f"BEI sessions    : {SESSION1_START}-{SESSION1_END} & {SESSION2_START}-{SESSION2_END} ({JKT_TZ})")
    print("Model candidates (validation):")
    for ev in evaluations:
        auc_text = "nan" if np.isnan(ev["val_auc"]) else f"{ev['val_auc']:.4f}"
        print(
            f"- {ev['name']}: AUC={auc_text}, "
            f"BalancedAcc={ev['val_bacc']:.4f}, Acc={ev['val_acc']:.4f}, Threshold={ev['threshold']:.2f}"
        )
    print(f"\nModel terpilih: {best['name']}")
    print(f"Best threshold (validation): {best_threshold:.2f}")
    print(f"Test accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, preds):.4f}")
    if TARGET_MODE == "three_class":
        print("Test ROC-AUC: nan (three_class mode)")
    else:
        print(f"Test ROC-AUC: {safe_auc(y_test, probs_up):.4f}" if len(np.unique(y_test)) > 1 else "Test ROC-AUC: nan")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=4, zero_division=0))
    result = pd.DataFrame(
        {
            "Date": test_df.index,
            "Close": test_df["Close"].to_numpy().reshape(-1),
            "Prob_Naik": probs_up,
            "Prob_Turun": probs_down,
            "Aktual": map_target_label_to_text(y_test),
            "INDOVIX_Level": test_df["INDOVIX_Level"].to_numpy().reshape(-1) if "INDOVIX_Level" in test_df.columns else np.nan,
            "INDOVIX_Change_1": test_df["INDOVIX_Change_1"].to_numpy().reshape(-1) if "INDOVIX_Change_1" in test_df.columns else np.nan,
            "Buy_Volume_Anomaly": test_df["Buy_Volume_Anomaly"].to_numpy().reshape(-1) if "Buy_Volume_Anomaly" in test_df.columns else np.nan,
            "Sell_Volume_Anomaly": test_df["Sell_Volume_Anomaly"].to_numpy().reshape(-1) if "Sell_Volume_Anomaly" in test_df.columns else np.nan,
            "Net_Volume_Anomaly": test_df["Net_Volume_Anomaly"].to_numpy().reshape(-1) if "Net_Volume_Anomaly" in test_df.columns else np.nan,
            "Volume_Anomaly_Spike": test_df["Volume_Anomaly_Spike"].to_numpy().reshape(-1) if "Volume_Anomaly_Spike" in test_df.columns else 0,
            "Support_Level": test_df["Support_Level"].to_numpy().reshape(-1) if "Support_Level" in test_df.columns else np.nan,
            "Resistance_Level": test_df["Resistance_Level"].to_numpy().reshape(-1) if "Resistance_Level" in test_df.columns else np.nan,
            "Trend_Filter": test_df["Trend_Filter"].to_numpy().reshape(-1) if "Trend_Filter" in test_df.columns else np.nan,
            "RSI": test_df["RSI"].to_numpy().reshape(-1) if "RSI" in test_df.columns else np.nan,
            "MACD": test_df["MACD"].to_numpy().reshape(-1) if "MACD" in test_df.columns else np.nan,
            "MA20": test_df["MA20"].to_numpy().reshape(-1) if "MA20" in test_df.columns else np.nan,
            "MA50": test_df["MA50"].to_numpy().reshape(-1) if "MA50" in test_df.columns else np.nan,
        }
    )
    if TARGET_MODE == "three_class":
        result["Prob_Sideways"] = probs_side
        result["Signal_Dasar"] = result.apply(lambda r: decide_signal_three_class(float(r["Prob_Turun"]), float(r["Prob_Sideways"]), float(r["Prob_Naik"]), best_threshold), axis=1)
    else:
        result["Prob_Sideways"] = 0.0
        result["Signal_Dasar"] = result["Prob_Naik"].apply(lambda p: decide_signal(float(p), best_threshold))
    result["Analisis_Chart"] = result.apply(
        lambda r: analyze_chart_bias(
            trend_filter=float(r["Trend_Filter"]) if "Trend_Filter" in result.columns else np.nan,
            rsi=float(r["RSI"]) if "RSI" in result.columns else np.nan,
            macd=float(r["MACD"]) if "MACD" in result.columns else np.nan,
            close_price=float(r["Close"]),
            ma20=float(r["MA20"]) if "MA20" in result.columns else np.nan,
            ma50=float(r["MA50"]) if "MA50" in result.columns else np.nan,
        ),
        axis=1,
    )
    result["Catatan_VolumeFlow"] = result.apply(
        lambda r: interpret_volume_flow(
            float(r["Net_Volume_Anomaly"]) if "Net_Volume_Anomaly" in result.columns else np.nan,
            float(r["Buy_Volume_Anomaly"]) if "Buy_Volume_Anomaly" in result.columns else np.nan,
            float(r["Sell_Volume_Anomaly"]) if "Sell_Volume_Anomaly" in result.columns else np.nan,
        ),
        axis=1,
    )
    result[["Signal_Akhir", "Catatan_VIX"]] = result.apply(
        lambda r: pd.Series(
            adjust_signal_with_vix_fear(
                signal=r["Signal_Dasar"],
                prob_up=float(r["Prob_Naik"]),
                vix_level=float(r["INDOVIX_Level"]) if "INDOVIX_Level" in result.columns else np.nan,
                vix_change_1=float(r["INDOVIX_Change_1"]) if "INDOVIX_Change_1" in result.columns else np.nan,
            )
        ),
        axis=1,
    )
    result_display = result[
        [
            "Date",
            "Close",
            "Prob_Naik",
            "Prob_Turun",
            "Prob_Sideways",
            "Aktual",
            "Analisis_Chart",
            "INDOVIX_Level",
            "Catatan_VolumeFlow",
            "Signal_Akhir",
        ]
    ]
    print("\nContoh output (10 baris terakhir):")
    print(result_display.tail(10).to_string(index=False))
    latest_row = replace_inf_with_nan(df[feature_cols].tail(1))
    latest_proba = np.nan_to_num(best_model.predict_proba(latest_row), nan=0.5, posinf=1.0, neginf=0.0)
    prob_down_arr, prob_side_arr, prob_up_arr = extract_direction_probs(best_model, latest_proba)
    prob_up_now = float(prob_up_arr[0])
    prob_down_now = float(prob_down_arr[0])
    prob_side_now = float(prob_side_arr[0])
    prob_with_ihsg, delta_ihsg = estimate_ihsg_influence_on_latest(best_model, latest_row)
    signal_now_base = decide_signal_three_class(prob_down_now, prob_side_now, prob_up_now, best_threshold) if TARGET_MODE == "three_class" else decide_signal(prob_up_now, best_threshold)
    vix_level_now = float(df["INDOVIX_Level"].iloc[-1]) if "INDOVIX_Level" in df.columns else np.nan
    vix_change_now = float(df["INDOVIX_Change_1"].iloc[-1]) if "INDOVIX_Change_1" in df.columns else np.nan
    buy_vol_anom_now = float(df["Buy_Volume_Anomaly"].iloc[-1]) if "Buy_Volume_Anomaly" in df.columns else np.nan
    sell_vol_anom_now = float(df["Sell_Volume_Anomaly"].iloc[-1]) if "Sell_Volume_Anomaly" in df.columns else np.nan
    net_vol_anom_now = float(df["Net_Volume_Anomaly"].iloc[-1]) if "Net_Volume_Anomaly" in df.columns else np.nan
    vol_flow_note = interpret_volume_flow(net_vol_anom_now, buy_vol_anom_now, sell_vol_anom_now)
    support_now = float(df["Support_Level"].iloc[-1]) if "Support_Level" in df.columns else np.nan
    resistance_now = float(df["Resistance_Level"].iloc[-1]) if "Resistance_Level" in df.columns else np.nan
    sr_note, sr_bias = interpret_sr_breakout(float(df["Close"].iloc[-1]), support_now, resistance_now)
    signal_now, vix_note = adjust_signal_with_vix_fear(signal_now_base, prob_up_now, vix_level_now, vix_change_now)
    if sr_bias == -1 and signal_now == "BELI":
        signal_now = "TAHAN"
    if sr_bias == 1 and signal_now == "JUAL":
        signal_now = "TAHAN"
    atr_now = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else np.nan
    stoploss_price, stoploss_pct, stoploss_note = suggest_stoploss(
        signal=signal_now,
        last_close=float(df["Close"].iloc[-1]),
        atr_value=atr_now,
        prob_up=prob_up_now,
    )
    tp_price, sl_price_sr, tp_sl_note = suggest_tp_sl_from_sr(
        signal=signal_now,
        entry_price=float(df["Close"].iloc[-1]),
        support=support_now,
        resistance=resistance_now,
        atr_value=atr_now,
        interval=interval,
    )
    entry_low, entry_high, entry_note = suggest_entry_range(
        signal=signal_now,
        last_close=float(df["Close"].iloc[-1]),
        support=support_now,
        resistance=resistance_now,
        atr_value=atr_now,
        interval=interval,
    )
    print("\nSignal saat ini:")
    print(f"Timestamp terakhir   : {df.index[-1]}")
    print(f"Prob_Naik saat ini   : {prob_up_now:.4f}")
    print(f"Prob_Turun saat ini  : {prob_down_now:.4f}")
    if TARGET_MODE == "three_class":
        print(f"Prob_Sideways kini   : {prob_side_now:.4f}")
    print(f"IHSG impact (delta)  : {delta_ihsg:+.4f} pada Prob_Naik (vs IHSG=0)")
    print(f"Signal dasar         : {signal_now_base}")
    print(f"INDO VIX level/change: {vix_level_now:.4f} / {vix_change_now:.4f}")
    print(f"Signal saat ini      : {signal_now}")
    print(f"Catatan INDO VIX     : {vix_note}")
    print(f"Volume anomaly (B/S/N): {buy_vol_anom_now:.4f} / {sell_vol_anom_now:.4f} / {net_vol_anom_now:.4f}")
    print(f"Catatan volume flow  : {vol_flow_note}")
    print(f"Support/Resistance   : {support_now:.2f} / {resistance_now:.2f}")
    print(f"Saran range entry    : {entry_low:.2f} - {entry_high:.2f}")
    print(f"Catatan entry range  : {entry_note}")
    print(f"Catatan breakout SR  : {sr_note}")
    if stoploss_price is not None:
        print(f"Stop-loss saran (ATR): {stoploss_price:.2f} ({stoploss_pct:.2f}%)")
    print(f"Catatan stop-loss    : {stoploss_note}")
    if tp_price is not None and sl_price_sr is not None:
        rr = abs((tp_price - float(df["Close"].iloc[-1])) / (float(df["Close"].iloc[-1]) - sl_price_sr)) if signal_now == "BELI" and float(df["Close"].iloc[-1]) != sl_price_sr else None
        if signal_now == "JUAL":
            rr = abs((float(df["Close"].iloc[-1]) - tp_price) / (sl_price_sr - float(df["Close"].iloc[-1]))) if sl_price_sr != float(df["Close"].iloc[-1]) else None
        print(f"Take-profit (SR)     : {tp_price:.2f}")
        print(f"Stop-loss (SR)       : {sl_price_sr:.2f}")
        if rr is not None and np.isfinite(rr):
            print(f"Risk/Reward approx   : 1:{rr:.2f}")
    print(f"Catatan TP/SL SR     : {tp_sl_note}")
    expected_ret = estimate_expected_return(prob_up_now, train_df["Return"])
    forecast = forecast_next_periods(
        last_close=float(df["Close"].iloc[-1]),
        expected_return=expected_ret,
        start_date=df.index[-1],
        periods=mode_config["forecast_periods"],
        interval=interval,
        return_history=train_df["Return"],
        prob_up=prob_up_now,
    )
    print(f"\n{mode_config['forecast_label']}:")
    print(forecast.to_string(index=False))

if __name__ == "__main__":
    main()
