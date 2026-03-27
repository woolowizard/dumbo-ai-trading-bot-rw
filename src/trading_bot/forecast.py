# ============================================================
#  LightGBM Stock Forecaster — Core Logic
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime


# Stato globale del modello
_state = {"model": None, "feature_cols": None}


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, n_lags: int, forecast_horizon: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["log_return"] = np.log(close / close.shift(1))
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    df["ema_9"] = EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close, window=21).ema_indicator()
    df["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["ema_cross"] = df["ema_9"] - df["ema_21"]

    macd_ind = MACD(close)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    df["rsi"] = RSIIndicator(close, window=14).rsi()

    stoch = StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    df["vol_change"] = volume.pct_change()
    df["vol_ma_ratio"] = volume / volume.rolling(20).mean()

    df["candle_body"] = (close - df["Open"]).abs()
    df["candle_range"] = high - low
    df["upper_wick"] = high - close.clip(lower=df["Open"])
    df["lower_wick"] = close.clip(upper=df["Open"]) - low

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 5)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 5)

    for lag in range(1, n_lags + 1):
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    for w in [4, 8, 16]:
        df[f"roll_mean_{w}"] = close.rolling(w).mean()
        df[f"roll_std_{w}"] = close.rolling(w).std()
        df[f"roll_max_{w}"] = close.rolling(w).max()
        df[f"roll_min_{w}"] = close.rolling(w).min()

    df["target"] = close.shift(-forecast_horizon)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(market_config, forecast_config):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Download + training...")

    raw = yf.download(
        market_config.ticker,
        period=market_config.period,
        interval=market_config.interval,
        auto_adjust=True,
        progress=False,
    )
    raw.dropna(inplace=True)

    df = build_features(
        raw,
        n_lags=forecast_config.n_lags,
        forecast_horizon=forecast_config.forecast_horizon,
    )

    drop_cols = ["target", "Close", "Open", "High", "Low", "Volume", "hour", "minute", "dayofweek"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target"]

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    tscv = TimeSeriesSplit(n_splits=5)
    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_estimators": 1000,
    }

    best_iters = []
    for tr_idx, val_idx in tscv.split(X_train):
        model_cv = lgb.LGBMRegressor(**lgb_params)
        model_cv.fit(
            X_train.iloc[tr_idx],
            y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        best_iters.append(model_cv.best_iteration_)

    final_n = int(np.mean(best_iters) * forecast_config.cv_boost_factor)
    params_final = {**lgb_params, "n_estimators": final_n}
    params_final.pop("metric", None)

    model = lgb.LGBMRegressor(**params_final)
    model.fit(X_train, y_train)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Modello pronto (n_estimators={final_n})")
    return model, feature_cols, raw


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def log_prediction(last_close, predicted_price, last_candle_time, log_file, forecast_config, market_config):
    change_pct = (predicted_price - last_close) / last_close * 100
    direction = "+" if change_pct > 0 else "-"

    now = datetime.now()

    line = (
        f"{now:%Y-%m-%d %H:%M:%S} | "
        f"Candle: {last_candle_time} | "
        f"Last Close: ${last_close:.2f} | "
        f"Predicted: ${predicted_price:.2f} | "
        f"Change: {direction}{abs(change_pct):.2f}%"
    )

    if 1 == 2:
        with open(log_file, "a") as f:
            f.write(line + "\n")

    decoder = {"+": "up", "-": "down"}

    return {
        "horizon_steps": forecast_config.forecast_horizon,
        "direction": decoder.get(direction),
        "expected_return": f"{abs(change_pct):.2f}%",
        "next_candle": predicted_price,
        "timeframe": market_config.interval,
        "last_price": last_close,
        "ticker": market_config.ticker,
    }


# ─────────────────────────────────────────────
# PREDICTION JOB
# ─────────────────────────────────────────────
def prediction_job(market_config, forecast_config, log_config):
    try:
        if _state["model"] is None:
            _state["model"], _state["feature_cols"], _ = train_model(
                market_config,
                forecast_config,
            )

        raw = yf.download(
            market_config.ticker,
            period="5d", # Da capire se differisce da quello sopra
            interval=market_config.interval,
            auto_adjust=True,
            progress=False,
        )
        raw.dropna(inplace=True)

        df = build_features(
            raw,
            n_lags=forecast_config.n_lags,
            forecast_horizon=forecast_config.forecast_horizon,
        )

        last_row = df[_state["feature_cols"]].iloc[[-1]]
        last_close = float(df["Close"].iloc[-1])
        last_candle_ts = df.index[-1]
        predicted_price = float(_state["model"].predict(last_row)[0])

        return log_prediction(
            last_close=last_close,
            predicted_price=predicted_price,
            last_candle_time=last_candle_ts,
            log_file=log_config.log_file,
            forecast_config=forecast_config,
            market_config=market_config,
        )

    except Exception as e:
        err_line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ERRORE: {e}"
        with open(log_config.log_file, "a") as f:
            f.write(err_line + "\n")
        return None


# ─────────────────────────────────────────────
# RETRAIN JOB
# ─────────────────────────────────────────────
def retrain_job(market_config, forecast_config):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Re-training giornaliero...")
    _state["model"], _state["feature_cols"], _ = train_model(
        market_config,
        forecast_config,
    )


# ─────────────────────────────────────────────
# INDICATORI QUANTITATIVI
# ─────────────────────────────────────────────
def get_quant_index(raw, forecast_config):
    df = build_features(
        raw,
        n_lags=forecast_config.n_lags,
        forecast_horizon=forecast_config.forecast_horizon,
    )

    last = df.iloc[-1]
    return {
        "log_return": round(float(last["log_return"]), 6),
        "rsi": round(float(last["rsi"]), 2),
        "macd_diff": round(float(last["macd_diff"]), 4),
        "macd_signal": round(float(last["macd_signal"]), 4),
        "vol_ma_ratio": round(float(last["vol_ma_ratio"]), 3),
    }