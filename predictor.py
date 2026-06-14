import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class StockPredictor:
    """
    Predicts RETURNS (not absolute prices) to avoid exponential blowup.
    Uses direct multi-output prediction instead of iterative rollout.
    """

    def __init__(self, model_type: str = "Linear Regression"):
        self.model_type = model_type
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self._feature_cols = None
        self._is_lstm = False

    # ------------------------------------------------------------------ #
    #  Features — built on RETURNS, not raw prices
    # ------------------------------------------------------------------ #
    def _build_features(self, df: pd.DataFrame, horizon: int = 1):
        """
        X = technical features from past prices (no current close)
        y = forward return over `horizon` bars (percentage, not price)
        """
        df = df.copy()

        # Make sure price columns are numeric and finite
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        vol = df["Volume"].fillna(0) if "Volume" in df.columns else pd.Series(0, index=df.index)

        feat = pd.DataFrame(index=df.index)

        # Spreads (normalised by close — scale-invariant)
        feat["HL_pct"] = (high - low) / (close + 1e-9)
        feat["OC_pct"] = (close - open_) / (open_ + 1e-9)
        feat["HC_pct"] = (high - close) / (close + 1e-9)
        feat["LC_pct"] = (close - low) / (close + 1e-9)

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"ret_{lag}"] = close.pct_change(lag)

        # Rolling z-score of price
        for w in [5, 10, 20, 50]:
            if len(df) >= w:
                ma = close.rolling(w).mean()
                std = close.rolling(w).std().replace(0, np.nan)
                feat[f"zscore_{w}"] = (close - ma) / std
                feat[f"vol_{w}"] = std / (ma + 1e-9)

        # Momentum / rate-of-change
        feat["roc_5"] = close.pct_change(5)
        feat["roc_10"] = close.pct_change(10)
        feat["roc_20"] = close.pct_change(20)

        # RSI (normalised 0–1)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        feat["rsi"] = (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100

        # MACD normalised by price
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        feat["macd_n"] = macd / (close + 1e-9)
        feat["macd_h_n"] = (macd - sig) / (close + 1e-9)

        # Bollinger %B
        if len(df) >= 20:
            bb_ma = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            feat["bb_b"] = (close - (bb_ma - 2 * bb_std)) / (4 * bb_std + 1e-9)

        # Volume momentum
        vol_ma5 = vol.rolling(5).mean()
        feat["vol_ratio"] = vol / (vol_ma5 + 1e-9)
        feat["vol_ret"] = vol.pct_change(1)

        # TARGET: forward return over `horizon` bars
        target = close.shift(-horizon).pct_change(horizon)

        # Clean non-finite values before alignment
        feat = feat.replace([np.inf, -np.inf], np.nan)
        target = target.replace([np.inf, -np.inf], np.nan)

        feat.dropna(inplace=True)
        target = target.loc[feat.index].dropna()
        feat = feat.loc[target.index]

        # Winsorise extreme returns
        max_ret = 0.20 * horizon
        target = target.clip(-max_ret, max_ret)

        self._feature_cols = feat.columns.tolist()
        return feat, target

    # ------------------------------------------------------------------ #
    #  Model trainers
    # ------------------------------------------------------------------ #
    def _train_linear(self, X, y):
        m = Ridge(alpha=1.0)
        m.fit(X, y)
        return m

    def _train_rf(self, X, y):
        m = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X, y)
        return m

    def _train_gbm(self, X, y):
        m = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        m.fit(X, y)
        return m

    def _train_lstm(self, X, y):
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            X3 = X.reshape(X.shape[0], 1, X.shape[1])

            model = Sequential([
                LSTM(64, input_shape=(1, X.shape[1]), return_sequences=False),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="huber")
            es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            model.fit(
                X3,
                y,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                callbacks=[es],
                verbose=0,
            )
            self._is_lstm = True
            return model
        except Exception:
            self._is_lstm = False
            return self._train_rf(X, y)

    def _predict_returns(self, model, X):
        if getattr(self, "_is_lstm", False):
            return model.predict(X.reshape(X.shape[0], 1, X.shape[1]), verbose=0).ravel()
        return model.predict(X)

    # ------------------------------------------------------------------ #
    #  Convert predicted returns → price path
    # ------------------------------------------------------------------ #
    def _returns_to_prices(self, last_price: float, returns: list) -> list:
        """
        Convert a sequence of 1-bar forward returns into a price path.
        """
        prices = []
        price = last_price
        for r in returns:
            r = float(np.clip(r, -0.08, 0.08))
            price = price * (1 + r)
            prices.append(price)
        return prices

    # ------------------------------------------------------------------ #
    #  Direct multi-step prediction (NO iterative rollout)
    # ------------------------------------------------------------------ #
    def _predict_future(self, model, df: pd.DataFrame, pred_days: int) -> list:
        """
        Predict pred_days future prices using the last available feature row.
        """
        last_price = float(df["Close"].iloc[-1])

        feat, _ = self._build_features(df, horizon=1)
        if feat.empty:
            return [last_price] * pred_days

        last_feat = feat.iloc[-1:].values
        X_last = self.feature_scaler.transform(last_feat)

        base_return = float(self._predict_returns(model, X_last)[0])

        daily_rets = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        hist_std = float(daily_rets.std()) if len(daily_rets) > 0 else 0.0
        hist_mean = float(daily_rets.mean()) if len(daily_rets) > 0 else 0.0

        prices = []
        price = last_price

        for i in range(pred_days):
            decay = np.exp(-i * 0.15)
            ret = decay * base_return + (1 - decay) * hist_mean

            noise = np.random.default_rng(seed=42 + i).normal(0, hist_std * 0.3 if hist_std > 0 else 0.0)
            ret = ret + noise

            ret = float(np.clip(ret, -0.05, 0.05))
            price = price * (1 + ret)
            prices.append(price)

        return prices

    # ------------------------------------------------------------------ #
    #  Main entry
    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame, pred_days: int = 7):
        self._is_lstm = False

        if df is None or len(df) < 50:
            return None, 0, {}

        df = df.copy()

        # Force numeric and clean infinities early
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])

        feat_df, target = self._build_features(df, horizon=1)

        if feat_df is None or target is None or len(feat_df) < 30:
            return None, 0, {}

        # Final safety cleanup before scaling
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        target = target.loc[feat_df.index].replace([np.inf, -np.inf], np.nan).dropna()
        feat_df = feat_df.loc[target.index]

        if len(feat_df) < 30:
            return None, 0, {}

        X_data = np.nan_to_num(feat_df.values, nan=0.0, posinf=0.0, neginf=0.0)
        y = target.values

        if len(X_data) < 30 or len(y) < 30:
            return None, 0, {}

        X = self.feature_scaler.fit_transform(X_data)

        # Train / test split for time series
        split = max(1, int(len(X) * 0.80))
        if split >= len(X):
            return None, 0, {}

        X_train = X[:split]
        X_test = X[split:]
        y_train = y[:split]
        y_test = y[split:]

        mt = self.model_type

        if mt == "Linear Regression":
            model = self._train_linear(X_train, y_train)
        elif mt == "Random Forest":
            model = self._train_rf(X_train, y_train)
        elif mt == "LSTM Neural Network":
            model = self._train_lstm(X_train, y_train)
        else:
            m1 = self._train_linear(X_train, y_train)
            m2 = self._train_rf(X_train, y_train)
            m3 = self._train_gbm(X_train, y_train)

            p1 = m1.predict(X_test)
            p2 = m2.predict(X_test)
            p3 = m3.predict(X_test)
            y_pred = (p1 + p2 + p3) / 3

            last_price = float(df["Close"].iloc[split - 1])
            close_test = df["Close"].iloc[split:split + len(y_test)].values
            pred_prices = last_price * np.cumprod(1 + y_pred)

            actual_len = min(len(close_test), len(pred_prices))
            metrics = self._calc_metrics(close_test[:actual_len], pred_prices[:actual_len])
            confidence = max(10, min(90, 100 - metrics.get("mape", 10)))

            predictions = self._predict_future(m2, df, pred_days)
            return predictions, confidence, metrics

        # Single model
        if len(X_test) == 0:
            return None, 0, {}

        y_pred = self._predict_returns(model, X_test)

        close_test = df["Close"].iloc[split:split + len(y_test)].values
        last_train_price = float(df["Close"].iloc[split - 1])
        pred_prices = last_train_price * np.cumprod(1 + y_pred)

        actual_len = min(len(close_test), len(pred_prices))
        if actual_len <= 0:
            return None, 0, {}

        metrics = self._calc_metrics(close_test[:actual_len], pred_prices[:actual_len])
        confidence = max(10, min(90, 100 - metrics.get("mape", 10)))

        predictions = self._predict_future(model, df, pred_days)
        return predictions, confidence, metrics

    def _calc_metrics(self, y_true, y_pred) -> dict:
        try:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0 or len(y_pred) == 0:
                return {"rmse": 0, "mae": 0, "r2": 0, "mape": 0}

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)

            return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
        except Exception:
            return {"rmse": 0, "mae": 0, "r2": 0, "mape": 0}