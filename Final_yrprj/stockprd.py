
#  LONG-TERM STOCK PRICE FORECASTING USING TRANSFORMER ARCHITECTURE
#  Final Semester CS Project - Accurate Version with Technical Indicators


import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
#  CONFIGURATION
# =============================================================================
def build_config(ticker):
    return {
        "ticker":           ticker.upper(),
        "start_date":       "2018-01-01",
        "end_date":         "2024-12-31",
        "target_col":       "Close",
        "lookback":         60,
        "forecast_horizon": 1,
        "train_ratio":      0.80,

        # Bigger model = more accurate
        "d_model":          128,
        "nhead":            8,
        "num_layers":       3,
        "dim_feedforward":  256,
        "dropout":          0.1,

        # More epochs = better learning
        "epochs":           100,
        "batch_size":       32,
        "learning_rate":    1e-3,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# =============================================================================
#  STEP 1 - ASK FOR TICKER
# =============================================================================
def get_ticker():
    print("\n" + "=" * 60)
    print("   STOCK PRICE FORECASTING - TRANSFORMER MODEL")
    print("=" * 60)
    print("\n  Example tickers:")
    print("  AAPL        -> Apple")
    print("  NVDA        -> NVIDIA")
    print("  MSFT        -> Microsoft")
    print("  TSLA        -> Tesla")
    print("  GOOGL       -> Google")
    print("  AMZN        -> Amazon")
    print("  TCS.NS      -> TCS (India)")
    print("  INFY        -> Infosys")
    print("  RELIANCE.NS -> Reliance (India)")
    print("\n  Find any ticker at: finance.yahoo.com\n")

    ticker = input("  Enter ticker: ").strip().upper()
    while not ticker:
        print("  [!] Cannot be empty. Try again.")
        ticker = input("  Enter ticker: ").strip().upper()
    return ticker


# =============================================================================
#  STEP 2 - DOWNLOAD DATA
# =============================================================================
def load_stock_data(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    print(f"\n[DATA]  Downloading {ticker} ...")

    # Try up to 3 times - yfinance sometimes fails on first attempt
    df = None
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False, timeout=30)
            if df is not None and not df.empty:
                break
            print(f"[DATA]  Attempt {attempt} returned empty. Retrying ...")
        except Exception as e:
            print(f"[DATA]  Attempt {attempt} failed ({e}). Retrying ...")

    if df is None or df.empty:
        raise ValueError(
            f"No data found for '{ticker}' after 3 attempts.\n"
            f"  Fixes:\n"
            f"  1. Run: pip install --upgrade yfinance\n"
            f"  2. Check ticker at finance.yahoo.com\n"
            f"  3. Wait a few seconds and try again"
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    print(f"[DATA]  {len(df)} trading days  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
#  STEP 3 - TECHNICAL INDICATORS
#  These extra features make the model learn company-specific patterns
# =============================================================================
def add_technical_indicators(df):
    """
    Adds 12 technical indicators to raw OHLCV data.
    Each indicator captures a different behaviour of the stock:

    MA7, MA21     - Short & medium term price trends
    MA50          - Long term trend
    EMA12, EMA26  - Exponential moving averages (more weight on recent data)
    MACD          - Momentum: difference between EMA12 and EMA26
    RSI           - Relative Strength Index: measures overbought/oversold (0-100)
    BB_upper      - Bollinger Band upper: price + 2 standard deviations
    BB_lower      - Bollinger Band lower: price - 2 standard deviations
    BB_width      - Width between bands: measures volatility
    Volatility    - 10-day rolling standard deviation of returns
    Return_1d     - Daily percentage return
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Moving Averages
    df["MA7"]   = close.rolling(window=7).mean()
    df["MA21"]  = close.rolling(window=21).mean()
    df["MA50"]  = close.rolling(window=50).mean()

    # Exponential Moving Averages
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df["MACD"]  = df["EMA12"] - df["EMA26"]

    # RSI (Relative Strength Index)
    delta       = close.diff()
    gain        = delta.clip(lower=0).rolling(window=14).mean()
    loss        = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs          = gain / (loss + 1e-9)
    df["RSI"]   = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean     = close.rolling(window=20).mean()
    rolling_std      = close.rolling(window=20).std()
    df["BB_upper"]   = rolling_mean + 2 * rolling_std
    df["BB_lower"]   = rolling_mean - 2 * rolling_std
    df["BB_width"]   = df["BB_upper"] - df["BB_lower"]

    # Volatility (10-day rolling std of daily returns)
    df["Volatility"] = close.pct_change().rolling(window=10).std()

    # Daily Return
    df["Return_1d"]  = close.pct_change()

    # Drop rows with NaN (from rolling calculations)
    df.dropna(inplace=True)

    print(f"[FEAT]  Added technical indicators  |  "
          f"Total features: {len(df.columns)}  |  Rows after cleaning: {len(df)}")
    return df


# =============================================================================
#  STEP 4 - PREPROCESS (Normalize 0 to 1)
# =============================================================================
def preprocess(df, target_col="Close"):
    df = df.dropna()

    # All columns become features
    feature_cols = df.columns.tolist()
    data         = df[feature_cols].values.astype(np.float32)

    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    target_idx  = feature_cols.index(target_col)

    print(f"[PREP]  Normalized {len(data)} rows  |  "
          f"{len(feature_cols)} features  |  Target: {target_col} (col {target_idx})")
    return data_scaled, scaler, target_idx, df.index, len(feature_cols)


# =============================================================================
#  STEP 5 - SLIDING WINDOW
# =============================================================================
def create_sequences(data, lookback, target_idx, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback + horizon - 1, target_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[SEQ]   {len(X)} sequences  |  Shape: {X.shape}")
    return X, y


# =============================================================================
#  STEP 6 - PYTORCH DATASET
# =============================================================================
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
#  STEP 7 - POSITIONAL ENCODING
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe           = torch.zeros(max_len, d_model)
        position     = torch.arange(0, max_len).unsqueeze(1)
        div_term     = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
#  STEP 8 - TRANSFORMER MODEL
# =============================================================================
class StockTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers,
                 dim_feedforward, dropout, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 10,
                                             dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Deeper output head for better accuracy
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.output_head(x).squeeze(-1)


# =============================================================================
#  STEP 9 - TRAINING
# =============================================================================
def train_model(model, train_loader, val_loader, cfg):
    device    = cfg["device"]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg["learning_rate"], weight_decay=1e-5)
    criterion = nn.HuberLoss()      # More robust than MSE - less sensitive to outliers
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]  # Smoothly reduces LR over all epochs
    )

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0
    patience_limit = 15   # Stop early if no improvement for 15 epochs

    print(f"\n[TRAIN] Starting {cfg['epochs']} epochs on {device} ...")

    for epoch in range(1, cfg["epochs"] + 1):
        # Training
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        # Validation
        model.eval()
        val_batch = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_batch.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_batch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg['epochs']}  |  "
                  f"Train: {train_loss:.6f}  |  "
                  f"Val: {val_loss:.6f}  |  "
                  f"Best: {best_val_loss:.6f}")

        # Early stopping
        if patience_count >= patience_limit:
            print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience_limit} epochs)")
            break

    # Load best weights before returning
    model.load_state_dict(best_weights)
    print("[TRAIN] Done. Best model weights restored.")
    return train_losses, val_losses


# =============================================================================
#  STEP 10 - METRICS
# =============================================================================
def evaluate_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    print("\n[METRICS]  Test Set Results:")
    print(f"   MAE   = ${mae:.4f}  (avg dollar error)")
    print(f"   RMSE  = ${rmse:.4f}  (penalizes large errors)")
    print(f"   MAPE  = {mape:.4f}%  (percentage error)")
    return metrics


# =============================================================================
#  STEP 11 - INFERENCE
# =============================================================================
def get_predictions(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(device)).cpu().numpy())
            actuals.extend(yb.numpy())
    return np.array(preds), np.array(actuals)


def inverse_transform_target(scaled_values, scaler, target_idx, num_features):
    dummy = np.zeros((len(scaled_values), num_features), dtype=np.float32)
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


# =============================================================================
#  STEP 12 - VISUALIZATIONS (saved with ticker name - unique per company)
# =============================================================================
def plot_loss_curves(train_losses, val_losses, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(val_losses,   label="Val Loss",   color="#FF5722",
            linewidth=2, linestyle="--")
    ax.set_title(f"{ticker} - Training & Validation Loss",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved -> {path}")


def plot_actual_vs_predicted(dates, actual, predicted, ticker, metrics):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual,    label="Actual Price",
            color="#1565C0", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted Price",
            color="#E53935",  linewidth=1.5, linestyle="--", alpha=0.9)

    ax.fill_between(dates, actual, predicted,
                    alpha=0.08, color="purple",
                    label="Prediction Error")

    ax.set_title(
        f"{ticker} - Actual vs Predicted Stock Price\n"
        f"MAE=${metrics['MAE']:.2f}  |  "
        f"RMSE=${metrics['RMSE']:.2f}  |  "
        f"MAPE={metrics['MAPE (%)']:.2f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=35)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved -> {path}")


def plot_scatter(actual, predicted, ticker):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.4, s=10, color="#7B1FA2")
    lim = [min(actual.min(), predicted.min()),
           max(actual.max(), predicted.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title(f"{ticker} - Scatter: Actual vs Predicted",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved -> {path}")


def plot_technical_indicators(df_with_indicators, ticker):
    """
    Extra chart showing the technical indicators used for this company.
    Helps visualize what the model learned from.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Price + Moving Averages
    axes[0].plot(df_with_indicators.index, df_with_indicators["Close"],
                 label="Close", color="#1565C0", linewidth=1)
    axes[0].plot(df_with_indicators.index, df_with_indicators["MA7"],
                 label="MA7",   color="#FF5722", linewidth=1, alpha=0.8)
    axes[0].plot(df_with_indicators.index, df_with_indicators["MA21"],
                 label="MA21",  color="#4CAF50", linewidth=1, alpha=0.8)
    axes[0].plot(df_with_indicators.index, df_with_indicators["MA50"],
                 label="MA50",  color="#9C27B0", linewidth=1, alpha=0.8)
    axes[0].fill_between(df_with_indicators.index,
                         df_with_indicators["BB_upper"],
                         df_with_indicators["BB_lower"],
                         alpha=0.1, color="gray", label="Bollinger Bands")
    axes[0].set_title(f"{ticker} - Price & Moving Averages",
                      fontweight="bold")
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: MACD
    axes[1].plot(df_with_indicators.index, df_with_indicators["MACD"],
                 color="#FF9800", linewidth=1.2, label="MACD")
    axes[1].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title(f"{ticker} - MACD", fontweight="bold")
    axes[1].set_ylabel("MACD")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: RSI
    axes[2].plot(df_with_indicators.index, df_with_indicators["RSI"],
                 color="#E91E63", linewidth=1.2, label="RSI")
    axes[2].axhline(y=70, color="red",   linewidth=0.8,
                    linestyle="--", label="Overbought (70)")
    axes[2].axhline(y=30, color="green", linewidth=0.8,
                    linestyle="--", label="Oversold (30)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title(f"{ticker} - RSI", fontweight="bold")
    axes[2].set_ylabel("RSI")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.xticks(rotation=35)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_technical_indicators.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved -> {path}")


# =============================================================================
#  STEP 13 - MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg):
    ticker   = cfg["ticker"]
    lookback = cfg["lookback"]
    device   = cfg["device"]

    print(f"\n[CONFIG]  Ticker: {ticker}  |  Device: {device}")

    # Download
    df = load_stock_data(ticker, cfg["start_date"], cfg["end_date"])

    # Add technical indicators (this is what makes each company unique)
    df = add_technical_indicators(df)

    # Save the indicator chart
    plot_technical_indicators(df, ticker)

    # Preprocess
    data_scaled, scaler, target_idx, dates, num_features = preprocess(
        df, cfg["target_col"]
    )

    # Sequences
    X, y      = create_sequences(data_scaled, lookback,
                                   target_idx, cfg["forecast_horizon"])
    seq_dates = dates[lookback + cfg["forecast_horizon"] - 1:]

    # Split
    split           = int(len(X) * cfg["train_ratio"])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test      = seq_dates[split:]

    val_split    = int(len(X_train) * 0.85)
    X_tr, X_val  = X_train[:val_split], X_train[val_split:]
    y_tr, y_val  = y_train[:val_split], y_train[val_split:]

    print(f"\n[SPLIT]  Train: {len(X_tr)}  |  "
          f"Val: {len(X_val)}  |  Test: {len(X_test)}")

    train_loader = DataLoader(StockDataset(X_tr, y_tr),
                              batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(StockDataset(X_val, y_val),
                              batch_size=cfg["batch_size"])
    test_loader  = DataLoader(StockDataset(X_test, y_test),
                              batch_size=cfg["batch_size"])

    # Build model
    model = StockTransformer(
        num_features    = num_features,
        d_model         = cfg["d_model"],
        nhead           = cfg["nhead"],
        num_layers      = cfg["num_layers"],
        dim_feedforward = cfg["dim_feedforward"],
        dropout         = cfg["dropout"],
        seq_len         = lookback
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL]  Trainable parameters: {params:,}  |  "
          f"Features used: {num_features}")

    # Train
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, cfg
    )

    # Predict
    preds_scaled, actuals_scaled = get_predictions(model, test_loader, device)
    preds_real   = inverse_transform_target(
        preds_scaled,   scaler, target_idx, num_features
    )
    actuals_real = inverse_transform_target(
        actuals_scaled, scaler, target_idx, num_features
    )

    # Metrics
    metrics = evaluate_metrics(actuals_real, preds_real)

    # Plots
    print("\n[VIZ]   Saving charts ...")
    plot_loss_curves(train_losses, val_losses, ticker)
    plot_actual_vs_predicted(
        dates_test[:len(preds_real)],
        actuals_real, preds_real, ticker, metrics
    )
    plot_scatter(actuals_real, preds_real, ticker)

    # Save model
    weights_path = os.path.join(OUTPUT_DIR, f"{ticker}_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"\n[SAVE]  Model saved -> {weights_path}")
    print("\n" + "=" * 60)
    print(f"  DONE!  4 charts saved for {ticker}:")
    print(f"  1. {ticker}_technical_indicators.png")
    print(f"  2. {ticker}_loss_curves.png")
    print(f"  3. {ticker}_actual_vs_predicted.png")
    print(f"  4. {ticker}_scatter.png")
    print("=" * 60)


def main():
    while True:
        ticker = get_ticker()
        cfg    = build_config(ticker)

        try:
            run_pipeline(cfg)
        except ValueError as e:
            print(f"\n[ERROR]  {e}")

        print()
        again = input("Predict another company? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
