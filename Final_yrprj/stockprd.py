# =============================================================================
#  LONG-TERM STOCK PRICE FORECASTING USING TRANSFORMER ARCHITECTURE
#  Final Semester CS Project
#  Supports multiple companies with case-insensitive name input
# =============================================================================

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
#  COMPANY NAME -> TICKER MAPPING
#  Supports any variation of spelling/case e.g. apple, Apple, APPLE, aapl
# =============================================================================
COMPANY_MAP = {
    # Apple
    "apple": "AAPL", "aapl": "AAPL",

    # NVIDIA
    "nvidia": "NVDA", "nvda": "NVDA", "nvdia": "NVDA",

    # Microsoft
    "microsoft": "MSFT", "msft": "MSFT",

    # Google / Alphabet
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL", "goog": "GOOGL",

    # Amazon
    "amazon": "AMZN", "amzn": "AMZN",

    # Meta / Facebook
    "meta": "META", "facebook": "META", "fb": "META",

    # Tesla
    "tesla": "TSLA", "tsla": "TSLA",

    # Samsung
    "samsung": "005930.KS",

    # Netflix
    "netflix": "NFLX", "nflx": "NFLX",

    # Intel
    "intel": "INTC", "intc": "INTC",

    # AMD
    "amd": "AMD", "advanced micro devices": "AMD",

    # Qualcomm
    "qualcomm": "QCOM", "qcom": "QCOM",

    # IBM
    "ibm": "IBM",

    # Oracle
    "oracle": "ORCL", "orcl": "ORCL",

    # Salesforce
    "salesforce": "CRM", "crm": "CRM",

    # PayPal
    "paypal": "PYPL", "pypl": "PYPL",

    # Uber
    "uber": "UBER",

    # Twitter / X
    "twitter": "TWTR", "x": "TWTR",

    # Spotify
    "spotify": "SPOT", "spot": "SPOT",

    # Zoom
    "zoom": "ZM", "zm": "ZM",

    # Infosys
    "infosys": "INFY", "infy": "INFY",

    # TCS (Tata Consultancy Services)
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",

    # Wipro
    "wipro": "WIPRO.NS",

    # Reliance
    "reliance": "RELIANCE.NS",

    # HDFC Bank
    "hdfc": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
}

def resolve_ticker(user_input: str) -> str:
    """
    Converts any company name or ticker (any case) to the correct ticker symbol.
    Examples:
        'apple'  -> 'AAPL'
        'APPLE'  -> 'AAPL'
        'Apple'  -> 'AAPL'
        'NVDA'   -> 'NVDA'
        'nvidia' -> 'NVDA'
    """
    cleaned = user_input.strip().lower()   # normalize to lowercase

    if cleaned in COMPANY_MAP:
        ticker = COMPANY_MAP[cleaned]
        return ticker
    else:
        ticker = user_input.strip().upper()
        return ticker


def get_user_input() -> dict:
    """
    Interactively ask the user which company and date range to use.
    User only needs to type the company name - no ticker knowledge needed.
    """
    print("\n" + "=" * 60)
    print("   STOCK PRICE FORECASTING - TRANSFORMER MODEL")
    print("=" * 60)
    print("\nYou can type the company name in ANY way:")
    print("  'apple' or 'Apple' or 'APPLE' -> all work the same!")
    print("\nSupported companies:")
    print("  Apple, NVIDIA, Microsoft, Google, Amazon, Meta,")
    print("  Tesla, Netflix, Intel, AMD, IBM, Oracle, Uber,")
    print("  Zoom, Spotify, PayPal, Qualcomm, Salesforce,")
    print("  Infosys, TCS, Wipro, Reliance, HDFC Bank\n")

    company_input = input("Which company do you want to predict? : ").strip()

    # Show error and re-ask if user types nothing
    while not company_input:
        print("[!] Please type a company name.")
        company_input = input("Which company do you want to predict? : ").strip()

    ticker = resolve_ticker(company_input)

    print(f"\nGreat! We will predict stock prices for: {company_input.title()}")
    print(f"Using dates: 2018-01-01 to 2024-12-31 (default)\n")

    change_dates = input("Do you want to change the date range? (yes/no) [default: no]: ").strip().lower()

    if change_dates in ("yes", "y"):
        start_date = input("  Enter start date (YYYY-MM-DD): ").strip()
        if not start_date:
            start_date = "2018-01-01"
        end_date = input("  Enter end date   (YYYY-MM-DD): ").strip()
        if not end_date:
            end_date = "2024-12-31"
    else:
        start_date = "2018-01-01"
        end_date   = "2024-12-31"

    return {"ticker": ticker, "start_date": start_date, "end_date": end_date}


# =============================================================================
#  SECTION 0 - GLOBAL CONFIGURATION
# =============================================================================
def build_config(ticker, start_date, end_date) -> dict:
    return {
        "ticker":           ticker,
        "start_date":       start_date,
        "end_date":         end_date,
        "target_col":       "Close",
        "lookback":         60,
        "forecast_horizon": 1,
        "train_ratio":      0.80,

        "d_model":          64,
        "nhead":            4,
        "num_layers":       2,
        "dim_feedforward":  128,
        "dropout":          0.1,

        "epochs":           50,
        "batch_size":       32,
        "learning_rate":    1e-3,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# =============================================================================
#  SECTION 1 - DATA LOADING
# =============================================================================
def load_stock_data(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run:  pip install yfinance")

    print(f"\n[DATA]  Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No data found for '{ticker}'.\n"
            f"Check the ticker is correct at finance.yahoo.com"
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    print(f"[DATA]  Loaded {len(df)} trading days  |  "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


# =============================================================================
#  SECTION 2 - PREPROCESSING
# =============================================================================
def preprocess(df, target_col="Close"):
    df = df.dropna()
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    data         = df[feature_cols].values.astype(np.float32)

    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    target_idx  = feature_cols.index(target_col)

    print(f"[PREP]  Normalized {len(data)} rows  |  Target: {target_col} (index {target_idx})")
    return data_scaled, scaler, target_idx, df.index


# =============================================================================
#  SECTION 3 - SLIDING WINDOW
# =============================================================================
def create_sequences(data, lookback, target_idx, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback + horizon - 1, target_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[SEQ]   Created {len(X)} sequences  |  X: {X.shape}  y: {y.shape}")
    return X, y


# =============================================================================
#  SECTION 4 - PYTORCH DATASET
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
#  SECTION 5 - POSITIONAL ENCODING
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
#  SECTION 6 - TRANSFORMER MODEL
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.output_head(x).squeeze(-1)


# =============================================================================
#  SECTION 7 - TRAINING
# =============================================================================
def train_model(model, train_loader, val_loader, cfg):
    device    = cfg["device"]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    train_losses, val_losses = [], []
    print(f"\n[TRAIN] Starting {cfg['epochs']} epochs on {device} ...")

    for epoch in range(1, cfg["epochs"] + 1):
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

        model.eval()
        val_batch = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_batch.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_batch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg['epochs']}  |  "
                  f"Train: {train_loss:.6f}  |  Val: {val_loss:.6f}")

    print("[TRAIN] Done.")
    return train_losses, val_losses


# =============================================================================
#  SECTION 8 - METRICS
# =============================================================================
def evaluate_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    print("\n[METRICS]  Test Set Results:")
    print(f"   MAE   = ${mae:.4f}")
    print(f"   RMSE  = ${rmse:.4f}")
    print(f"   MAPE  = {mape:.4f}%")
    return metrics


# =============================================================================
#  SECTION 9 - INFERENCE
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
#  SECTION 10 - VISUALIZATIONS
# =============================================================================
def plot_loss_curves(train_losses, val_losses, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(val_losses,   label="Val Loss",   color="#FF5722", linewidth=2, linestyle="--")
    ax.set_title(f"{ticker} - Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved: {path}")


def plot_actual_vs_predicted(dates, actual, predicted, ticker, metrics):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual,    label="Actual Price",    color="#1565C0", linewidth=1.5)
    ax.plot(dates, predicted, label="Predicted Price", color="#E53935",
            linewidth=1.5, linestyle="--", alpha=0.85)
    ax.set_title(
        f"{ticker} - Actual vs Predicted  "
        f"(MAE=${metrics['MAE']:.2f}  RMSE=${metrics['RMSE']:.2f}  MAPE={metrics['MAPE (%)']:.2f}%)",
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
    print(f"[PLOT]  Saved: {path}")


def plot_scatter(actual, predicted, ticker):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.4, s=10, color="#7B1FA2")
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title(f"{ticker} - Scatter: Actual vs Predicted", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]  Saved: {path}")


# =============================================================================
#  SECTION 11 - MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg):
    ticker   = cfg["ticker"]
    lookback = cfg["lookback"]
    device   = cfg["device"]

    print(f"\n[CONFIG]  Ticker: {ticker}  |  Device: {device}")

    df = load_stock_data(ticker, cfg["start_date"], cfg["end_date"])

    data_scaled, scaler, target_idx, dates = preprocess(df, cfg["target_col"])
    num_features = data_scaled.shape[1]

    X, y      = create_sequences(data_scaled, lookback, target_idx,
                                  cfg["forecast_horizon"])
    seq_dates = dates[lookback + cfg["forecast_horizon"] - 1:]

    split           = int(len(X) * cfg["train_ratio"])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test      = seq_dates[split:]

    val_split    = int(len(X_train) * 0.85)
    X_tr, X_val  = X_train[:val_split], X_train[val_split:]
    y_tr, y_val  = y_train[:val_split], y_train[val_split:]

    print(f"\n[SPLIT]  Train: {len(X_tr)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    train_loader = DataLoader(StockDataset(X_tr, y_tr),
                              batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(StockDataset(X_val, y_val), batch_size=cfg["batch_size"])
    test_loader  = DataLoader(StockDataset(X_test, y_test), batch_size=cfg["batch_size"])

    model = StockTransformer(
        num_features=num_features,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        seq_len=lookback
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL]  Trainable parameters: {params:,}")

    train_losses, val_losses = train_model(model, train_loader, val_loader, cfg)

    preds_scaled, actuals_scaled = get_predictions(model, test_loader, device)
    preds_real   = inverse_transform_target(preds_scaled,   scaler, target_idx, num_features)
    actuals_real = inverse_transform_target(actuals_scaled, scaler, target_idx, num_features)

    metrics = evaluate_metrics(actuals_real, preds_real)

    print("\n[VIZ]   Generating plots ...")
    plot_loss_curves(train_losses, val_losses, ticker)
    plot_actual_vs_predicted(dates_test[:len(preds_real)],
                             actuals_real, preds_real, ticker, metrics)
    plot_scatter(actuals_real, preds_real, ticker)

    weights_path = os.path.join(OUTPUT_DIR, f"{ticker}_transformer_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"\n[SAVE]  Weights saved -> {weights_path}")

    return metrics


def main():
    while True:
        # Ask user which company to predict
        user_input = get_user_input()
        cfg        = build_config(
            user_input["ticker"],
            user_input["start_date"],
            user_input["end_date"]
        )

        try:
            run_pipeline(cfg)
        except ValueError as e:
            print(f"\n[ERROR]  {e}")

        print("\n" + "=" * 60)
        again = input("Run prediction for another company? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nThank you! Exiting.")
            break


if __name__ == "__main__":
    main()
