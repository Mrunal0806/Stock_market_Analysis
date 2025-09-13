import os
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Directory to save model
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "stock_classifier.joblib")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# Tickers
TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# -----------------------------
# Feature Engineering Function
# -----------------------------
def add_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Return"] = df["Close"].pct_change()

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD and Signal
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["Bollinger_Upper"] = ma20 + 2 * std20
    df["Bollinger_Lower"] = ma20 - 2 * std20

    # Volatility
    df["Volatility"] = df["Close"].rolling(5).std()

    # Lag features
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Target: 1 if next close > current close
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()

# -----------------------------
# Training Function
# -----------------------------
def train_model():
    all_data = []

    for ticker in TICKERS:
        try:
            print(f"⬇️ Downloading {ticker} ...")
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            df_feat = add_features(df)
            if not df_feat.empty:
                df_feat["Symbol"] = ticker
                all_data.append(df_feat)
            else:
                print(f"⚠️ Skipping {ticker}, no features generated.")
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to error: {e}")

    if not all_data:
        raise ValueError("🚨 No data available for training!")

    data = pd.concat(all_data, ignore_index=True)

    FEATURES = [
        "Return", "RSI", "MACD", "Signal",
        "Bollinger_Upper", "Bollinger_Lower", "Volatility",
        "Open_lag1", "High_lag1", "Low_lag1", "Close_lag1", "Volume_lag1"
    ]

    X = data[FEATURES]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training Random Forest Classifier ...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model and metadata
    model.feature_names_in_ = FEATURES
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "tickers": TICKERS,
        "features": FEATURES,
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH
    }
    pd.Series(metadata).to_json(META_PATH)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"🗂️ Metadata saved to {META_PATH}")

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    train_model()