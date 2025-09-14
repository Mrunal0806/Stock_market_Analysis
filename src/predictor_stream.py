import os
import json
import joblib
import datetime
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

# ==============================
# Configuration
# ==============================
MODEL_PATH = "model/stock_classifier.joblib"
META_PATH = "model/metadata.json"
TOPIC_IN = "stock_prices"
TOPIC_OUT = "stock_predictions"
BROKER = "localhost:29092"

# ==============================
# Load Model and Metadata
# ==============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
print(f"✅ Random Forest model loaded with features: {FEATURES}")

# ==============================
# Kafka Setup
# ==============================
consumer = KafkaConsumer(
    TOPIC_IN,
    bootstrap_servers=BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="stock-classifier-group"
)

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print(f"🚀 Streaming classifier started... Listening on topic: {TOPIC_IN}")

# ==============================
# Streaming Loop
# ==============================
history = []

for msg in consumer:
    rec = msg.value
    symbol = rec.get("symbol")
    close_price = rec.get("close")
    ts = rec.get("ts", datetime.datetime.now(datetime.UTC).isoformat())

    print(f"📩 Received {symbol} @ {ts} = {close_price}")

    try:
        df = pd.DataFrame([rec])

        # Feature Engineering
        df["Return"] = df["close"].pct_change().fillna(0)

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["Bollinger_Upper"] = ma20 + 2 * std20
        df["Bollinger_Lower"] = ma20 - 2 * std20

        df["Volatility"] = df["close"].rolling(5).std()

        df["Open_lag1"] = df["open"].shift(1).fillna(df["open"])
        df["High_lag1"] = df["high"].shift(1).fillna(df["high"])
        df["Low_lag1"] = df["low"].shift(1).fillna(df["low"])
        df["Close_lag1"] = df["close"].shift(1).fillna(df["close"])
        df["Volume_lag1"] = df["volume"].shift(1).fillna(df["volume"])

        # Prepare input for prediction
        X = df[FEATURES].tail(1)

        pred_class = int(model.predict(X)[0])
        pred_proba = float(model.predict_proba(X)[0][pred_class])

        prediction = {
            "ts": ts,
            "symbol": symbol,
            "last_close": close_price,
            "predicted_class": pred_class,
            "confidence": round(pred_proba, 4),
            "model_version": "rf_v1"
        }

        producer.send(TOPIC_OUT, prediction)
        print(f"📤 Sent prediction: {prediction}")

    except Exception as e:
        print(f"⚠️ Prediction failed for {symbol}: {e}")