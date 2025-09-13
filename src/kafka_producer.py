# kafka_producer.py
import json
import time
import yfinance as yf
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# -------------------------
# Kafka Setup
# -------------------------
BOOTSTRAP_SERVERS = ["localhost:9092", "host.docker.internal:9092"]
TOPIC = "stock_prices"

producer = None
for server in BOOTSTRAP_SERVERS:
    try:
        producer = KafkaProducer(
            bootstrap_servers=server,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"✅ Connected to Kafka at {server}")
        break
    except NoBrokersAvailable:
        print(f"❌ Could not connect to {server}, trying next...")

if not producer:
    raise Exception("🚨 Could not connect to any Kafka broker!")

# -------------------------
# Symbols to Track
# -------------------------
symbols = ["AAPL", "MSFT", "TSLA", "GOOG"]

print(f"🚀 Producer started... streaming {symbols} to topic: {TOPIC}")

# -------------------------
# Main Loop
# -------------------------
while True:
    for sym in symbols:
        try:
            # Fetch last 2 days of 1m candles
            df = yf.download(sym, period="2d", interval="1m", progress=False)

            if not df.empty:
                # Last row
                last_index = df.index[-1]  # pandas.Timestamp
                last_row = df.iloc[-1]

                message = {
                    "symbol": sym,
                    "ts": str(last_index),   # standardized timestamp key
                    "open": float(last_row["Open"]),
                    "high": float(last_row["High"]),
                    "low": float(last_row["Low"]),
                    "close": float(last_row["Close"]),
                    "volume": int(last_row["Volume"]),
                }

                producer.send(TOPIC, value=message)
                print(f"📤 Sent: {message}")

        except Exception as e:
            print(f"⚠️ Error for {sym}: {e}")

    # wait before next fetch
    time.sleep(60)  # 1 minute
 # fetch every 1 minute
