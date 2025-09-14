import json
import time
import yfinance as yf
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable, KafkaTimeoutError

# -------------------------
# Kafka Configuration
# -------------------------
# If Kafka is running via Docker Compose, use localhost:29092
BOOTSTRAP_SERVERS = ["localhost:29092", "host.docker.internal:29092"]
TOPIC = "stock_prices"

# -------------------------
# Initialize Kafka Producers
# -------------------------
producer = None
for server in BOOTSTRAP_SERVERS:
    try:
        producer = KafkaProducer(
            bootstrap_servers=server,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            linger_ms=10,
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
# Main Streaming Loop
# -------------------------
while True:
    for sym in symbols:
        try:
            # Fetch last 2 days of 1-minute candles
            df = yf.download(sym, period="2d", interval="1m", progress=False, auto_adjust=False)

            if not df.empty:
                last_index = df.index[-1]
                last_row = df.iloc[-1]

                message = {
                    "symbol": sym,
                    "ts": str(last_index),
                    "open": float(last_row["Open"]),
                    "high": float(last_row["High"]),
                    "low": float(last_row["Low"]),
                    "close": float(last_row["Close"]),
                    "volume": int(last_row["Volume"]),
                }

                try:
                    producer.send(TOPIC, value=message)
                    producer.flush()
                    print(f"📤 Sent: {message}")
                except KafkaTimeoutError as e:
                    print(f"⚠️ Kafka timeout while sending {sym}: {e}")

        except Exception as e:
            print(f"⚠️ Error for {sym}: {e}")

    time.sleep(60)  # Wait 1 minute before next fetch
