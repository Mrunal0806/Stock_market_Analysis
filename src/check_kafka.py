from kafka import KafkaConsumer
import json

KAFKA_BROKER = "localhost:9092"   # or "host.docker.internal:9092" if Docker
TOPIC = "predictions"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="test-consumer",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print(f"🔍 Listening for messages on topic '{TOPIC}'...")

for msg in consumer:
    print("📩", msg.value)
