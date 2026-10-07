"""Central configuration, read from environment variables (see .env.example)."""
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
KAFKA_GROUP = os.getenv("KAFKA_GROUP", "fraud-scorer")

CASSANDRA_HOSTS = [h.strip() for h in os.getenv("CASSANDRA_HOSTS", "localhost").split(",")]
CASSANDRA_PORT = _int("CASSANDRA_PORT", 9042)
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "fraud")

PRODUCER_RATE = _int("PRODUCER_RATE", 40)
DATA_FILE = os.getenv("DATA_FILE", "data/paysim.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_forest.joblib")
METRICS_PORT = _int("METRICS_PORT", 8000)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
