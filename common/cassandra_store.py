"""Cassandra connection, schema and row writer."""
import time
from datetime import datetime, timezone

from common import config
from common.logging_setup import get_logger

log = get_logger("cassandra_store")

COLUMNS = [
    "hour_bucket", "event_time", "txn_id", "step", "type", "amount",
    "name_orig", "name_dest", "old_balance_org", "new_balance_org",
    "old_balance_dest", "new_balance_dest", "anomaly_score", "predicted_fraud",
    "label", "processed_at",
]

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {ks}.{table} (
    hour_bucket text,
    event_time timestamp,
    txn_id text,
    step int,
    type text,
    amount double,
    name_orig text,
    name_dest text,
    old_balance_org double,
    new_balance_org double,
    old_balance_dest double,
    new_balance_dest double,
    anomaly_score double,
    predicted_fraud boolean,
    label int,
    processed_at timestamp,
    PRIMARY KEY ((hour_bucket), event_time, txn_id)
) WITH CLUSTERING ORDER BY (event_time DESC, txn_id ASC)
"""


def hour_bucket(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d-%H")


def connect(retries: int = 40):
    """Connect (retrying while Cassandra boots) and make sure the schema exists."""
    from cassandra.cluster import Cluster  # lazy: keeps unit tests free of the driver

    last = None
    for attempt in range(1, retries + 1):
        try:
            cluster = Cluster(config.CASSANDRA_HOSTS, port=config.CASSANDRA_PORT, connect_timeout=10)
            session = cluster.connect()
            ks = config.CASSANDRA_KEYSPACE
            session.execute(
                f"CREATE KEYSPACE IF NOT EXISTS {ks} "
                "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
            session.set_keyspace(ks)
            for table in ("scored_transactions", "fraud_alerts"):
                session.execute(_TABLE_DDL.format(ks=ks, table=table))
            log.info("Connected to Cassandra, schema ready")
            return cluster, session
        except Exception as exc:  # driver raises several exception types while booting
            last = exc
            log.warning("Cassandra not ready (attempt %d/%d): %s", attempt, retries, str(exc)[:120])
            time.sleep(4)
    raise RuntimeError(f"Could not connect to Cassandra: {last}")


class Writer:
    def __init__(self, session):
        cols = ", ".join(COLUMNS)
        marks = ", ".join(["?"] * len(COLUMNS))
        self.session = session
        self.insert_all = session.prepare(f"INSERT INTO scored_transactions ({cols}) VALUES ({marks})")
        self.insert_alert = session.prepare(f"INSERT INTO fraud_alerts ({cols}) VALUES ({marks})")

    def write(self, rows: list, alerts: list) -> None:
        """rows / alerts are lists of tuples in COLUMNS order. Upserts, so retries are safe."""
        from cassandra.concurrent import execute_concurrent_with_args

        for stmt, data in ((self.insert_all, rows), (self.insert_alert, alerts)):
            if not data:
                continue
            results = execute_concurrent_with_args(self.session, stmt, data, concurrency=50, raise_on_first_error=True)
            for ok, res in results:
                if not ok:
                    raise res
