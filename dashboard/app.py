"""Streamlit live fraud-alert feed."""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import config  # noqa: E402
from common.cassandra_store import hour_bucket  # noqa: E402

st.set_page_config(page_title="Fraud Alerts", page_icon="🚨", layout="wide")


@st.cache_resource
def get_session():
    from cassandra.cluster import Cluster
    cluster = Cluster(config.CASSANDRA_HOSTS, port=config.CASSANDRA_PORT)
    return cluster.connect(config.CASSANDRA_KEYSPACE)


def load_alerts(limit_per_bucket: int = 300) -> pd.DataFrame:
    session = get_session()
    now = datetime.now(timezone.utc)
    frames = []
    for h in range(3):  # current hour + previous two
        b = hour_bucket(now - timedelta(hours=h))
        rows = session.execute(
            "SELECT event_time, txn_id, type, amount, name_orig, name_dest, anomaly_score, label "
            "FROM fraud_alerts WHERE hour_bucket=%s LIMIT %s", (b, limit_per_bucket))
        frames.append(pd.DataFrame(list(rows)))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_values("event_time", ascending=False).reset_index(drop=True)
    return df


def prom(query: str):
    try:
        r = requests.get(f"{config.PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        res = r.json()["data"]["result"]
        return float(res[0]["value"][1]) if res else 0.0
    except Exception:
        return None


st.title("🚨 Real-time fraud alerts")
refresh = st.sidebar.slider("Refresh every (seconds)", 2, 30, 5)
st.sidebar.caption("Alerts come from Cassandra; totals from Prometheus.")

processed = prom("sum(fraud_transactions_processed_total)")
flagged = prom("sum(fraud_alerts_total)")
amount = prom("sum(fraud_flagged_amount_total)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scored", f"{int(processed):,}" if processed is not None else "n/a")
c2.metric("Alerts raised", f"{int(flagged):,}" if flagged is not None else "n/a")
c3.metric("Alert rate", f"{100 * flagged / processed:.2f}%" if processed else "n/a")
c4.metric("Flagged amount", f"{amount:,.0f}" if amount is not None else "n/a")

try:
    alerts = load_alerts()
except Exception as exc:
    st.warning(f"Cassandra not reachable yet: {exc}")
    alerts = pd.DataFrame()

if alerts.empty:
    st.info("No alerts yet. Is the producer running? (.\\run.ps1 start)")
else:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Latest alerts")
        view = alerts.head(100).copy()
        view["correct"] = view["label"].map({1: "✔ real fraud", 0: "✖ false alarm"})
        st.dataframe(view.drop(columns=["label"]), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Alerts by transaction type")
        st.bar_chart(alerts["type"].value_counts())
        st.subheader("Precision on recent alerts")
        st.metric("Real fraud among alerts", f"{100 * (alerts['label'] == 1).mean():.1f}%")

time.sleep(refresh)
st.rerun()
