from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import AppConfig, load_dotenv
from common.schema import ALERT_BUCKET_MINUTES, alert_bucket

load_dotenv()
CONFIG = AppConfig.from_env()


@st.cache_resource
def _cassandra_session():
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster

    cass = CONFIG.cassandra
    auth = PlainTextAuthProvider(cass.username, cass.password) if cass.auth_required else None
    cluster = Cluster(
        contact_points=cass.contact_points,
        port=cass.port,
        auth_provider=auth,
        connect_timeout=3,
        control_connection_timeout=3,
    )
    return cluster, cluster.connect(cass.keyspace)


def _recent_buckets(lookback_minutes: int, now: datetime | None = None) -> list[str]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    count = max(1, (lookback_minutes + ALERT_BUCKET_MINUTES - 1) // ALERT_BUCKET_MINUTES)
    local = datetime.now().astimezone()
    local_as_utc = datetime(
        local.year, local.month, local.day, local.hour, local.minute,
        tzinfo=timezone.utc,
    )
    buckets = set()
    for moment in (current, local_as_utc):
        start = moment.replace(
            minute=moment.minute // ALERT_BUCKET_MINUTES * ALERT_BUCKET_MINUTES,
            second=0,
            microsecond=0,
        )
        buckets.update(
            alert_bucket(start - timedelta(minutes=i * ALERT_BUCKET_MINUTES))
            for i in range(count)
        )
    return sorted(buckets, reverse=True)


def _read_alerts(lookback_minutes: int, limit: int) -> tuple[pd.DataFrame, str | None]:
    try:
        _, session = _cassandra_session()
        statement = (
            "SELECT alert_id, detected_at, name_orig, name_dest, step, tx_type, amount, "
            "anomaly_score, decision_threshold, model_version, is_fraud_label "
            "FROM fraud_alerts WHERE alert_bucket = %s LIMIT %s"
        )
        rows = []
        for bucket in _recent_buckets(lookback_minutes):
            rows.extend(session.execute(statement, (bucket, limit)))
        records = [
            {
                "alert_id": str(row.alert_id),
                "detected_at": row.detected_at,
                "name_orig": row.name_orig,
                "name_dest": row.name_dest,
                "step": row.step,
                "tx_type": row.tx_type,
                "amount": float(row.amount or 0),
                "anomaly_score": float(row.anomaly_score or 0),
                "decision_threshold": float(row.decision_threshold or 0),
                "model_version": row.model_version or "unknown",
                "is_fraud_label": bool(row.is_fraud_label),
            }
            for row in rows
        ]
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = (
                frame.sort_values("detected_at", ascending=False)
                .drop_duplicates(subset=["alert_id"])
                .head(limit)
                .reset_index(drop=True)
            )
        return frame, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _prometheus_value(query: str) -> float | None:
    url = "http://localhost:9090/api/v1/query?" + urlencode({"query": query})
    try:
        with urlopen(url, timeout=2) as response:
            payload = json.load(response)
        results = payload.get("data", {}).get("result", [])
        return sum(float(item["value"][1]) for item in results) if results else 0.0
    except Exception:
        return None


def _metric(*queries: str) -> float | None:
    for query in queries:
        value = _prometheus_value(query)
        if value is not None:
            return value
    return None


st.set_page_config(page_title="Fraud operations", page_icon=":material/security:", layout="wide")
st.title("Fraud operations")
st.caption("Live transaction scoring, alert activity, and service health")

with st.sidebar:
    st.subheader("View")
    window = st.selectbox("Alert window", ["15 minutes", "1 hour", "6 hours"], index=1)
    window_minutes = {"15 minutes": 15, "1 hour": 60, "6 hours": 360}[window]
    alert_limit = st.slider(
        "Alerts to show", min_value=10, max_value=200,
        value=min(200, max(10, CONFIG.dashboard.alert_limit)),
    )
    st.caption(f"Refreshes every {CONFIG.dashboard.refresh_seconds:g} seconds")


@st.fragment(run_every=CONFIG.dashboard.refresh_seconds)
def render_live_dashboard(lookback: int, limit: int) -> None:
    frame, error = _read_alerts(lookback, limit)
    published = _metric("sum(fraud_producer_messages_published_total)")
    processed = _metric("sum(records_processed_total)", "sum(fraud_stream_records_processed_total)")
    prometheus_ok = _prometheus_value("up") is not None

    status = st.columns(3)
    status[0].metric("Cassandra", "Connected" if error is None else "Unavailable")
    status[1].metric("Prometheus", "Connected" if prometheus_ok else "Unavailable")
    status[2].metric("Last refresh", datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))
    if error:
        st.error(f"Could not read alerts from Cassandra: {error}")
        st.info("Start Docker Desktop and run `docker compose up -d`, then refresh this page.")

    amount = float(frame["amount"].sum()) if not frame.empty else 0.0
    score = float(frame["anomaly_score"].mean()) if not frame.empty else 0.0
    known_fraud = int(frame["is_fraud_label"].sum()) if not frame.empty else 0
    cards = st.columns(4)
    cards[0].metric("Alerts in window", f"{len(frame):,}")
    cards[1].metric("Flagged amount", f"${amount:,.2f}")
    cards[2].metric("Mean anomaly score", f"{score:.4f}")
    cards[3].metric("Known fraud labels", f"{known_fraud:,}")
    metrics = st.columns(2)
    metrics[0].metric("Transactions published", "Unavailable" if published is None else f"{published:,.0f}")
    metrics[1].metric("Transactions processed", "Unavailable" if processed is None else f"{processed:,.0f}")

    if frame.empty:
        if error is None:
            st.info("No scored alerts in this time window. Keep the producer and Spark stream running or choose a wider window.")
        return

    st.subheader("Alert activity")
    trend = frame.set_index(pd.to_datetime(frame["detected_at"])).resample("5min").size().rename("Alerts")
    st.bar_chart(trend, height=220)
    st.subheader("Recent alerts")
    filters = st.columns([1, 1, 2])
    kinds = filters[0].multiselect("Transaction type", sorted(frame["tx_type"].dropna().unique()))
    minimum = filters[1].number_input("Minimum amount", min_value=0.0, value=0.0, step=100.0)
    account = filters[2].text_input("Account contains", placeholder="Origin or destination")
    visible = frame.copy()
    if kinds:
        visible = visible[visible["tx_type"].isin(kinds)]
    visible = visible[visible["amount"] >= minimum]
    if account:
        match = visible["name_orig"].str.contains(account, case=False, na=False)
        match |= visible["name_dest"].str.contains(account, case=False, na=False)
        visible = visible[match]
    visible = visible.rename(columns={
        "alert_id": "Alert ID",
        "detected_at": "Detected at", "name_orig": "Origin account",
        "name_dest": "Destination account", "step": "Simulation step",
        "tx_type": "Type", "amount": "Amount (USD)",
        "anomaly_score": "Anomaly score", "decision_threshold": "Threshold",
        "model_version": "Model", "is_fraud_label": "Known fraud",
    })
    st.dataframe(
        visible,
        column_config={
            "Detected at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "Amount (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Anomaly score": st.column_config.NumberColumn(format="%.5f"),
            "Threshold": st.column_config.NumberColumn(format="%.5f"),
        },
        hide_index=True,
        width="stretch",
        height=420,
    )
    st.download_button(
        "Download visible alerts", data=visible.to_csv(index=False).encode("utf-8"),
        file_name="fraud-alerts.csv", mime="text/csv", icon=":material/download:",
    )


render_live_dashboard(window_minutes, alert_limit)