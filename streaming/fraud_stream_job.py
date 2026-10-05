"""Spark Structured Streaming fraud scoring job."""

import os
import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path for local runs
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.config import AppConfig, load_dotenv

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, from_json, to_timestamp, lit, pandas_udf
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, LongType, IntegerType, FloatType
    )
    import pandas as pd
except ImportError:
    pass  # Allow syntax check to pass without PySpark

import numpy as np
from common.features import VelocityTracker, feature_names
from training.bundle import ModelBundle, load_bundle

logger = logging.getLogger(__name__)

_bundle_cache: dict[str, ModelBundle] = {}


def _paysim_schema() -> "StructType":
    """Return the StructType matching PaySim's schema."""
    return StructType([
        StructField("step", LongType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("nameOrig", StringType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
        StructField("nameDest", StringType(), True),
        StructField("oldbalanceDest", DoubleType(), True),
        StructField("newbalanceDest", DoubleType(), True),
        StructField("isFraud", LongType(), True),
        StructField("isFlaggedFraud", LongType(), True),
        StructField("producedAtMs", LongType(), True),
    ])


def _build_spark_session(cfg: "AppConfig") -> "SparkSession":
    """Build and configure the SparkSession."""
    scfg = cfg.streaming
    spark = (
        SparkSession.builder.master(scfg.master)
        .appName(scfg.app_name)
        .config("spark.sql.shuffle.partitions", str(scfg.shuffle_partitions))
        .config("spark.executor.memory", scfg.executor_memory)
        .config("spark.driver.memory", scfg.driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Included directly in spark-submit, but good to have here for local runs
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0")
        .getOrCreate()
    )
    # Reduce spark logging verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _kafka_source(spark: "SparkSession", cfg: AppConfig) -> "DataFrame":
    """Read raw messages from the transactions Kafka topic."""
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg.kafka.bootstrap_servers_string)
        .option("subscribe", cfg.kafka.transactions_topic)
        .option("startingOffsets", cfg.streaming.starting_offsets)
        .option("maxOffsetsPerTrigger", cfg.streaming.max_offsets_per_trigger)
        .option("failOnDataLoss", "false")
        .load()
    )


def main() -> None:
    """Entrypoint for the streaming job."""
    load_dotenv()
    cfg = AppConfig.from_env()

    # We will expand this as we build out the pipeline
    spark = _build_spark_session(cfg)
    raw = _kafka_source(spark, cfg)
    parsed, dlq = _parse_messages(raw, cfg)
    
    score_udf = get_score_batch_udf(
        str(cfg.model.artifact_path.resolve()), 
        cfg.features.velocity_cache_max_accounts
    )
    
    from pyspark.sql.functions import struct
    # We pass the relevant columns as a struct to the UDF so it receives a pandas DataFrame
    paysim_cols = [f.name for f in _paysim_schema().fields]
    scored = parsed.withColumn("if_score", score_udf(struct(*paysim_cols)))
    
    # Load bundle here just to get the threshold, handle gracefully if missing
    threshold = float("inf")
    try:
        bundle = load_bundle(cfg.model.artifact_path)
        threshold = bundle.threshold
    except Exception:
        logger.warning("Model bundle not found, alerts will not be generated until a model is trained.")
        
    alerts = scored.filter(col("if_score") >= lit(threshold))
    
    # Placeholder for starting the stream, to be replaced by full logic
    # query = alerts.writeStream.format("console").start()
    # query.awaitTermination()

if __name__ == "__main__":
    main()

def _parse_messages(raw_df: "DataFrame", cfg: AppConfig) -> tuple["DataFrame", "DataFrame"]:
    """Parse JSON and route malformed records to DLQ.
    
    Returns:
        ``(parsed, dlq)`` DataFrames.
    """
    # 1 step = 1 hour simulation time, relative to an arbitrary epoch
    # Let's say epoch is 2026-01-01 for watermarking
    base_epoch = 1767225600  # 2026-01-01 00:00:00 UTC
    
    parsed_all = (
        raw_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
        .withColumn("data", from_json(col("value"), _paysim_schema()))
        .select("key", "value", "data.*")
    )
    
    # Event time: base_epoch + step * 3600
    parsed_all = parsed_all.withColumn(
        "event_time", 
        to_timestamp(lit(base_epoch) + col("step") * 3600)
    )
    
    parsed_all = parsed_all.withWatermark("event_time", cfg.streaming.watermark)
    
    # Well-formed: has an amount
    parsed = parsed_all.filter(col("amount").isNotNull())
    dlq = parsed_all.filter(col("amount").isNull())
    
    return parsed, dlq

def _get_bundle(artifact_dir: str) -> ModelBundle:
    """Load and cache the ModelBundle per executor."""
    if artifact_dir not in _bundle_cache:
        _bundle_cache[artifact_dir] = load_bundle(Path(artifact_dir))
    return _bundle_cache[artifact_dir]


def get_score_batch_udf(artifact_dir: str, velocity_cache_max_accounts: int):
    """Return a pandas UDF for scoring batches of transactions."""
    
    # We use a SCALAR_ITER UDF so we can initialize the VelocityTracker once per partition
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from collections.abc import Iterator
    
    @pandas_udf("double", PandasUDFType.SCALAR_ITER)
    def _score_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
        try:
            bundle = _get_bundle(artifact_dir)
        except Exception:
            # If the bundle isn't available yet, just yield NaNs.
            # This allows the streaming job to start before the model is trained.
            for batch in iterator:
                yield pd.Series([float('nan')] * len(batch))
            return

        tracker = VelocityTracker(
            window=bundle.velocity_window,
            cache_max_accounts=velocity_cache_max_accounts
        )
        
        for batch in iterator:
            if batch.empty:
                yield pd.Series([], dtype=np.float64)
                continue
                
            features_list = []
            for record in batch.to_dict('records'):
                features_list.append(tracker.process(record))
                
            feature_matrix = np.vstack(features_list)
            if_scores, _ = bundle.score(feature_matrix)
            yield pd.Series(if_scores)
            
    return _score_batch

def _write_cassandra(alert_df: "DataFrame", cfg_cass: "CassandraConfig") -> "DataFrame":
    """Sink alerts to Cassandra using foreachBatch."""
    
    def process_batch(df: "DataFrame", batch_id: int):
        if df.isEmpty():
            return
            
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.query import BatchStatement
        except ImportError:
            logger.warning("Cassandra driver not available, skipping write.")
            return

        auth_provider = None
        if cfg_cass.auth_required:
            auth_provider = PlainTextAuthProvider(cfg_cass.username, cfg_cass.password)
            
        cluster = Cluster(
            contact_points=cfg_cass.contact_points,
            port=cfg_cass.port,
            auth_provider=auth_provider
        )
        session = cluster.connect(cfg_cass.keyspace)
        
        insert_stmt = session.prepare(
            f"INSERT INTO {cfg_cass.keyspace}.fraud_alerts "
            "(bucket, event_time, name_orig, name_dest, amount, if_score, tx_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        
        # Convert DataFrame to a list of dicts
        rows = [row.asDict() for row in df.collect()]
        
        from common.schema import alert_bucket
        from datetime import datetime, timezone
        
        # Write in batches
        for i in range(0, len(rows), cfg_cass.write_batch_size):
            batch = rows[i:i + cfg_cass.write_batch_size]
            batch_stmt = BatchStatement()
            
            for row in batch:
                # Reconstruct event time and format bucket
                event_time_dt = row.get("event_time")
                if not event_time_dt:
                    event_time_dt = datetime.now(timezone.utc)
                    
                bucket = alert_bucket(event_time_dt)
                
                batch_stmt.add(insert_stmt, (
                    bucket,
                    event_time_dt,
                    row.get("nameOrig", ""),
                    row.get("nameDest", ""),
                    float(row.get("amount", 0.0)),
                    float(row.get("if_score", 0.0)),
                    row.get("type", "")
                ))
            
            # Simple retry loop
            for attempt in range(cfg_cass.max_write_retries):
                try:
                    session.execute(batch_stmt, timeout=cfg_cass.request_timeout_seconds)
                    break
                except Exception as e:
                    if attempt == cfg_cass.max_write_retries - 1:
                        logger.error(f"Failed to write batch to Cassandra: {e}")
                        raise
        
        cluster.shutdown()

    # Returns the DataStreamWriter
    return alert_df.writeStream.foreachBatch(process_batch)
