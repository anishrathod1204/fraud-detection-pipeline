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
    from pyspark.sql.functions import col, from_json, to_timestamp, lit
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, LongType, IntegerType
    )
except ImportError:
    pass  # Allow syntax check to pass without PySpark

logger = logging.getLogger(__name__)


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
    
    # Placeholder for starting the stream, to be replaced by full logic
    # query = parsed.writeStream.format("console").start()
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
