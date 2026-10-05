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
except ImportError:
    pass  # Allow syntax check to pass without PySpark

logger = logging.getLogger(__name__)


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
    
    # Placeholder for starting the stream, to be replaced by full logic
    # query = raw.writeStream.format("console").start()
    # query.awaitTermination()

if __name__ == "__main__":
    main()
