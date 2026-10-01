-- =============================================================================
-- Cassandra schema for the fraud detection pipeline
-- =============================================================================
-- Applied once at stack startup by infra/cassandra/init.sh. Every statement is
-- IF NOT EXISTS, so re-running the stack is safe.
--
-- DESIGN NOTE - Cassandra is queried, not joined. There are no secondary
-- indexes and no ALLOW FILTERING anywhere in this project; instead each access
-- pattern gets its own table and the streaming job writes to all of them. That
-- duplication is the intended tradeoff: writes are cheap on a log-structured
-- engine, unbounded-partition scans are not.
--
-- Access patterns served:
--   1. recent fraud alerts, newest first          -> fraud_alerts
--   2. alert history for one account              -> fraud_alerts_by_account
--   3. audit trail of scored transactions         -> transactions_raw
--   4. most-flagged accounts leaderboard          -> account_alert_counters
-- =============================================================================

-- SimpleStrategy with RF=1 is correct for this single-node development cluster.
-- A production deployment would use
--   NetworkTopologyStrategy with {'dc1': 3} and LOCAL_QUORUM reads/writes.
CREATE KEYSPACE IF NOT EXISTS fraud_detection
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
  AND durable_writes = true;

USE fraud_detection;

-- -----------------------------------------------------------------------------
-- 1. transactions_raw - audit trail of every scored transaction.
--
-- Partition key is (name_orig, event_date) rather than name_orig alone. A busy
-- account would otherwise accumulate an unbounded partition; bucketing by day
-- keeps each partition to at most one account-day of activity.
--
-- TimeWindowCompactionStrategy suits this append-only, TTL'd, time-ordered
-- workload far better than the default SizeTieredCompactionStrategy: expired
-- SSTables are dropped whole instead of being rewritten.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions_raw (
    name_orig           text,
    event_date          date,
    event_time          timestamp,
    transaction_id      uuid,
    step                int,
    tx_type             text,
    amount              decimal,
    old_balance_orig    decimal,
    new_balance_orig    decimal,
    name_dest           text,
    old_balance_dest    decimal,
    new_balance_dest    decimal,
    anomaly_score       double,
    is_predicted_fraud  boolean,
    is_fraud_label      boolean,
    is_flagged_fraud    boolean,
    model_version       text,
    PRIMARY KEY ((name_orig, event_date), event_time, transaction_id)
) WITH CLUSTERING ORDER BY (event_time DESC, transaction_id ASC)
  AND compaction = {
        'class': 'TimeWindowCompactionStrategy',
        'compaction_window_unit': 'HOURS',
        'compaction_window_size': 6
      }
  AND default_time_to_live = 7776000   -- 90 days
  AND comment = 'All scored transactions, bucketed by account-day.';

-- -----------------------------------------------------------------------------
-- 2. fraud_alerts - the dashboard's primary read target.
--
-- Partitioned by a five-minute time bucket so "show me recent alerts" is a
-- single-partition read of a bounded, already-sorted row set. alert_bucket is
-- the UTC start of the window formatted 'yyyyMMddHHmm' (e.g. '202610011205');
-- see common/schema.py:alert_bucket for the single implementation of that
-- truncation.
--
-- Clustering DESC on detected_at means the newest alert is the first row, so the
-- dashboard never sorts client-side. alert_id breaks ties between alerts sharing
-- a millisecond.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_bucket        text,
    detected_at         timestamp,
    alert_id            uuid,
    name_orig           text,
    name_dest           text,
    step                int,
    tx_type             text,
    amount              decimal,
    old_balance_orig    decimal,
    new_balance_orig    decimal,
    old_balance_dest    decimal,
    new_balance_dest    decimal,
    anomaly_score       double,
    decision_threshold  double,
    model_version       text,
    is_fraud_label      boolean,
    PRIMARY KEY (alert_bucket, detected_at, alert_id)
) WITH CLUSTERING ORDER BY (detected_at DESC, alert_id ASC)
  AND compaction = {
        'class': 'TimeWindowCompactionStrategy',
        'compaction_window_unit': 'HOURS',
        'compaction_window_size': 1
      }
  AND comment = 'Flagged transactions, bucketed into five-minute windows.';

-- -----------------------------------------------------------------------------
-- 3. fraud_alerts_by_account - same alerts, keyed for investigator lookup.
--
-- A deliberate denormalised duplicate of fraud_alerts. Querying alert history
-- for one account from the table above would require scanning every time
-- bucket; here it is one partition. Alerts per account are naturally bounded,
-- so no secondary bucketing is needed.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fraud_alerts_by_account (
    name_orig           text,
    detected_at         timestamp,
    alert_id            uuid,
    name_dest           text,
    step                int,
    tx_type             text,
    amount              decimal,
    anomaly_score       double,
    decision_threshold  double,
    model_version       text,
    PRIMARY KEY (name_orig, detected_at, alert_id)
) WITH CLUSTERING ORDER BY (detected_at DESC, alert_id ASC)
  AND comment = 'Alert history per originating account.';

-- -----------------------------------------------------------------------------
-- 4. account_alert_counters - "top flagged accounts" leaderboard.
--
-- Cassandra counters are signed 64-bit integers only, so monetary totals are
-- stored in minor units (cents) and divided by 100 at read time. Counter
-- updates are not idempotent, which is why this table is updated from the same
-- idempotency-checked path as the alert inserts and is treated as an
-- approximate aggregate rather than a ledger.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS account_alert_counters (
    name_orig           text PRIMARY KEY,
    alert_count         counter,
    flagged_amount_minor counter
);
