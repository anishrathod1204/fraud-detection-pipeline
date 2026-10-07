import time

from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError

from common.logging_setup import get_logger

log = get_logger("kafka_utils")


def ensure_topic(bootstrap: str, topic: str, partitions: int = 3, retries: int = 30) -> None:
    """Wait for the broker, then create the topic if it does not exist."""
    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="fraud-admin")
            try:
                admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
                log.info("Created topic %s (%d partitions)", topic, partitions)
            except TopicAlreadyExistsError:
                log.info("Topic %s already exists", topic)
            finally:
                admin.close()
            return
        except NoBrokersAvailable:
            log.warning("Kafka not reachable at %s (attempt %d/%d)", bootstrap, attempt, retries)
            time.sleep(3)
    raise RuntimeError(f"Kafka broker {bootstrap} never became reachable")
