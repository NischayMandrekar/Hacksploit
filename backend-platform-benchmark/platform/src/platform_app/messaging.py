import json, logging
from datetime import datetime, timezone
from platform_app.config import settings
from platform_app.obs import ADAPTER_FAILURES
logger = logging.getLogger(__name__)

class WakeupPublisher:
    """Best-effort wake-up transport; PostgreSQL polling remains the correctness path."""
    def publish(self, job_id: str) -> None:
        if not settings.rabbitmq_url: return
        try:
            import pika
            params = pika.URLParameters(settings.rabbitmq_url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
            channel.basic_publish(exchange="", routing_key=settings.rabbitmq_queue, body=job_id.encode(), properties=pika.BasicProperties(delivery_mode=2))
            connection.close()
        except Exception:
            ADAPTER_FAILURES.labels("rabbitmq").inc()
            logger.warning("rabbitmq_wakeup_publish_failed; database polling will recover", exc_info=True)

class EventPublisher:
    """Best-effort lifecycle event publisher; not a durable audit/outbox."""
    def publish(self, event_type: str, job_id: str, request_id: str, attempt: int = 0) -> None:
        if not settings.kafka_bootstrap: return
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(bootstrap_servers=settings.kafka_bootstrap.split(","), value_serializer=lambda value: json.dumps(value).encode("utf-8"), request_timeout_ms=1500, max_block_ms=1500)
            producer.send(settings.kafka_topic, {
                "event_id": __import__("uuid").uuid4().hex,
                "event_type": event_type,
                "job_id": job_id,
                "request_id": request_id,
                "attempt": attempt,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
            })
            producer.flush(timeout=1.5); producer.close(timeout=1.5)
        except Exception:
            ADAPTER_FAILURES.labels("kafka").inc()
            logger.warning("kafka_lifecycle_event_failed; event is best-effort", exc_info=True)
