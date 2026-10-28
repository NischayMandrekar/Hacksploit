import concurrent.futures, logging, os, signal, threading, time, uuid
from sqlalchemy.exc import SQLAlchemyError
from platform_app.config import settings
from platform_app.db import SessionLocal
from platform_app.jobs import claim_one, refresh_queue_depth, run_claimed
from platform_app.obs import configure_logging, CLAIM_FAILURES

configure_logging(settings.app_log_level)
logger = logging.getLogger(__name__)
stop_event = threading.Event()
wakeup_event = threading.Event()
worker_id = settings.worker_id or f"worker-{uuid.uuid4().hex[:10]}"

def _stop(*_): stop_event.set(); wakeup_event.set()
signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGINT, _stop)

def rabbit_wakeup_listener():
    if not settings.rabbitmq_url: return
    while not stop_event.is_set():
        connection = None
        try:
            import pika
            connection = pika.BlockingConnection(pika.URLParameters(settings.rabbitmq_url))
            channel = connection.channel(); channel.queue_declare(queue=settings.rabbitmq_queue, durable=True)
            channel.basic_qos(prefetch_count=1)
            def callback(ch, method, properties, body):
                wakeup_event.set(); ch.basic_ack(delivery_tag=method.delivery_tag)
            channel.basic_consume(queue=settings.rabbitmq_queue, on_message_callback=callback)
            logger.info("rabbitmq_wakeup_listener_started", extra={"service": "worker", "worker_id": worker_id})
            while connection.is_open and not stop_event.is_set(): connection.process_data_events(time_limit=1)
        except Exception:
            logger.warning("rabbitmq_wakeup_listener_unavailable; PostgreSQL polling remains active", extra={"service": "worker", "worker_id": worker_id}, exc_info=True)
            stop_event.wait(3)
        finally:
            try:
                if connection and connection.is_open: connection.close()
            except Exception: pass

def main():
    if settings.prometheus_enabled:
        from prometheus_client import start_http_server
        start_http_server(9101, addr="0.0.0.0")
    logger.info("worker_started", extra={"service": "worker", "worker_id": worker_id})
    listener = threading.Thread(target=rabbit_wakeup_listener, name="rabbitmq-wakeup", daemon=True)
    listener.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.worker_concurrency) as pool:
        running = set()
        while not stop_event.is_set():
            done = {future for future in running if future.done()}
            for future in done:
                try: future.result()
                except Exception: logger.exception("worker_future_failed", extra={"service": "worker", "worker_id": worker_id})
            running -= done
            if len(running) < settings.worker_concurrency:
                try:
                    with SessionLocal() as db:
                        job = claim_one(db, worker_id)
                        refresh_queue_depth(db)
                    if job:
                        logger.info("job_claimed", extra={"service": "worker", "worker_id": worker_id, "job_id": str(job.id), "request_id": job.request_id})
                        running.add(pool.submit(run_claimed, job))
                        continue
                except SQLAlchemyError:
                    CLAIM_FAILURES.inc(); logger.exception("job_claim_failed", extra={"service": "worker", "worker_id": worker_id})
            wakeup_event.wait(settings.worker_poll_interval_seconds)
            wakeup_event.clear()
        for future in running:
            try: future.result(timeout=settings.worker_execution_timeout_seconds + 5)
            except Exception: logger.warning("worker_shutdown_task_did_not_finish_cleanly", extra={"service": "worker", "worker_id": worker_id})
    logger.info("worker_stopped", extra={"service": "worker", "worker_id": worker_id})

if __name__ == "__main__": main()
