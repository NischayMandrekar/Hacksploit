import concurrent.futures
import logging
import signal
import threading
import uuid
from collections.abc import Callable
from sqlalchemy.orm import Session

from app.core.config import settings
from app.infrastructure.database import SessionLocal
from app.infrastructure.jobs.registry import HandlerRegistry, default_registry
from app.infrastructure.jobs.service import claim_one, execute_job_lifecycle

logger = logging.getLogger(__name__)


class WorkerService:
    """
    Zero-broker background worker with bounded concurrency and PostgreSQL row-level locking.
    Recovers expired leases and executes claimed tasks through HandlerRegistry.
    """

    def __init__(
        self,
        worker_id: str | None = None,
        concurrency: int | None = None,
        poll_interval: float | None = None,
        registry: HandlerRegistry | None = None,
        db_factory: Callable[[], Session] | None = None,
    ):
        self.worker_id = worker_id or settings.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.concurrency = concurrency or settings.worker_concurrency
        self.poll_interval = poll_interval or settings.worker_poll_interval_seconds
        self.registry = registry or default_registry
        self.db_factory = db_factory or SessionLocal
        self.stop_event = threading.Event()

    def stop(self) -> None:
        """Signals the worker to finish claimed tasks and shut down."""
        self.stop_event.set()

    def run_once(self) -> bool:
        """
        Polls and executes at most one job synchronously.
        Useful for unit testing and deterministic worker loops.
        """
        with self.db_factory() as db:
            job = claim_one(db, self.worker_id, settings.worker_lease_seconds)

        if not job:
            return False

        logger.info(
            "job_claimed",
            extra={"worker_id": self.worker_id, "job_id": str(job.id), "type": job.job_type},
        )
        execute_job_lifecycle(
            job=job,
            registry=self.registry,
            db_factory=self.db_factory,
            execution_timeout_seconds=settings.worker_execution_timeout_seconds,
        )
        return True

    def run(self) -> None:
        """Main bounded worker pool execution loop."""
        logger.info(
            "worker_service_started",
            extra={"worker_id": self.worker_id, "concurrency": self.concurrency},
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            running: set[concurrent.futures.Future] = set()

            while not self.stop_event.is_set():
                # Clean up completed tasks
                done = {fut for fut in running if fut.done()}
                for fut in done:
                    try:
                        fut.result()
                    except Exception:
                        logger.exception("worker_task_future_exception", extra={"worker_id": self.worker_id})
                running -= done

                # Claim new work if capacity remains
                if len(running) < self.concurrency:
                    try:
                        with self.db_factory() as db:
                            job = claim_one(db, self.worker_id, settings.worker_lease_seconds)
                        if job:
                            logger.info(
                                "job_claimed",
                                extra={"worker_id": self.worker_id, "job_id": str(job.id), "type": job.job_type},
                            )
                            future = pool.submit(
                                execute_job_lifecycle,
                                job=job,
                                registry=self.registry,
                                db_factory=self.db_factory,
                                execution_timeout_seconds=settings.worker_execution_timeout_seconds,
                            )
                            running.add(future)
                            continue
                    except Exception:
                        logger.exception("worker_claim_exception", extra={"worker_id": self.worker_id})

                self.stop_event.wait(self.poll_interval)

            # Clean shutdown: wait for running tasks to complete
            for fut in running:
                try:
                    fut.result(timeout=settings.worker_execution_timeout_seconds + 5)
                except Exception:
                    logger.warning("worker_task_did_not_finish_cleanly", extra={"worker_id": self.worker_id})

        logger.info("worker_service_stopped", extra={"worker_id": self.worker_id})


def main() -> None:
    worker = WorkerService()

    def handle_signal(*_):
        logger.info("shutdown_signal_received")
        worker.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    worker.run()


if __name__ == "__main__":
    main()
