import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class PermanentJobError(Exception):
    """Unrecoverable error (e.g. invalid input, corrupted data). Will not retry."""
    def __init__(self, message: str, code: str = "permanent_job_error"):
        self.code = code
        self.message = message
        super().__init__(message)


class RetryableJobError(Exception):
    """Transient error (e.g. momentary lock contention, timeout). Will retry if attempts remain."""
    def __init__(self, message: str, code: str = "retryable_job_error"):
        self.code = code
        self.message = message
        super().__init__(message)


class JobCancelled(Exception):
    """Raised when cancellation was requested by client and detected during cooperative checkpoints."""
    def __init__(self, message: str = "Job execution cancelled by client request."):
        super().__init__(message)


@dataclass(frozen=True)
class ExecutionContext:
    job_id: str
    request_id: str
    attempt: int
    deadline_monotonic: float
    cancellation_check: Callable[[], bool]

    def check(self) -> None:
        """Call periodically in handler loops/stages to yield on cancellation or timeout."""
        if self.cancellation_check():
            raise JobCancelled()
        if time.monotonic() > self.deadline_monotonic:
            raise RetryableJobError("Job execution timeout exceeded.", code="execution_timeout")


HandlerFunc = Callable[[dict[str, Any], ExecutionContext], dict[str, Any]]


class HandlerRegistry:
    """
    Extensible task registry. Modules register their bounded handlers without
    modifying worker core (satisfies Open/Closed principle).
    """

    def __init__(self):
        self._handlers: dict[str, HandlerFunc] = {}

    def register(self, job_type: str, handler: HandlerFunc) -> None:
        if job_type in self._handlers:
            raise ValueError(f"Handler for job_type '{job_type}' is already registered.")
        self._handlers[job_type] = handler

    def get(self, job_type: str) -> HandlerFunc | None:
        return self._handlers.get(job_type)

    def validate_type(self, job_type: str) -> None:
        if job_type not in self._handlers:
            raise PermanentJobError(
                f"Unsupported or unregistered job type: '{job_type}'. Registered: {list(self._handlers.keys())}",
                code="unknown_job_type",
            )

    def execute(self, job_type: str, payload: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        self.validate_type(job_type)
        handler = self._handlers[job_type]
        context.check()
        result = handler(payload, context)
        context.check()
        return result


# Global handler registry instance for the modular monolith
default_registry = HandlerRegistry()
