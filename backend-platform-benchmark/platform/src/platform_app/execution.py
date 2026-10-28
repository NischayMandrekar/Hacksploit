import json, math, time
from dataclasses import dataclass
from typing import Any, Callable
from platform_app.config import settings

class PermanentJobError(Exception):
    def __init__(self, code: str, message: str): self.code, self.message = code, message
class RetryableJobError(Exception): pass
class JobCancelled(Exception): pass

@dataclass(frozen=True)
class ExecutionContext:
    job_id: str
    request_id: str
    attempt: int
    deadline_monotonic: float
    cancellation_check: Callable[[], bool]
    def check(self):
        if self.cancellation_check(): raise JobCancelled()
        if time.monotonic() > self.deadline_monotonic: raise RetryableJobError("execution_timeout")

class CPUExecutionBackend:
    """Explicit allow-list of bounded built-in handlers; never executes user-supplied code."""
    def __init__(self): self._handlers = {"demo.echo": self._echo, "demo.compute": self._compute}
    def validate_type(self, job_type: str):
        if job_type not in self._handlers: raise PermanentJobError("unknown_job_type", f"Unsupported job type: {job_type}")
    def execute(self, job_type: str, payload: dict[str, Any], context: ExecutionContext):
        self.validate_type(job_type); context.check()
        result = self._handlers[job_type](payload, context)
        context.check()
        if len(json.dumps(result, ensure_ascii=False).encode()) > settings.job_max_output_bytes: raise PermanentJobError("output_too_large", "Handler output exceeded the configured limit.")
        return result
    @staticmethod
    def _echo(payload, context):
        context.check(); return {"echo": payload.get("message", ""), "job_id": context.job_id, "attempt": context.attempt}
    @staticmethod
    def _compute(payload, context):
        operation = payload.get("operation", "sum"); values = payload.get("values", [])
        if not isinstance(values, list) or len(values) > 1000 or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise PermanentJobError("invalid_compute_input", "values must be a list of at most 1000 finite numbers.")
        context.check()
        if operation == "sum": result = sum(values)
        elif operation == "mean": result = sum(values) / len(values) if values else 0
        elif operation == "min": result = min(values) if values else None
        elif operation == "max": result = max(values) if values else None
        else: raise PermanentJobError("invalid_operation", "operation must be sum, mean, min, or max.")
        if isinstance(result, float) and not math.isfinite(result):
            raise PermanentJobError("non_finite_result", "The operation produced a non-finite numeric result.")
        return {"operation": operation, "count": len(values), "result": result}
