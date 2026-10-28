import json, logging, sys, time
from prometheus_client import Counter, Histogram, Gauge, CONTENT_TYPE_LATEST, generate_latest

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status_class"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "route"])
JOBS_CREATED = Counter("jobs_created_total", "Jobs accepted")
JOBS_COMPLETED = Counter("jobs_completed_total", "Jobs reaching terminal outcome", ["outcome"])
JOBS_RETRIED = Counter("jobs_retried_total", "Jobs retried")
JOBS_ACTIVE = Gauge("jobs_active", "Jobs currently executing")
QUEUE_DEPTH = Gauge("job_queue_depth", "Jobs waiting for execution")
CLAIM_FAILURES = Counter("worker_claim_failures_total", "Worker claim failures")
ADAPTER_FAILURES = Counter("adapter_publish_failures_total", "Optional event/wakeup adapter failures", ["adapter"])

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)), "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        for key in ("service", "request_id", "job_id", "worker_id", "event_type", "duration_ms", "error_code"):
            value = getattr(record, key, None)
            if value is not None: data[key] = value
        if record.exc_info: data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)

def configure_logging(level: str):
    handler = logging.StreamHandler(sys.stdout); handler.setFormatter(JsonFormatter())
    root = logging.getLogger(); root.handlers.clear(); root.addHandler(handler); root.setLevel(level.upper())

def metrics_payload():
    return generate_latest(), CONTENT_TYPE_LATEST
