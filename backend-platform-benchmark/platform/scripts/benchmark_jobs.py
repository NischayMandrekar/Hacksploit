#!/usr/bin/env python3
"""Small repeatable API + worker benchmark for the local backend platform.

Example:
  python scripts/benchmark_jobs.py --base-url http://127.0.0.1:8080 \
    --api-key "$API_KEY" --jobs 100 --concurrency 8 --report benchmark.json

This measures the whole HTTP submission path and eventual job completion. It is
not a substitute for a dedicated load-testing tool or a controlled benchmark.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

TERMINAL = {"succeeded", "failed", "cancelled"}


class RequestFailure(RuntimeError):
    pass


def request_json(base_url: str, path: str, api_key: str, *, method: str = "GET",
                 body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, dict[str, Any], float]:
    data = None if body is None else json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    headers = {"Accept": "application/json", "X-API-Key": api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - started
            payload = json.loads(raw) if raw else {}
            return response.status, payload, elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"body": raw.decode("utf-8", "replace")[:500]}
        return exc.code, payload, time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RequestFailure(str(exc)) from exc


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("PLATFORM_BASE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--api-key", default=os.getenv("PLATFORM_API_KEY"), help="key with jobs:create and jobs:read scopes")
    parser.add_argument("--jobs", type=int, default=100, help="number of jobs (1..10000)")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel submitters (1..128)")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--deadline", type=float, default=120.0, help="seconds to wait for all jobs to reach a terminal state")
    parser.add_argument("--request-timeout", type=float, default=10.0)
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or PLATFORM_API_KEY")
    if not 1 <= args.jobs <= 10000:
        parser.error("--jobs must be between 1 and 10000")
    if not 1 <= args.concurrency <= 128:
        parser.error("--concurrency must be between 1 and 128")
    if args.poll_interval <= 0 or args.deadline <= 0 or args.request_timeout <= 0:
        parser.error("poll interval, deadline, and request timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    # Fail early if the target isn't ready. This is a separate probe, not part of latency stats.
    try:
        status, ready, _ = request_json(base_url, "/ready", args.api_key, timeout=args.request_timeout)
    except RequestFailure as exc:
        print(f"Cannot reach {base_url}/ready: {exc}", file=sys.stderr)
        return 2
    if status != 200:
        print(f"Platform is not ready (HTTP {status}): {json.dumps(ready)}", file=sys.stderr)
        return 2

    submitted: list[dict[str, Any]] = []
    submit_errors: list[dict[str, Any]] = []
    submission_started = time.perf_counter()

    def submit(index: int) -> dict[str, Any]:
        payload = {
            "job_type": "demo.compute",
            "input": {"operation": "sum", "values": [index, 1, 2, 3]},
            "timeout_seconds": 30,
            "max_attempts": 2,
        }
        code, result, elapsed = request_json(
            base_url, "/api/v1/jobs", args.api_key, method="POST", body=payload,
            timeout=args.request_timeout,
        )
        if code not in (200, 202) or "id" not in result:
            return {"index": index, "ok": False, "status_code": code, "response": result, "latency_seconds": elapsed}
        return {"index": index, "ok": True, "id": result["id"], "status": result.get("status"), "latency_seconds": elapsed}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(submit, i) for i in range(args.jobs)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except RequestFailure as exc:
                submit_errors.append({"error": str(exc)})
                continue
            if result["ok"]:
                submitted.append(result)
            else:
                submit_errors.append(result)
    submission_elapsed = time.perf_counter() - submission_started

    jobs = {item["id"]: {"submission_latency_seconds": item["latency_seconds"], "last_status": item.get("status")} for item in submitted}
    poll_started = time.perf_counter()
    deadline_at = poll_started + args.deadline
    poll_errors: list[dict[str, Any]] = []
    while jobs and time.perf_counter() < deadline_at:
        pending = [job_id for job_id, meta in jobs.items() if meta.get("status") not in TERMINAL]
        if not pending:
            break

        def poll(job_id: str) -> tuple[str, int, dict[str, Any], float]:
            code, result, elapsed = request_json(base_url, f"/api/v1/jobs/{job_id}", args.api_key, timeout=args.request_timeout)
            return job_id, code, result, elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.concurrency, max(1, len(pending)))) as pool:
            for future in concurrent.futures.as_completed([pool.submit(poll, job_id) for job_id in pending]):
                try:
                    job_id, code, result, elapsed = future.result()
                except RequestFailure as exc:
                    poll_errors.append({"error": str(exc)})
                    continue
                if code != 200:
                    poll_errors.append({"job_id": job_id, "status_code": code, "response": result})
                    continue
                jobs[job_id]["status"] = result.get("status", "unknown")
                jobs[job_id]["last_status"] = result.get("status", "unknown")
                jobs[job_id]["poll_latency_seconds"] = elapsed
                jobs[job_id]["attempt_count"] = result.get("attempt_count")
                jobs[job_id]["error_code"] = result.get("error_code")
        if any(meta.get("status") not in TERMINAL for meta in jobs.values()):
            time.sleep(args.poll_interval)

    finished_elapsed = time.perf_counter() - poll_started
    status_counts: dict[str, int] = {}
    for meta in jobs.values():
        status = meta.get("status", "timed_out")
        status_counts[status] = status_counts.get(status, 0) + 1
    terminal_count = sum(count for status, count in status_counts.items() if status in TERMINAL)
    successful = status_counts.get("succeeded", 0)
    submit_latencies = [item["latency_seconds"] for item in submitted]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": base_url,
        "configuration": {"requested_jobs": args.jobs, "concurrency": args.concurrency, "poll_interval_seconds": args.poll_interval, "completion_deadline_seconds": args.deadline},
        "readiness": {"status_code": status, "response": ready},
        "submission": {
            "accepted": len(submitted), "rejected_or_failed": len(submit_errors),
            "elapsed_seconds": round(submission_elapsed, 4),
            "accepted_jobs_per_second": round(len(submitted) / submission_elapsed, 3) if submission_elapsed else None,
            "latency_ms": {
                "p50": round((percentile(submit_latencies, .50) or 0) * 1000, 3) if submit_latencies else None,
                "p95": round((percentile(submit_latencies, .95) or 0) * 1000, 3) if submit_latencies else None,
                "p99": round((percentile(submit_latencies, .99) or 0) * 1000, 3) if submit_latencies else None,
                "max": round(max(submit_latencies) * 1000, 3) if submit_latencies else None,
            },
        },
        "completion": {
            "observed_seconds": round(finished_elapsed, 4), "status_counts": status_counts,
            "terminal_jobs": terminal_count, "successful_jobs": successful,
            "successful_jobs_per_second_from_poll_start": round(successful / finished_elapsed, 3) if finished_elapsed else None,
            "completion_deadline_reached": any(meta.get("status") not in TERMINAL for meta in jobs.values()),
        },
        "errors": {"submission": submit_errors[:50], "polling": poll_errors[:50]},
        "notes": [
            "Run only against a system you own or have permission to test.",
            "Submission rate includes HTTP, Nginx, API validation, and database commit latency.",
            "Completion rate is approximate and depends on polling interval and client machine.",
            "This small harness is for repeatable baseline measurements, not a capacity certification.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if submit_errors or poll_errors or report["completion"]["completion_deadline_reached"] or successful != args.jobs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
