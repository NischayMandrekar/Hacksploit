import time
import pytest
from platform_app.execution import CPUExecutionBackend, ExecutionContext, PermanentJobError


def context():
    return ExecutionContext(job_id="job-1", request_id="req-1", attempt=1, deadline_monotonic=time.monotonic()+2, cancellation_check=lambda: False)


def test_compute_handler_returns_expected_result():
    result = CPUExecutionBackend().execute("demo.compute", {"operation": "sum", "values": [10, 20, 30]}, context())
    assert result == {"operation": "sum", "count": 3, "result": 60}


def test_unknown_job_type_is_rejected_not_executed():
    with pytest.raises(PermanentJobError) as error:
        CPUExecutionBackend().validate_type("python.exec")
    assert error.value.code == "unknown_job_type"


def test_compute_rejects_unbounded_or_invalid_input():
    with pytest.raises(PermanentJobError):
        CPUExecutionBackend().execute("demo.compute", {"operation": "sum", "values": ["not-a-number"]}, context())


def test_compute_rejects_overflowing_result():
    with pytest.raises(PermanentJobError) as error:
        CPUExecutionBackend().execute(
            "demo.compute", {"operation": "sum", "values": [1e308, 1e308]}, context()
        )
    assert error.value.code == "non_finite_result"
