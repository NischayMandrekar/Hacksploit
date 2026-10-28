import math
import pytest
from pydantic import ValidationError
from platform_app.schemas import JobCreate


def test_job_input_rejects_non_finite_numbers():
    with pytest.raises(ValidationError, match="finite"):
        JobCreate.model_validate({"job_type": "demo.compute", "input": {"values": [math.nan]}})


def test_job_input_rejects_deeply_nested_payloads():
    nested = "leaf"
    for _ in range(34):
        nested = {"child": nested}
    with pytest.raises(ValidationError, match="nesting"):
        JobCreate.model_validate({"job_type": "demo.echo", "input": nested})


def test_job_input_accepts_normal_nested_json():
    job = JobCreate.model_validate({"job_type": "demo.echo", "input": {"message": "hello", "meta": [1, 2.5, None, True]}})
    assert job.input["message"] == "hello"


def test_boolean_is_not_accepted_as_timeout_or_attempt_count():
    with pytest.raises(ValidationError):
        JobCreate.model_validate({"job_type": "demo.echo", "input": {}, "timeout_seconds": True})
    with pytest.raises(ValidationError):
        JobCreate.model_validate({"job_type": "demo.echo", "input": {}, "max_attempts": False})
