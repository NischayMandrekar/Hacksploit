from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError

from platform_app.db_errors import integrity_constraint_name


def make_error(constraint_name):
    diagnostic = SimpleNamespace(constraint_name=constraint_name)
    original = SimpleNamespace(diag=diagnostic)
    return IntegrityError("statement", {}, original)


def test_integrity_error_reports_expected_idempotency_constraint():
    assert integrity_constraint_name(make_error("uq_jobs_idempotency")) == "uq_jobs_idempotency"


def test_integrity_error_does_not_misclassify_other_constraints():
    assert integrity_constraint_name(make_error("api_keys_key_hash_key")) == "api_keys_key_hash_key"
    assert integrity_constraint_name(IntegrityError("statement", {}, Exception("driver error"))) is None
