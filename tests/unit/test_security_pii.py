import pytest
from app.core.security import (
    constant_time_compare,
    hash_pii,
    normalize_identifier,
)


def test_normalize_identifier_email():
    raw_email = "  John.Doe+KYC@Example.COM  "
    normalized = normalize_identifier(raw_email)
    assert normalized == "john.doe+kyc@example.com"


def test_normalize_identifier_phone():
    raw_phone = " +91 (987) 654-3210 "
    normalized = normalize_identifier(raw_phone, is_numeric_id=True)
    assert normalized == "919876543210"


def test_normalize_identifier_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_identifier("")
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_identifier("   ")


def test_hash_pii_determinism():
    phone = "+91 99999 88888"
    hash1 = hash_pii(phone, is_numeric_id=True)
    hash2 = hash_pii(phone, is_numeric_id=True)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string


def test_hash_pii_salt_sensitivity():
    phone = "+91 99999 88888"
    hash_default = hash_pii(phone, is_numeric_id=True)
    hash_salted = hash_pii(phone, salt="tenant_123", is_numeric_id=True)
    assert hash_default != hash_salted
    assert len(hash_salted) == 64


def test_constant_time_compare():
    assert constant_time_compare("secret_token_abc", "secret_token_abc") is True
    assert constant_time_compare("secret_token_abc", "secret_token_xyz") is False
    assert constant_time_compare("short", "longer_token") is False
