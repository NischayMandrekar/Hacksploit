import hashlib
import hmac
import re
import unicodedata
from app.core.config import settings


def normalize_identifier(value: str, is_numeric_id: bool = False) -> str:
    """
    Normalizes raw PII identifiers to prevent trivial evasions (e.g. spaces,
    casing, dashes in phone numbers or IDs, Unicode homoglyphs).
    """
    if not value or not value.strip():
        raise ValueError("Identifier value cannot be empty.")
    
    # NFKD decomposes compatibility characters and homoglyphs into canonical forms
    cleaned = unicodedata.normalize("NFKD", value).strip().casefold()
    if is_numeric_id:
        # Strip all formatting characters: spaces, dashes, parentheses, plus signs
        cleaned = re.sub(r"[\s\-\(\)\+]", "", cleaned)
    return cleaned


def hash_pii(value: str, salt: str | None = None, is_numeric_id: bool = False) -> str:
    """
    Cryptographic PII pseudonymization using salted HMAC-SHA256.
    Ensures raw PII (Aadhaar, SSN, phone number, email) is never stored in graph
    nodes, database rows, or application logs while preserving linkability.
    """
    normalized = normalize_identifier(value, is_numeric_id=is_numeric_id)
    
    base_secret = settings.app_secret_key
    if salt:
        derived_key = hashlib.sha256(f"{base_secret}:{salt}".encode("utf-8")).digest()
    else:
        derived_key = hashlib.sha256(base_secret.encode("utf-8")).digest()
        
    return hmac.new(
        derived_key,
        normalized.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def constant_time_compare(left: str, right: str) -> bool:
    """Timing-safe string comparison to prevent side-channel timing attacks."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
