from sqlalchemy.exc import IntegrityError


def integrity_constraint_name(exc: IntegrityError) -> str | None:
    """Return the PostgreSQL constraint name, when the driver exposes it."""
    original = getattr(exc, "orig", None)
    diagnostic = getattr(original, "diag", None)
    name = getattr(diagnostic, "constraint_name", None)
    return name if isinstance(name, str) else None
