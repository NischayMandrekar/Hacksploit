import hashlib
import re
import uuid
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.core.errors import ArtifactStorageError

MAGIC_BYTES: dict[str, list[bytes]] = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "application/pdf": [b"%PDF"],
    "image/webp": [b"RIFF"],  # also checks WEBP at bytes 8:12
}

EXTENSIONS: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


class ArtifactStore(Protocol):
    """Protocol for binary artifact storage outside database tables."""

    def save_artifact(
        self, applicant_id: str, artifact_type: str, data: bytes, mime_type: str
    ) -> str:
        """Saves binary data and returns a persistent URI."""
        ...

    def load_artifact(self, storage_uri: str) -> bytes:
        """Retrieves binary data by storage URI."""
        ...

    def delete_artifact(self, storage_uri: str) -> bool:
        """Deletes binary data by storage URI."""
        ...


class LocalDiskArtifactStore:
    """
    Local filesystem artifact store with MIME verification, magic-byte checking,
    size-bounding, and path traversal prevention.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.artifact_storage_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_id(self, identifier: str, param_name: str) -> str:
        if not identifier or not SAFE_ID_REGEX.match(identifier):
            raise ArtifactStorageError(
                f"Invalid {param_name}: must contain only alphanumeric characters, hyphens, or underscores.",
                code="invalid_identifier",
            )
        return identifier

    def _verify_magic_bytes(self, data: bytes, mime_type: str) -> None:
        if mime_type not in settings.allowed_artifact_mime_types:
            raise ArtifactStorageError(
                f"MIME type '{mime_type}' is not permitted. Allowed: {settings.allowed_artifact_mime_types}",
                code="unsupported_mime_type",
            )

        signatures = MAGIC_BYTES.get(mime_type)
        if not signatures:
            raise ArtifactStorageError(
                f"No magic byte signature configured for '{mime_type}'",
                code="unsupported_mime_type",
            )

        matches_magic = any(data.startswith(sig) for sig in signatures)
        if mime_type == "image/webp":
            matches_magic = matches_magic and (len(data) >= 12 and data[8:12] == b"WEBP")

        if not matches_magic:
            raise ArtifactStorageError(
                f"File contents do not match declared MIME type '{mime_type}'.",
                code="magic_bytes_mismatch",
            )

    def save_artifact(
        self, applicant_id: str, artifact_type: str, data: bytes, mime_type: str
    ) -> str:
        clean_applicant_id = self._sanitize_id(applicant_id, "applicant_id")
        clean_artifact_type = self._sanitize_id(artifact_type, "artifact_type")

        if len(data) == 0:
            raise ArtifactStorageError("Artifact cannot be empty.", code="empty_file")

        if len(data) > settings.artifact_max_bytes:
            raise ArtifactStorageError(
                f"Artifact size ({len(data)} bytes) exceeds maximum allowed limit ({settings.artifact_max_bytes} bytes).",
                code="file_too_large",
            )

        self._verify_magic_bytes(data, mime_type)

        ext = EXTENSIONS.get(mime_type, ".bin")
        file_id = uuid.uuid4().hex[:12]
        content_hash = hashlib.sha256(data).hexdigest()[:16]

        target_dir = self.base_dir / clean_applicant_id
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{clean_artifact_type}_{file_id}_{content_hash}{ext}"
        target_file = target_dir / filename

        with open(target_file, "wb") as f:
            f.write(data)

        return f"file://{target_file.resolve()}"

    def _resolve_uri(self, storage_uri: str) -> Path:
        if not storage_uri.startswith("file://"):
            raise ArtifactStorageError("Invalid storage URI scheme.", code="invalid_uri")

        path_str = storage_uri[len("file://") :]
        target_path = Path(path_str).resolve()

        # Enforce strict path jail inside self.base_dir
        try:
            target_path.relative_to(self.base_dir)
        except ValueError:
            raise ArtifactStorageError(
                "Access denied: Target path outside artifact storage jail.",
                code="path_traversal_detected",
                status_code=403,
            )

        return target_path

    def load_artifact(self, storage_uri: str) -> bytes:
        target_path = self._resolve_uri(storage_uri)

        if not target_path.is_file():
            raise ArtifactStorageError(
                "Requested artifact does not exist.",
                code="artifact_not_found",
                status_code=404,
            )

        with open(target_path, "rb") as f:
            return f.read()

    def delete_artifact(self, storage_uri: str) -> bool:
        target_path = self._resolve_uri(storage_uri)
        if target_path.is_file():
            target_path.unlink()
            return True
        return False


def get_artifact_store() -> ArtifactStore:
    return LocalDiskArtifactStore()
