import tempfile
import pytest
from app.core.errors import ArtifactStorageError
from app.infrastructure.storage.artifact_store import LocalDiskArtifactStore

# Minimal valid magic byte headers for testing
VALID_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 50
VALID_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 50
VALID_PDF = b"%PDF-1.4\n" + b"\x00" * 50
VALID_WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 50
FAKE_IMAGE_TEXT = b"Hello, this is just a plain text file pretending to be JPEG."


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield LocalDiskArtifactStore(base_dir=temp_dir)


def test_save_and_load_valid_jpeg(store):
    uri = store.save_artifact("app_1001", "id_front", VALID_JPEG, "image/jpeg")
    assert uri.startswith("file://")

    loaded_bytes = store.load_artifact(uri)
    assert loaded_bytes == VALID_JPEG


def test_save_and_load_valid_png(store):
    uri = store.save_artifact("app_1002", "selfie", VALID_PNG, "image/png")
    assert store.load_artifact(uri) == VALID_PNG


def test_save_and_load_valid_pdf(store):
    uri = store.save_artifact("app_1003", "bank_statement", VALID_PDF, "application/pdf")
    assert store.load_artifact(uri) == VALID_PDF


def test_save_and_load_valid_webp(store):
    uri = store.save_artifact("app_1004", "selfie_crop", VALID_WEBP, "image/webp")
    assert store.load_artifact(uri) == VALID_WEBP


def test_reject_forged_magic_bytes(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("app_9999", "id_front", FAKE_IMAGE_TEXT, "image/jpeg")
    assert exc_info.value.code == "magic_bytes_mismatch"


def test_reject_empty_file(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("app_9999", "id_front", b"", "image/jpeg")
    assert exc_info.value.code == "empty_file"


def test_reject_unsupported_mime_type(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("app_9999", "script", b"#!/bin/bash", "text/x-shellscript")
    assert exc_info.value.code == "unsupported_mime_type"


def test_reject_size_exceeded(store, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config.settings, "artifact_max_bytes", 100)

    oversized = VALID_JPEG + b"extra_padding_bytes" * 10
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("app_9999", "id_front", oversized, "image/jpeg")
    assert exc_info.value.code == "file_too_large"


def test_reject_path_traversal_in_applicant_id(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("../../etc", "id_front", VALID_JPEG, "image/jpeg")
    assert exc_info.value.code == "invalid_identifier"


def test_reject_path_traversal_in_artifact_type(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.save_artifact("app_1001", "../hack", VALID_JPEG, "image/jpeg")
    assert exc_info.value.code == "invalid_identifier"


def test_reject_path_traversal_on_load(store):
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.load_artifact("file:///etc/passwd")
    assert exc_info.value.code == "path_traversal_detected"


def test_delete_artifact(store):
    uri = store.save_artifact("app_1005", "id_front", VALID_JPEG, "image/jpeg")
    assert store.load_artifact(uri) == VALID_JPEG

    assert store.delete_artifact(uri) is True
    with pytest.raises(ArtifactStorageError) as exc_info:
        store.load_artifact(uri)
    assert exc_info.value.code == "artifact_not_found"
