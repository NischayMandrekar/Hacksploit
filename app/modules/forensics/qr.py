import base64
import logging
import zlib

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from app.modules.forensics.schemas import SecureQRVerificationResult

logger = logging.getLogger(__name__)

# Canonical field index order in UIDAI Secure QR Specification
FIELD_INDEX_MAP = {
    0: "email_mobile_present",
    1: "reference_id",
    2: "name",
    3: "dob",
    4: "gender",
    5: "care_of",
    6: "district",
    7: "landmark",
    8: "house",
    9: "location",
    10: "pincode",
    11: "post_office",
    12: "state",
    13: "street",
    14: "sub_district",
}


def generate_test_uidai_keypair() -> tuple[RSAPrivateKey, RSAPublicKey]:
    """Generates an in-memory 2048-bit RSA keypair simulating the UIDAI root authority."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def create_signed_secure_qr_payload(
    private_key: RSAPrivateKey,
    demographics: dict[str, str],
) -> bytes:
    """
    Constructs and signs a canonical UIDAI Secure QR code byte payload.
    Used for integration tests, local benchmarks, and seed data generation.
    """
    fields = [
        demographics.get("email_mobile_present", "3"),
        demographics.get("reference_id", "901220261028120000"),
        demographics.get("name", "Arjun Sharma"),
        demographics.get("dob", "24-07-1992"),
        demographics.get("gender", "M"),
        demographics.get("care_of", "C/O Sharma"),
        demographics.get("district", "New Delhi"),
        demographics.get("landmark", "Near Metro"),
        demographics.get("house", "Flat 101"),
        demographics.get("location", "Connaught Place"),
        demographics.get("pincode", "110001"),
        demographics.get("post_office", "Connaught Place PO"),
        demographics.get("state", "Delhi"),
        demographics.get("street", "Barakhamba Road"),
        demographics.get("sub_district", "New Delhi"),
    ]

    data_bytes = b"\xff".join([f.encode("utf-8") for f in fields])
    signature = private_key.sign(data_bytes, padding.PKCS1v15(), hashes.SHA256())
    full_stream = data_bytes + signature
    return zlib.compress(full_stream)


class AadhaarSecureQRVerifier:
    """
    100% Offline cryptographic verifier for UIDAI Aadhaar Secure QR codes.
    Requires ZERO external API calls or UIDAI CIDR connectivity.
    Validates the 2048-bit RSA PKCS#1 v1.5 SHA-256 digital signature embedded in the QR data.
    """

    def __init__(self, public_key: RSAPublicKey | None = None) -> None:
        self.public_key = public_key

    def verify_qr_bytes(
        self,
        raw_qr_bytes: bytes,
        public_key: RSAPublicKey | None = None,
    ) -> SecureQRVerificationResult:
        """
        Parses decompressed QR data, extracts the 256-byte digital signature,
        and cryptographically verifies the demographic payload against the trusted public key.
        """
        active_key = public_key or self.public_key
        if not active_key:
            return SecureQRVerificationResult(
                status="UNAVAILABLE",
                is_valid=False,
                reason_codes=["NO_TRUSTED_KEY"],
                details={"error": "No trusted UIDAI public key configured for verification."},
            )

        if not raw_qr_bytes:
            return SecureQRVerificationResult(
                status="UNAVAILABLE",
                is_valid=False,
                reason_codes=["EMPTY_QR_PAYLOAD"],
                details={"error": "QR payload is empty."},
            )

        # 1. Attempt decompression (Zlib / Gzip)
        decompressed: bytes
        try:
            decompressed = zlib.decompress(raw_qr_bytes)
        except Exception:
            # If not zlib compressed, check if already decompressed
            decompressed = raw_qr_bytes

        # 2. Extract 256-byte RSA signature (RSA-2048 signature is exactly 256 bytes)
        if len(decompressed) <= 256:
            return SecureQRVerificationResult(
                status="INVALID_FORMAT",
                is_valid=False,
                reason_codes=["QR_PAYLOAD_TOO_SHORT"],
                details={"length": len(decompressed), "error": "Payload shorter than 256-byte signature."},
            )

        data_bytes = decompressed[:-256]
        signature_bytes = decompressed[-256:]

        # 3. Cryptographic Signature Verification
        try:
            active_key.verify(
                signature_bytes,
                data_bytes,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            is_signature_valid = True
        except InvalidSignature:
            logger.warning("Aadhaar Secure QR digital signature mismatch; payload has been tampered with.")
            return SecureQRVerificationResult(
                status="TAMPERED",
                is_valid=False,
                reason_codes=["R7_SECURE_QR_TAMPERED"],
                details={"error": "Digital signature verification failed. Data has been modified."},
            )
        except Exception as e:
            return SecureQRVerificationResult(
                status="TAMPERED",
                is_valid=False,
                reason_codes=["R7_SECURE_QR_TAMPERED"],
                details={"error": f"Cryptographic verification error: {str(e)}"},
            )

        # 4. Parse Delimited Demographic Fields (0xFF separator)
        field_parts = data_bytes.split(b"\xff")
        demographics: dict[str, str] = {}
        for idx, part in enumerate(field_parts):
            field_name = FIELD_INDEX_MAP.get(idx, f"field_{idx}")
            try:
                demographics[field_name] = part.decode("utf-8", errors="replace").strip()
            except Exception:
                demographics[field_name] = ""

        return SecureQRVerificationResult(
            status="VERIFIED",
            is_valid=is_signature_valid,
            reason_codes=["SECURE_QR_VERIFIED"],
            extracted_demographics=demographics,
            signature_algorithm="RSA-2048-SHA256",
            details={
                "reference_id": demographics.get("reference_id", ""),
                "name": demographics.get("name", ""),
                "dob": demographics.get("dob", ""),
                "gender": demographics.get("gender", ""),
                "pincode": demographics.get("pincode", ""),
                "state": demographics.get("state", ""),
            },
        )

    def verify_from_image(
        self,
        image_bytes: bytes,
        public_key: RSAPublicKey | None = None,
    ) -> SecureQRVerificationResult:
        """
        Detects QR code inside document image using OpenCV and verifies its offline signature.
        """
        try:
            import cv2

            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return SecureQRVerificationResult(
                    status="UNAVAILABLE",
                    is_valid=False,
                    reason_codes=["IMAGE_DECODE_FAILED"],
                    details={"error": "Could not decode image bytes."},
                )

            detector = cv2.QRCodeDetector()
            decoded_info, points, _ = detector.detectAndDecode(img)
            if not decoded_info or points is None:
                return SecureQRVerificationResult(
                    status="UNAVAILABLE",
                    is_valid=False,
                    reason_codes=["QR_NOT_FOUND"],
                    details={"info": "No QR code detected in document image."},
                )

            # Try parsing decoded_info as base64 or raw bytes
            try:
                payload = base64.b64decode(decoded_info)
            except Exception:
                payload = decoded_info.encode("latin-1", errors="ignore")

            return self.verify_qr_bytes(payload, public_key=public_key)
        except Exception as e:
            logger.warning("QR extraction from image encountered error: %s", str(e))
            return SecureQRVerificationResult(
                status="UNAVAILABLE",
                is_valid=False,
                reason_codes=["QR_EXTRACTION_ERROR"],
                details={"error": str(e)},
            )
