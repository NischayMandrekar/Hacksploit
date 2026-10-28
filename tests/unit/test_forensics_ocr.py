import zlib
from app.modules.forensics.ocr import (
    FieldContradictionDetector,
    OCRExtractor,
    compare_dobs,
    compute_verhoeff_check_digit,
    levenshtein_distance,
    name_similarity_ratio,
    normalize_dob_string,
    validate_pan_format,
    validate_verhoeff,
)
from app.modules.forensics.qr import (
    AadhaarSecureQRVerifier,
    create_signed_secure_qr_payload,
    generate_test_uidai_keypair,
)


def test_verhoeff_checksum_algorithm():
    """
    Verifies mathematical Verhoeff checksum algorithm for 12-digit Indian Aadhaar numbers.
    Tests check digit computation, single-digit corruption, and adjacent transposition detection.
    """
    # 1. Compute check digit for valid 11-digit prefix
    prefix = "36759834521"
    check_digit = compute_verhoeff_check_digit(prefix)
    valid_aadhaar = prefix + str(check_digit)

    assert len(valid_aadhaar) == 12
    assert validate_verhoeff(valid_aadhaar) is True

    # 2. Corrupt single digit
    corrupted_digit = valid_aadhaar[:5] + str((int(valid_aadhaar[5]) + 1) % 10) + valid_aadhaar[6:]
    assert validate_verhoeff(corrupted_digit) is False

    # 3. Adjacent transposition error (swap two adjacent digits)
    transposed = list(valid_aadhaar)
    transposed[3], transposed[4] = transposed[4], transposed[3]
    transposed_aadhaar = "".join(transposed)
    assert validate_verhoeff(transposed_aadhaar) is False

    # 4. Invalid length check
    assert validate_verhoeff("12345") is False
    assert validate_verhoeff("1234567890123") is False


def test_pan_format_validation():
    """Verifies Indian PAN card format regex and surname initial matching."""
    # Standard valid PAN
    assert validate_pan_format("ABCDE1234F") is True
    assert validate_pan_format("abcde1234f") is True  # Case insensitive

    # Surname initial matching (5th char must match surname's first letter)
    assert validate_pan_format("ABCDE1234F", surname="Edwin") is True
    assert validate_pan_format("ABCDE1234F", surname="Sharma") is False  # Expected S, found E

    # Invalid structures
    assert validate_pan_format("ABCD12345F") is False  # 4 letters, 5 digits
    assert validate_pan_format("ABCDE12345") is False   # No trailing letter
    assert validate_pan_format("12345ABCDE") is False
    assert validate_pan_format("ABCDE1234") is False    # Too short


def test_levenshtein_and_name_similarity():
    """
    Verifies Levenshtein distance and fuzzy name matching with token sorting and honorific stripping.
    """
    # 1. Levenshtein edit distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("rohit", "rohit") == 0
    assert levenshtein_distance("", "test") == 4

    # 2. Exact match
    assert name_similarity_ratio("Rahul Sharma", "Rahul Sharma") == 1.0

    # 3. Honorific stripping
    assert name_similarity_ratio("Mr. Rahul Sharma", "Rahul Sharma") == 1.0
    assert name_similarity_ratio("Shri Rahul Sharma", "Rahul Sharma") == 1.0

    # 4. Token-sort / Transposition (Indian documents often swap Surname and Given name)
    assert name_similarity_ratio("Rahul Sharma", "Sharma Rahul") == 1.0

    # 5. Minor OCR noise / typo
    ratio_noisy = name_similarity_ratio("Rahul Sharma", "Rahull Sharma")
    assert ratio_noisy >= 0.85

    # 6. Complete contradiction / different identity
    ratio_mismatch = name_similarity_ratio("Rahul Sharma", "Vikramaditya Rathore")
    assert ratio_mismatch < 0.40


def test_dob_normalization_and_comparison():
    """Verifies cross-format date parsing and matching."""
    # 1. Normalization
    assert normalize_dob_string("1992-07-24") == "1992-07-24"
    assert normalize_dob_string("24/07/1992") == "1992-07-24"
    assert normalize_dob_string("24-07-1992") == "1992-07-24"

    # 2. Exact match
    score, matched = compare_dobs("1992-07-24", "24/07/1992")
    assert score == 1.0
    assert matched is True

    # 3. Same year, different day/month
    score_partial, matched_partial = compare_dobs("1992-07-24", "01/01/1992")
    assert score_partial == 0.5
    assert matched_partial is False

    # 4. Divergent year
    score_diff, matched_diff = compare_dobs("1992-07-24", "1985-07-24")
    assert score_diff == 0.0
    assert matched_diff is False


def test_field_contradiction_detector_clean():
    """Verifies that matching applicant demographics and valid IDs pass with no contradictions."""
    detector = FieldContradictionDetector(name_similarity_threshold=0.70)

    # Valid Aadhaar
    prefix = "45678912345"
    valid_aadhaar = prefix + str(compute_verhoeff_check_digit(prefix))

    extracted = {
        "name": "Arjun Sharma",
        "dob": "24/07/1992",
        "aadhaar": valid_aadhaar,
    }

    report = detector.evaluate(
        claimed_name="Arjun Sharma",
        claimed_dob="1992-07-24",
        claimed_id_number=valid_aadhaar,
        extracted_fields=extracted,
        id_type="aadhaar",
    )

    assert report.has_contradiction is False
    assert report.name_match_score == 1.0
    assert report.dob_match_score == 1.0
    assert report.id_checksum_valid is True
    assert report.id_match is True
    assert len(report.contradictions_detected) == 0


def test_field_contradiction_detector_violations():
    """
    Verifies detection of:
    - R3_NAME_CONTRADICTION (claimed vs extracted name mismatch)
    - R4_DOB_CONTRADICTION (claimed vs extracted DOB mismatch)
    - R5_INVALID_ID_CHECKSUM (Aadhaar Verhoeff failure)
    """
    detector = FieldContradictionDetector(name_similarity_threshold=0.70)

    # Corrupted Aadhaar
    corrupt_aadhaar = "123456789012"  # Fails Verhoeff checksum

    extracted = {
        "name": "Completely Different Person",
        "dob": "1980-01-01",
        "aadhaar": "999988887777",
    }

    report = detector.evaluate(
        claimed_name="Rahul Sharma",
        claimed_dob="1995-10-15",
        claimed_id_number=corrupt_aadhaar,
        extracted_fields=extracted,
        id_type="aadhaar",
    )

    assert report.has_contradiction is True
    assert "R3_NAME_CONTRADICTION" in report.contradictions_detected
    assert "R4_DOB_CONTRADICTION" in report.contradictions_detected
    assert "R5_INVALID_ID_CHECKSUM" in report.contradictions_detected
    assert report.id_checksum_valid is False
    assert report.id_match is False


def test_aadhaar_secure_qr_offline_verification_clean():
    """
    Verifies offline RSA-2048 verification on a cryptographically genuine Aadhaar Secure QR code.
    Requires 0 external API calls.
    """
    private_key, public_key = generate_test_uidai_keypair()
    verifier = AadhaarSecureQRVerifier(public_key=public_key)

    demographics = {
        "name": "Siddharth Verma",
        "dob": "20-06-1985",
        "gender": "M",
        "reference_id": "9876202610280001",
        "pincode": "110001",
        "state": "Delhi",
    }

    signed_qr_payload = create_signed_secure_qr_payload(private_key, demographics)

    result = verifier.verify_qr_bytes(signed_qr_payload)
    assert result.status == "VERIFIED"
    assert result.is_valid is True
    assert "SECURE_QR_VERIFIED" in result.reason_codes
    assert result.extracted_demographics["name"] == "Siddharth Verma"
    assert result.extracted_demographics["dob"] == "20-06-1985"
    assert result.signature_algorithm == "RSA-2048-SHA256"


def test_aadhaar_secure_qr_tampering_detection():
    """
    Verifies that altering demographic data in a signed QR code without the private key
    causes digital signature verification to fail and triggers R7_SECURE_QR_TAMPERED.
    """
    private_key, public_key = generate_test_uidai_keypair()
    verifier = AadhaarSecureQRVerifier(public_key=public_key)

    demographics = {
        "name": "Original Citizen",
        "dob": "10-10-1990",
    }
    signed_qr_payload = create_signed_secure_qr_payload(private_key, demographics)

    # Attacker alters the uncompressed data stream (e.g. changes name)
    decompressed = zlib.decompress(signed_qr_payload)
    tampered_decompressed = decompressed.replace(b"Original Citizen", b"Forged Attacker Name")
    tampered_payload = zlib.compress(tampered_decompressed)

    result = verifier.verify_qr_bytes(tampered_payload)
    assert result.status == "TAMPERED"
    assert result.is_valid is False
    assert "R7_SECURE_QR_TAMPERED" in result.reason_codes


def test_aadhaar_secure_qr_wrong_key():
    """Verifies that an untrusted or mismatched public key flags tampering/signature failure."""
    private_key_a, _ = generate_test_uidai_keypair()
    _, public_key_b = generate_test_uidai_keypair()  # Different keypair

    demographics = {"name": "Test User"}
    signed_qr = create_signed_secure_qr_payload(private_key_a, demographics)

    verifier = AadhaarSecureQRVerifier(public_key=public_key_b)
    result = verifier.verify_qr_bytes(signed_qr)

    assert result.status == "TAMPERED"
    assert result.is_valid is False
    assert "R7_SECURE_QR_TAMPERED" in result.reason_codes


def test_ocr_extractor_text_parsing():
    """Verifies regex field parsing from document text."""
    extractor = OCRExtractor()
    sample_ocr_text = """
    GOVERNMENT OF INDIA
    Unique Identification Authority of India
    To:
    Arjun Sharma
    DOB: 24/07/1992
    Male
    1234 5678 9012
    Help: 1947
    """
    fields = extractor.parse_document_fields(sample_ocr_text)

    assert fields["aadhaar"] == "123456789012"
    assert fields["dob"] == "24/07/1992"
    assert "Arjun" in fields["name"]
