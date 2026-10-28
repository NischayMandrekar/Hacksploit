import logging
import re
from datetime import date

from app.core.forensics import EvidenceStatus, ExtractedDocument, TamperingSignal
from app.modules.forensics.schemas import FieldContradictionReport

logger = logging.getLogger(__name__)

# --- Verhoeff Algorithm Tables (for Aadhaar 12-digit Checksum Validation) ---
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def validate_verhoeff(number_str: str) -> bool:
    """
    Validates a 12-digit Indian Aadhaar number using the official Verhoeff checksum algorithm.
    Detects all single-digit errors and over 95% of adjacent transposition errors.
    """
    clean = re.sub(r"\D", "", number_str)
    if len(clean) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(clean)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


def compute_verhoeff_check_digit(prefix_11_digits: str) -> int:
    """Computes the 12th Verhoeff check digit for an 11-digit Aadhaar prefix."""
    clean = re.sub(r"\D", "", prefix_11_digits)
    c = 0
    for i, item in enumerate(reversed(clean)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(item)]]
    return _VERHOEFF_INV[c]


# --- Indian PAN (Permanent Account Number) Validation ---
PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


def validate_pan_format(pan_str: str, surname: str | None = None) -> bool:
    """
    Validates an Indian PAN card number format.
    Optionally checks that the 5th character matches the first letter of the applicant's surname.
    """
    clean = pan_str.strip().upper()
    if not PAN_REGEX.match(clean):
        return False
    if surname and len(surname.strip()) > 0:
        expected_letter = surname.strip()[0].upper()
        if clean[4] != expected_letter:
            logger.info("PAN 5th character '%s' does not match surname initial '%s'", clean[4], expected_letter)
            return False
    return True


# --- Levenshtein Distance & Fuzzy Demographic Similarity ---

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the minimum edit distance between two strings (insertions, deletions, substitutions)."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0[:] = v1[:]

    return v1[len(s2)]


_HONORIFICS = re.compile(r"\b(mr|mrs|ms|shri|smt|dr|prof|kumar|kumari)\b\.?", re.IGNORECASE)


def normalize_name_string(name: str) -> str:
    """Strips honorifics, punctuation, and extraneous spaces for OCR name comparison."""
    clean = _HONORIFICS.sub("", name)
    clean = re.sub(r"[^\w\s]", "", clean)
    return " ".join(clean.lower().split())


def name_similarity_ratio(claimed: str, extracted: str) -> float:
    """
    Calculates fuzzy name similarity ratio between claimed and extracted names.
    Employs token-sort matching to handle name order transposition (e.g. 'Rahul Sharma' vs 'Sharma Rahul').
    Returns a score between 0.0 (total mismatch) and 1.0 (exact match).
    """
    norm_claimed = normalize_name_string(claimed)
    norm_extracted = normalize_name_string(extracted)

    if not norm_claimed or not norm_extracted:
        return 0.0
    if norm_claimed == norm_extracted:
        return 1.0

    # 1. Direct character similarity
    max_len = max(len(norm_claimed), len(norm_extracted))
    direct_dist = levenshtein_distance(norm_claimed, norm_extracted)
    direct_ratio = max(0.0, 1.0 - (direct_dist / max_len))

    # 2. Token-sort similarity
    tokens_claimed = sorted(norm_claimed.split())
    tokens_extracted = sorted(norm_extracted.split())
    sorted_claimed = " ".join(tokens_claimed)
    sorted_extracted = " ".join(tokens_extracted)

    sort_max_len = max(len(sorted_claimed), len(sorted_extracted))
    sort_dist = levenshtein_distance(sorted_claimed, sorted_extracted)
    token_sort_ratio = max(0.0, 1.0 - (sort_dist / sort_max_len))

    return max(direct_ratio, token_sort_ratio)


# --- Date Normalization & Matching ---

_DATE_PATTERNS = [
    re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"),  # YYYY-MM-DD
    re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})"),  # DD-MM-YYYY
]


def normalize_dob_string(dob_str: str) -> str | None:
    """Parses various date string formats into standard YYYY-MM-DD ISO date."""
    clean = dob_str.strip()
    for pattern in _DATE_PATTERNS:
        match = pattern.search(clean)
        if match:
            g1, g2, g3 = match.groups()
            try:
                if len(g1) == 4:  # YYYY-MM-DD
                    d = date(int(g1), int(g2), int(g3))
                else:  # DD-MM-YYYY
                    d = date(int(g3), int(g2), int(g1))
                return d.isoformat()
            except ValueError:
                continue
    return None


def compare_dobs(claimed_dob: str, extracted_dob: str) -> tuple[float, bool]:
    """
    Compares claimed DOB against extracted DOB.
    Returns (score, is_match).
    Full match: 1.0, True.
    Same year but day/month missing or divergent: 0.5, False.
    Divergent year: 0.0, False.
    """
    norm_claimed = normalize_dob_string(claimed_dob)
    norm_extracted = normalize_dob_string(extracted_dob)

    if norm_claimed and norm_extracted:
        if norm_claimed == norm_extracted:
            return 1.0, True
        # Check if at least birth year matches
        if norm_claimed[:4] == norm_extracted[:4]:
            return 0.5, False
        return 0.0, False

    # Check 4-digit year fallback (e.g. if document only displays Year of Birth)
    claimed_year = re.search(r"\b(19\d{2}|20\d{2})\b", claimed_dob)
    extracted_year = re.search(r"\b(19\d{2}|20\d{2})\b", extracted_dob)
    if claimed_year and extracted_year:
        if claimed_year.group(1) == extracted_year.group(1):
            return 0.7, True
        return 0.0, False

    return 0.0, False


# --- OCR Field Extraction Engine ---

class OCRExtractor:
    """
    Extracts text and demographic fields from identity document scans.
    Uses pytesseract if the local system has OCR binaries available;
    provides deterministic regex parsing and heuristic fallback.
    """

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.tesseract_cmd = tesseract_cmd

    def extract_text_from_image(self, document_bytes: bytes) -> str:
        """Extracts raw text from image bytes."""
        try:
            import io
            from PIL import Image
            import pytesseract

            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

            img = Image.open(io.BytesIO(document_bytes))
            return pytesseract.image_to_string(img)
        except Exception as e:
            logger.info("Direct OCR extraction unavailable (%s); falling back to raw byte text parsing", str(e))
            # Fallback text extraction for plaintext / test documents
            return document_bytes.decode("utf-8", errors="ignore")

    def parse_document_fields(self, text: str) -> dict[str, str]:
        """Extracts structured fields (Aadhaar, PAN, DOB, Name) from OCR text."""
        fields: dict[str, str] = {}

        # 1. 12-digit Aadhaar number (with or without spaces: 1234 5678 9012)
        aadhaar_match = re.search(r"\b(\d{4}\s\d{4}\s\d{4}|\d{12})\b", text)
        if aadhaar_match:
            fields["aadhaar"] = re.sub(r"\s", "", aadhaar_match.group(1))

        # 2. 10-character PAN number
        pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
        if pan_match:
            fields["pan"] = pan_match.group(1)

        # 3. DOB extraction
        dob_match = re.search(
            r"(?:DOB|Date of Birth|Birth|DOB:)\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})",
            text,
            re.IGNORECASE,
        )
        if dob_match:
            fields["dob"] = dob_match.group(1)
        else:
            # Standalone date pattern
            date_fallback = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", text)
            if date_fallback:
                fields["dob"] = date_fallback.group(1)

        # 4. Name extraction
        name_match = re.search(r"(?:Name|Name:)\s*([A-Za-z\s]{3,40})", text, re.IGNORECASE)
        if name_match:
            fields["name"] = name_match.group(1).strip()
        else:
            # Heuristic: Find first line with 2+ capitalized alphabetic words
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines:
                if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$", line):
                    fields["name"] = line
                    break

        return fields

    def extract_fields(self, document_bytes: bytes, mime_type: str) -> ExtractedDocument:
        """Full extraction fulfilling ForensicExtractor protocol."""
        raw_text = self.extract_text_from_image(document_bytes)
        fields = self.parse_document_fields(raw_text)
        return ExtractedDocument(
            raw_text=raw_text,
            fields=fields,
            extraction_method="ocr",
            confidence=0.85 if fields else 0.2,
        )

    def assess_tampering(self, document_bytes: bytes, mime_type: str) -> TamperingSignal:
        """Baseline tampering signal."""
        return TamperingSignal(
            signal_name="ocr_consistency",
            status=EvidenceStatus.PRESENT,
            score=0.9,
            details={"method": "regex_ocr_parser"},
        )


# --- Field Contradiction Detector ---

class FieldContradictionDetector:
    """
    Cross-checks extracted document fields against claimed applicant onboarding data.
    Generates actionable reason codes:
    - R3_NAME_CONTRADICTION: Claimed name differs significantly from document OCR.
    - R4_DOB_CONTRADICTION: Claimed date of birth does not match document date.
    - R5_INVALID_ID_CHECKSUM: National ID fails official mathematical checksum (e.g. Verhoeff).
    """

    def __init__(
        self,
        name_similarity_threshold: float = 0.70,
    ) -> None:
        self.name_similarity_threshold = name_similarity_threshold

    def evaluate(
        self,
        claimed_name: str,
        claimed_dob: str,
        claimed_id_number: str,
        extracted_fields: dict[str, str],
        id_type: str = "aadhaar",
    ) -> FieldContradictionReport:
        contradictions: list[str] = []
        reasons: list[str] = []

        # 1. National ID Checksum Validation
        id_checksum_valid = True
        clean_id = re.sub(r"[\s\-]", "", claimed_id_number)

        if id_type.lower() == "aadhaar":
            id_checksum_valid = validate_verhoeff(clean_id)
            if not id_checksum_valid:
                contradictions.append("R5_INVALID_ID_CHECKSUM")
                reasons.append(f"Aadhaar number '{clean_id}' failed mathematical Verhoeff checksum validation.")
        elif id_type.lower() == "pan":
            id_checksum_valid = validate_pan_format(clean_id)
            if not id_checksum_valid:
                contradictions.append("R5_INVALID_ID_CHECKSUM")
                reasons.append(f"PAN number '{clean_id}' does not match official 10-character PAN pattern.")

        # 2. Extracted vs Claimed ID Number Matching
        id_match = True
        doc_id = extracted_fields.get("aadhaar") or extracted_fields.get("pan") or extracted_fields.get("id_number")
        if doc_id:
            clean_doc_id = re.sub(r"[\s\-]", "", doc_id)
            if clean_doc_id != clean_id:
                id_match = False
                reasons.append(f"Extracted document ID '{clean_doc_id}' does not match claimed ID '{clean_id}'.")

        # 3. Name Similarity Matching
        extracted_name = extracted_fields.get("name")
        if extracted_name:
            name_score = name_similarity_ratio(claimed_name, extracted_name)
            if name_score < self.name_similarity_threshold:
                contradictions.append("R3_NAME_CONTRADICTION")
                reasons.append(
                    f"Name mismatch: claimed '{claimed_name}' vs extracted '{extracted_name}' "
                    f"(similarity {round(name_score, 2)} < {self.name_similarity_threshold})."
                )
        else:
            name_score = 1.0  # Unavailable extracted name does not falsely penalize

        # 4. Date of Birth Matching
        extracted_dob = extracted_fields.get("dob")
        if extracted_dob:
            dob_score, dob_matched = compare_dobs(claimed_dob, extracted_dob)
            if not dob_matched:
                contradictions.append("R4_DOB_CONTRADICTION")
                reasons.append(f"Date of birth mismatch: claimed '{claimed_dob}' vs extracted '{extracted_dob}'.")
        else:
            dob_score = 1.0

        return FieldContradictionReport(
            name_match_score=name_score,
            dob_match_score=dob_score,
            id_checksum_valid=id_checksum_valid,
            id_match=id_match,
            extracted_name=extracted_name,
            extracted_dob=extracted_dob,
            extracted_id_number=doc_id,
            claimed_name=claimed_name,
            claimed_dob=claimed_dob,
            claimed_id_number=claimed_id_number,
            contradictions_detected=contradictions,
            reasons=reasons,
        )
