"""
OCR and Post-Processing Module (Part2.py renamed)
Advanced OCR processing and Malaysian state identification for license plates
"""

import cv2
import re
import numpy as np
import pytesseract
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class OCRResult:
    """OCR result data structure"""
    text: str
    state_label: str
    confidence: float
    box: Tuple[int, int, int, int]
    raw: str = ""
    psm_used: int = 0

# Malaysian license plate patterns
PLATE_REGEX = re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})$")

STATE_PREFIX = {
    "A": "Perak", "B": "Selangor", "C": "Pahang", "D": "Kelantan", "F": "Putrajaya",
    "J": "Johor", "K": "Kedah", "L": "Labuan", "M": "Melaka", "N": "Negeri Sembilan",
    "P": "Pulau Pinang", "Q": "Sarawak", "R": "Perlis", "S": "Sabah", "T": "Terengganu",
    "V": "Kuala Lumpur (new)", "W": "Kuala Lumpur (old)"
}

SPECIAL_PREFIX = {
    "KV": "Langkawi (Kedah Special)",
    "CD": "Diplomatic",
    "Z": "Armed Forces"
}

# Character correction mappings
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}

def _clean_text(t: str) -> str:
    """Clean and normalize text for OCR processing"""
    t = t.upper()
    t = re.sub(r"[\s\-\._:]", "", t)
    return "".join(ch for ch in t if ch.isalnum())

def _correct_ambiguous(text: str) -> str:
    """Correct commonly misrecognized characters"""
    if not text:
        return text
    t = list(text.upper())
    for i, ch in enumerate(t):
        if ch in LETTER_TO_DIGIT:
            t[i] = LETTER_TO_DIGIT[ch]
    return "".join(t)

def _split_blocks(text: str):
    """Split license plate text into prefix, digits, and suffix"""
    m = PLATE_REGEX.match(text)
    if m:
        return m.group(1), m.group(2), m.group(3) or ""
    
    t2 = _correct_ambiguous(text)
    m2 = PLATE_REGEX.match(t2)
    if m2:
        return m2.group(1), m2.group(2), m2.group(3) or ""
    
    return None, None, None

def _map_state(prefix: str) -> str:
    """Map license plate prefix to Malaysian state"""
    if not prefix:
        return "Unknown"
    
    if prefix[:2] in SPECIAL_PREFIX:
        return SPECIAL_PREFIX[prefix[:2]]
    
    if prefix[0] == "Z":
        return SPECIAL_PREFIX["Z"]
    
    return STATE_PREFIX.get(prefix[0], "Unknown")

def _score(prefix, digits, suffix, raw_len):
    """Calculate confidence score for OCR result"""
    s = 0.0
    if prefix:
        s += 0.25
    if digits and 1 <= len(digits) <= 4:
        s += 0.25
    if suffix and len(suffix) <= 2:
        s += 0.1
    if prefix and (prefix[:2] in SPECIAL_PREFIX or prefix[0] in STATE_PREFIX):
        s += 0.2
    if raw_len >= 4:
        s += 0.1
    return min(1.0, s)

def _binarize_for_ocr(roi_bgr):
    """Prepare ROI for OCR processing"""
    roi = cv2.resize(roi_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 75, 75)
    binm = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 13)
    return binm

def _tesseract_try(bin_img, psms=[7, 8, 6]):
    """Try different PSM modes for best OCR result"""
    best, psm_used = "", 0
    for psm in psms:
        cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        txt = pytesseract.image_to_string(bin_img, config=cfg)
        txt = _clean_text(txt)
        if len(txt) > len(best):
            best, psm_used = txt, psm
    return best, psm_used

def _normalize(raw: str):
    """Normalize and validate OCR result"""
    t = _clean_text(raw)
    prefix, digits, suffix = _split_blocks(t)
    if not prefix:
        return t, "Unknown", 0.0
    
    norm = f"{prefix}{digits}{suffix}"
    state = _map_state(prefix)
    conf = _score(prefix, digits, suffix, len(t))
    return norm, state, conf

def ocr_from_bbox(image_bgr, box) -> OCRResult:
    """Extract text from bounding box using OCR"""
    x, y, w, h = box
    roi = image_bgr[y:y+h, x:x+w]
    
    if roi.size == 0:
        return OCRResult("", "Unknown", 0.0, box)
    
    binm = _binarize_for_ocr(roi)
    raw, psm = _tesseract_try(binm)
    norm, state, conf = _normalize(raw)
    
    return OCRResult(norm, state, conf, box, raw, psm)

def process_detections(image_bgr, boxes: List[Tuple[int, int, int, int]], topk=3):
    """Process multiple detection boxes and return top results"""
    results = [ocr_from_bbox(image_bgr, b) for b in boxes]
    results.sort(key=lambda r: (r.confidence, len(r.text)), reverse=True)
    return results[:topk], results[0] if results else None

def clean_and_map(text: str) -> Tuple[str, str]:
    """Clean OCR text and map to state information (compatibility function)"""
    if not text:
        return "", "Unknown"
    
    # Clean text
    clean_text = _clean_text(text)
    
    # Try to split into blocks
    prefix, digits, suffix = _split_blocks(clean_text)
    
    if prefix:
        # Map state
        state = _map_state(prefix)
        # Reconstruct full text
        full_text = prefix + digits + suffix
        return full_text, state
    else:
        # If no valid pattern, return cleaned text
        return clean_text, "Unknown"

def enhance_ocr_result(text: str, confidence: float = 0.0) -> OCRResult:
    """Enhance OCR result with additional processing"""
    if not text:
        return text, "Unknown", confidence
    
    # Apply character correction
    corrected = _correct_ambiguous(text)
    
    # Split and validate
    prefix, digits, suffix = _split_blocks(corrected)
    if not prefix:
        return text, "Unknown", confidence * 0.5
    
    # Reconstruct normalized text
    normalized = f"{prefix}{digits}{suffix}"
    state = _map_state(prefix)
    
    # Adjust confidence based on validation
    new_conf = confidence
    if len(normalized) >= 4:
        new_conf += 0.1
    if state != "Unknown":
        new_conf += 0.1
    
    return normalized, state, min(1.0, new_conf)

# Test function for standalone testing
if __name__ == "__main__":
    print("OCR Processor Module")
    print("Import this module to use advanced OCR processing functionality")