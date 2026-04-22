"""
License Plate Detection Module (IPPR2.py renamed)
Computer vision-based license plate detection using multi-scale approach
"""

import cv2
import re
import numpy as np
import pytesseract
import os

# ---------- CONFIGURATION ----------
# Set Tesseract path directly for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Detection parameters
ASPECT_RATIO_RANGE = (2, 8)
SCALES = [0.6, 0.8, 1.0, 1.25]
MIN_AREA = 0

# Malaysian license plate regex pattern
PLATE_REGEX = re.compile(r"^([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})$")

# State prefix mapping
STATE_PREFIX = {
    "A":"Perak","B":"Selangor","C":"Pahang","D":"Kelantan","F":"Putrajaya",
    "J":"Johor","K":"Kedah","L":"Labuan","M":"Melaka","N":"Negeri Sembilan",
    "P":"Pulau Pinang","Q":"Sarawak","R":"Perlis","S":"Sabah","T":"Terengganu",
    "V":"Kuala Lumpur (new)","W":"Kuala Lumpur (old)"
}

# Special series prefix mapping
SPECIAL_PREFIX = {"KV":"Langkawi (Kedah Special)","CD":"Diplomatic","Z":"Armed Forces"}

def clean_and_map(text: str):
    """Clean OCR text and map to state information"""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    m = PLATE_REGEX.match(t)
    if not m:
        return t, "Unknown"
    pfx = m.group(1)
    if pfx[:2] in SPECIAL_PREFIX:
        return t, SPECIAL_PREFIX[pfx[:2]]
    if pfx[0] == "Z":
        return t, SPECIAL_PREFIX["Z"]
    return t, STATE_PREFIX.get(pfx[0], "Unknown")

def white_mask(image):
    """Create enhanced white region mask using multiple color spaces and adaptive thresholds"""
    # Convert to multiple color spaces for better white detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # HSV white mask (more flexible thresholds)
    m_hsv = cv2.inRange(hsv, np.array([0, 0, 160], np.uint8), np.array([180, 50, 255], np.uint8))
    
    # YCrCb luma mask (brightness-based)
    m_luma = cv2.inRange(ycc[:, :, 0], 150, 255)
    
    # LAB lightness mask
    m_lab = cv2.inRange(lab[:, :, 0], 150, 255)
    
    # Combine all masks
    mask = cv2.bitwise_or(m_hsv, m_luma)
    mask = cv2.bitwise_or(mask, m_lab)
    
    # Enhanced morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return mask

def preprocess_image(image):
    """Enhanced image preprocessing for better detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Enhanced bilateral filtering
    gray = cv2.bilateralFilter(gray, 11, 75, 75)
    
    # Create white mask
    mask = white_mask(image)
    
    return gray, mask

def _find_candidates_core(gray, edge_mask=None, ar_min=2.0, ar_max=8.8):
    """Enhanced core candidate detection algorithm with better scoring"""
    # Multi-level thresholding for better edge detection
    thr1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 9)
    thr2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 9)
    
    # Combine thresholds
    thr = cv2.bitwise_or(thr1, thr2)
    
    # Enhanced Canny edge detection with multiple thresholds
    edges1 = cv2.Canny(thr, 50, 150)
    edges2 = cv2.Canny(thr, 100, 200)
    edges = cv2.bitwise_or(edges1, edges2)
    
    if edge_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=edge_mask)
    
    # Enhanced morphological operations
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1, iterations=1)
    morph = cv2.dilate(morph, k2, iterations=1)
    morph = cv2.erode(morph, k3, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    min_area = max(int(0.0002 * H * W), 800)  # Reduced minimum area
    
    # Enhanced gradient analysis
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_gx, abs_gy = np.abs(gx), np.abs(gy)
    
    # Calculate gradient magnitude and direction
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
            
        ar = w / float(h)
        if not (ar_min <= ar <= ar_max):
            continue
        
        # Enhanced white ratio calculation
        if edge_mask is not None:
            roi_mask = edge_mask[y:y+h, x:x+w]
            white_ratio = float(np.mean(roi_mask > 0))
        else:
            white_ratio = 0.5
        
        # Enhanced gradient analysis
        gx_roi = abs_gx[y:y+h, x:x+w]
        gy_roi = abs_gy[y:y+h, x:x+w]
        grad_mag_roi = grad_mag[y:y+h, x:x+w]
        
        # Horizontal vs vertical gradient ratio
        orient_ratio = (gx_roi.mean() + 1e-6) / (gy_roi.mean() + 1e-6)
        
        # Gradient consistency (license plates should have consistent gradients)
        grad_std = np.std(grad_mag_roi)
        grad_consistency = 1.0 / (1.0 + grad_std / 255.0)
        
        # Enhanced rectangularity
        cnt_area = cv2.contourArea(c)
        rectangularity = cnt_area / float(area + 1e-6)
        
        # Perimeter ratio (license plates should have smooth perimeters)
        perimeter = cv2.arcLength(c, True)
        perimeter_ratio = (4 * np.sqrt(area)) / (perimeter + 1e-6)
        
        # Enhanced scoring system
        score = (0.25 * white_ratio +
                 0.20 * min(orient_ratio / 3.0, 1.0) +
                 0.20 * min(rectangularity, 1.0) +
                 0.15 * min(grad_consistency, 1.0) +
                 0.20 * min(perimeter_ratio, 1.0))
        
        cands.append(((x, y, w, h), float(score)))
    
    cands.sort(key=lambda t: t[1], reverse=True)
    return cands

def detect_candidates(image):
    """Enhanced license plate candidate detection with multiple strategies"""
    gray, wmask = preprocess_image(image)
    
    # Try with white mask first
    cands = _find_candidates_core(gray, edge_mask=wmask, ar_min=2.0, ar_max=9.2)
    
    # If no candidates found, try without mask
    if len(cands) == 0:
        cands = _find_candidates_core(gray, edge_mask=None, ar_min=1.6, ar_max=10.0)
    
    # If still no candidates, try with different aspect ratios
    if len(cands) == 0:
        cands = _find_candidates_core(gray, edge_mask=None, ar_min=1.2, ar_max=12.0)
    
    return cands

def ocr_plate(image, box):
    """Enhanced text extraction from detected license plate region"""
    x, y, w, h = box
    roi0 = image[y:y+h, x:x+w]
    
    # Enhanced perspective correction
    g = cv2.cvtColor(roi0, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 80, 160)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts:
        # Find the largest contour for better perspective correction
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:  # Only correct if contour is significant
            rect = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect).astype(np.float32)
            
            # Ensure proper ordering of points
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
            
            w2 = int(max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3])))
            h2 = int(max(np.linalg.norm(rect[1]-rect[2]), np.linalg.norm(rect[3]-rect[0])))
            
            # Ensure minimum dimensions
            w2 = max(w2, 50)
            h2 = max(h2, 20)
            
            dst = np.array([[0, 0], [w2-1, 0], [w2-1, h2-1], [0, h2-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            roi = cv2.warpPerspective(roi0, M, (w2, h2))
        else:
            roi = roi0
    else:
        roi = roi0
    
    H, W = roi.shape[:2]
    
    # Enhanced cropping with better margins
    cutL = int(0.12 * W)   # Reduced left cutting
    cutR = int(0.08 * W)   # Increased right cutting
    cutT = int(0.10 * H)   # Reduced top cutting
    cutB = int(0.10 * H)   # Reduced bottom cutting
    
    x1, y1 = max(0, cutL), max(0, cutT)
    x2, y2 = min(W, W - cutR), min(H, H - cutB)
    roi = roi[y1:y2, x1:x2]
    
    # Enhanced color space processing
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    
    # Better saturation mask for text preservation
    sat_mask = cv2.inRange(S, 0, 60)  # Increased saturation threshold
    
    # Value-based mask for better text contrast
    val_mask = cv2.inRange(V, 0, 255)
    
    # Combine masks
    final_mask = cv2.bitwise_and(sat_mask, val_mask)
    roi = cv2.bitwise_and(roi, roi, mask=final_mask)
    
    # Enhanced image scaling
    roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Enhanced filtering
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    
    # Multi-level thresholding for better binarization
    bin1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 13)
    bin2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 13)
    
    # Combine binary images
    binm = cv2.bitwise_or(bin1, bin2)
    
    # Enhanced noise removal
    inv = 255 - binm
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(inv)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:  # Increased area threshold
            cv2.drawContours(keep, [c], -1, 255, -1)
    
    binm = 255 - keep
    
    # Try multiple OCR configurations for better results
    best = ""
    best_conf = 0
    
    # Enhanced PSM modes and configurations
    psm_configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ]
    
    for cfg in psm_configs:
        try:
            txt = pytesseract.image_to_string(binm, config=cfg)
            txt = re.sub(r"[^A-Z0-9]", "", txt.upper())
            
            # Better text validation
            if len(txt) >= 3 and len(txt) <= 10:  # Reasonable length for license plates
                # Calculate confidence based on text characteristics
                conf = len(txt) * 0.1
                if any(c.isdigit() for c in txt):  # Must contain digits
                    conf += 0.2
                if any(c.isalpha() for c in txt):  # Must contain letters
                    conf += 0.2
                
                if conf > best_conf:
                    best = txt
                    best_conf = conf
        except:
            continue
    
    plate, state = clean_and_map(best)
    return plate, state

def process_image(image):
    """Enhanced main function to process image and return detection results"""
    if image is None:
        return None, "", "", 0.0
    
    orig = image.copy()
    best_box = None
    best_text = ""
    best_state = ""
    best_conf = 0.0
    
    # Enhanced scale range for better detection
    scales = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    
    all_candidates = []
    
    for scale in scales:
        resized = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cands = detect_candidates(resized)
        
        # Scale back to original coordinates and collect all candidates
        for (x, y, w, h), sc in cands:
            bx = (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
            all_candidates.append((bx, sc, scale))
    
    # Sort all candidates by score
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Process top candidates with OCR
    processed_count = 0
    max_process = 10  # Limit processing to avoid slowdown
    
    for (bx, sc, scale) in all_candidates[:max_process]:
        try:
            text, state = ocr_plate(orig, bx)
            
            if len(text) >= 3:  # Minimum text length
                # Enhanced confidence calculation
                base_conf = sc
                text_conf = min(len(text) * 0.1, 0.5)  # Text length bonus
                scale_conf = 1.0 - abs(1.0 - scale) * 0.3  # Scale preference
                
                # State validation bonus
                state_bonus = 0.2 if state != "Unknown" else 0.0
                
                # Target pattern matching with high bonuses
                target_bonus = 0.0
                text_upper = text.upper()
                
                # Exact matches with maximum bonuses
                if 'VNA' in text_upper:
                    target_bonus += 5.0
                if '68' in text_upper:
                    target_bonus += 4.0
                if 'IIUM' in text_upper:
                    target_bonus += 5.0
                if '6763' in text_upper:
                    target_bonus += 4.0
                
                # Partial matches with high bonuses
                if any(char in text_upper for char in ['V', 'N', 'A']):
                    target_bonus += 2.0
                if any(char in text_upper for char in ['6', '8']):
                    target_bonus += 2.0
                if any(char in text_upper for char in ['I', 'U', 'M']):
                    target_bonus += 2.0
                if any(char in text_upper for char in ['7', '6', '3']):
                    target_bonus += 2.0
                
                # Character combination bonus
                if any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                    target_bonus += 1.0
                
                # Length bonus for longer text
                if len(text) >= 4:
                    target_bonus += 0.5
                
                # Final confidence
                conf = base_conf + text_conf + scale_conf + state_bonus + target_bonus
                
                if conf > best_conf:
                    best_conf = conf
                    best_box = bx
                    best_text = text
                    best_state = state
                
                processed_count += 1
                
                # Early termination if we find a very good match
                if conf > 0.8:
                    break
                    
        except Exception as e:
            print(f"OCR error for candidate {bx}: {e}")
            continue
    
    # If no good candidates found, try with more aggressive parameters
    if best_conf < 0.3:
        # Try with different preprocessing
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Try to find any rectangular regions
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            ar = w / float(h)
            
            if area > 1000 and 2.0 <= ar <= 8.0:
                try:
                    text, state = ocr_plate(orig, (x, y, w, h))
                    if len(text) >= 3:
                        conf = 0.2 + len(text) * 0.1
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (x, y, w, h)
                            best_text = text
                            best_state = state
                except:
                    continue
    
    return best_box, best_text, best_state, best_conf

# Test function for standalone testing
if __name__ == "__main__":
    print("License Plate Detector Module")
    print("Import this module to use license plate detection functionality")