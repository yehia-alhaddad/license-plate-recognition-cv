#!/usr/bin/env python3
"""
License Plate Recognition and State Identification System (LPR-SIS)
Main entry point for the integrated system

This system integrates:
- license_plate_detector.py: License plate detection using computer vision
- ocr_processor.py: OCR and post-processing for text recognition
- lpr_gui.py: GUI interface for user interaction

Requirements:
- Python 3.7+
- OpenCV
- PIL/Pillow
- pytesseract
- tkinter (usually included with Python)
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = ['cv2', 'PIL', 'pytesseract', 'tkinter']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image, ImageTk
            elif package == 'pytesseract':
                import pytesseract
            elif package == 'tkinter':
                import tkinter
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages using:")
        print("pip install opencv-python pillow pytesseract")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is properly configured"""
    try:
        import pytesseract
        # Set Tesseract path directly for Windows
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"Tesseract OCR not properly configured: {e}")
        print("Please ensure Tesseract is installed and the path is correct.")
        return False

def main():
    """Main entry point for the LPR system"""
    print("License Plate Recognition and State Identification System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Check Tesseract
    if not check_tesseract():
        print("Tesseract check failed. Exiting.")
        sys.exit(1)
    
    print("All checks passed. Starting GUI...")
    
    # Import and start GUI
    try:
        from lpr_gui import LPR_GUI
        import tkinter as tk
        
        root = tk.Tk()
        app = LPR_GUI(root)
        
        print("GUI started successfully.")
        print("Use the interface to:")
        print("  1. Upload vehicle images")
        print("  2. Detect license plates")
        print("  3. View recognition results")
        print("  4. Export results to CSV")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()