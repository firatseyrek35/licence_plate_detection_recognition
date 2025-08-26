#!/usr/bin/env python3
"""
Simple OCR test script for license plate recognition
==================================================

This script tests only the OCR functionality without YOLO detection.
"""

import cv2
import easyocr
import numpy as np
from pathlib import Path

def simple_preprocess(plate_img):
    """Simple preprocessing that works well for license plates"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Simple noise reduction
    gray = cv2.medianBlur(gray, 3)
    
    # Basic contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Simple thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def test_ocr_on_image(image_path):
    """Test OCR on a single image"""
    print(f"🔍 Testing OCR on: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📏 Image size: {img.shape}")
    
    # Initialize OCR reader
    print("🤖 Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])
    
    # Test different regions (simulate YOLO detection)
    height, width = img.shape[:2]
    
    # Test center region
    center_x1 = width // 4
    center_y1 = height // 4
    center_x2 = 3 * width // 4
    center_y2 = 3 * height // 4
    
    print(f"🎯 Testing center region: ({center_x1}, {center_y1}) to ({center_x2}, {center_y2})")
    
    # Extract region
    plate_img = img[center_y1:center_y2, center_x1:center_x2]
    
    if plate_img.size == 0:
        print("❌ Could not extract region")
        return
    
    # Save extracted region
    cv2.imwrite("test_region.jpg", plate_img)
    print("💾 Saved extracted region as 'test_region.jpg'")
    
    # Test original image
    print("\n📖 Testing OCR on original region...")
    results_original = reader.readtext(plate_img)
    print(f"Original results: {results_original}")
    
    # Test preprocessed image
    print("\n🔧 Testing OCR on preprocessed region...")
    processed_img = simple_preprocess(plate_img)
    cv2.imwrite("test_processed.jpg", processed_img)
    print("💾 Saved processed region as 'test_processed.jpg'")
    
    results_processed = reader.readtext(processed_img)
    print(f"Processed results: {results_processed}")
    
    # Compare results
    print("\n📊 OCR Results Comparison:")
    print("=" * 50)
    
    if results_original:
        best_original = max(results_original, key=lambda x: x[2])
        print(f"Original: '{best_original[1]}' (Confidence: {best_original[2]:.3f})")
    else:
        print("Original: No text detected")
    
    if results_processed:
        best_processed = max(results_processed, key=lambda x: x[2])
        print(f"Processed: '{best_processed[1]}' (Confidence: {best_processed[2]:.3f})")
    else:
        print("Processed: No text detected")
    
    # Test different preprocessing methods
    print("\n🧪 Testing different preprocessing methods...")
    
    # Method 1: Grayscale only
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results_gray = reader.readtext(gray)
    print(f"Grayscale: {results_gray}")
    
    # Method 2: Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    results_blurred = reader.readtext(blurred)
    print(f"Gaussian blur: {results_blurred}")
    
    # Method 3: Histogram equalization
    equalized = cv2.equalizeHist(gray)
    results_equalized = reader.readtext(equalized)
    print(f"Histogram equalization: {results_equalized}")
    
    print("\n✅ OCR test completed!")
    print("📁 Check the generated images:")
    print("   - test_region.jpg (extracted region)")
    print("   - test_processed.jpg (preprocessed region)")

def main():
    """Main function"""
    print("🚗 License Plate OCR Test")
    print("=" * 40)
    
    # Test image path
    test_image = "test.webp"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please make sure 'test.webp' exists in the current directory.")
        return
    
    try:
        test_ocr_on_image(test_image)
    except Exception as e:
        print(f"❌ Error during OCR test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
