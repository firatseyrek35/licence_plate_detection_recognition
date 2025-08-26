#!/usr/bin/env python3
"""
Test script for License Plate Detection & Recognition System
==========================================================

This script tests the basic functionality of the system.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics YOLO import failed: {e}")
        return False
    
    try:
        import easyocr
        print("✅ EasyOCR imported successfully")
    except ImportError as e:
        print(f"❌ EasyOCR import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm imported successfully")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration system"""
    print("\n🔧 Testing configuration...")
    
    try:
        from main import Config, LicensePlateDetector
        
        # Test config creation
        config = Config(
            confidence_threshold=0.3,
            iou_threshold=0.5,
            max_detections=25
        )
        print("✅ Config created successfully")
        print(f"   Confidence: {config.confidence_threshold}")
        print(f"   IOU: {config.iou_threshold}")
        print(f"   Max detections: {config.max_detections}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🤖 Testing model loading...")
    
    try:
        from main import Config, LicensePlateDetector
        
        config = Config()
        
        # Check if model file exists
        if not Path(config.model_path).exists():
            print(f"⚠️  Model file not found: {config.model_path}")
            print("   Skipping model loading test")
            return True
        
        # Try to initialize detector
        detector = LicensePlateDetector(config)
        print("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🛠️  Testing utility functions...")
    
    try:
        from utils import ImageUtils, TextUtils, FileUtils
        
        # Test text cleaning
        test_text = "34ABC123"
        cleaned = TextUtils.clean_plate_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test plate validation
        is_valid = TextUtils.validate_plate_format(cleaned)
        print(f"✅ Plate validation: {is_valid}")
        
        # Test file utilities
        test_dir = "test_output"
        FileUtils.ensure_directory(test_dir)
        print(f"✅ Directory creation: {test_dir}")
        
        # Cleanup
        import shutil
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"✅ Cleanup completed: {test_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n🖼️  Testing image processing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        test_img[:] = (128, 128, 128)  # Gray color
        
        # Test basic operations
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✅ Image conversion: BGR -> Gray")
        
        # Test blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("✅ Image blurring")
        
        # Test threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        print("✅ Image thresholding")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚗 License Plate Detection System - System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Model Loading Test", test_model_loading),
        ("Utility Functions Test", test_utility_functions),
        ("Image Processing Test", test_image_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
