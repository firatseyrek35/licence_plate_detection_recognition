#!/usr/bin/env python3
"""
Utility functions for License Plate Detection & Recognition System
================================================================

This module contains helper functions, data validation, and common operations.
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class ImageUtils:
    """Utility class for image processing operations"""
    
    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> bool:
        """Validate if image path exists and is readable"""
        try:
            path = Path(image_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive image information"""
        try:
            path = Path(image_path)
            if not path.exists():
                return {"error": "File does not exist"}
            
            # Basic file info
            stat = path.stat()
            file_info = {
                "path": str(path),
                "name": path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            # Image-specific info
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    file_info.update({
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channels": img.shape[2] if len(img.shape) > 2 else 1,
                        "dtype": str(img.dtype),
                        "aspect_ratio": round(img.shape[1] / img.shape[0], 2)
                    })
                else:
                    file_info["error"] = "Could not read image"
            except Exception as e:
                file_info["error"] = f"Image reading error: {str(e)}"
            
            return file_info
            
        except Exception as e:
            return {"error": f"Error getting image info: {str(e)}"}
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    preserve_aspect: bool = True) -> np.ndarray:
        """Resize image with optional aspect ratio preservation"""
        if preserve_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray, 
                            brightness: float = 1.0,
                            contrast: float = 1.0,
                            saturation: float = 1.0) -> np.ndarray:
        """Enhance image quality with brightness, contrast, and saturation adjustments"""
        try:
            # Convert to float for calculations
            img_float = image.astype(np.float32) / 255.0
            
            # Apply brightness and contrast
            img_enhanced = img_float * contrast + brightness - 0.5
            
            # Apply saturation (convert to HSV)
            if len(img_enhanced.shape) == 3:
                hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation
                img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Clip values and convert back to uint8
            img_enhanced = np.clip(img_enhanced, 0, 1)
            return (img_enhanced * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    @staticmethod
    def create_image_montage(images: List[np.ndarray], 
                            titles: List[str] = None,
                            max_width: int = 1200,
                            max_height: int = 800) -> np.ndarray:
        """Create a montage of multiple images"""
        if not images:
            return np.array([])
        
        # Calculate grid dimensions
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        
        # Find maximum dimensions
        max_img_w = max(img.shape[1] for img in images)
        max_img_h = max(img.shape[0] for img in images)
        
        # Calculate cell size
        cell_w = min(max_img_w, max_width // cols)
        cell_h = min(max_img_h, max_height // rows)
        
        # Create montage canvas
        montage_h = cell_h * rows
        montage_w = cell_w * cols
        montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            # Resize image to fit cell
            resized = cv2.resize(img, (cell_w, cell_h))
            
            # Calculate position
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w
            
            # Place image
            montage[y_start:y_end, x_start:x_end] = resized
            
            # Add title if provided
            if titles and idx < len(titles):
                cv2.putText(montage, titles[idx], (x_start + 5, y_start + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return montage

class TextUtils:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_plate_text(text: str) -> str:
        """Clean and normalize license plate text"""
        if not text:
            return ""
        
        # Remove special characters except alphanumeric
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        # Common OCR corrections
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8',
            'Z': '2', 'A': '4', 'E': '3'
        }
        
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        
        return cleaned
    
    @staticmethod
    def validate_plate_format(text: str, country: str = "TR") -> bool:
        """Validate license plate format for specific country"""
        if not text:
            return False
        
        # Turkish license plate format: 34ABC123
        if country == "TR":
            pattern = r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$'
            return bool(re.match(pattern, text))
        
        # Generic format: alphanumeric, 5-10 characters
        elif country == "GENERIC":
            pattern = r'^[A-Z0-9]{5,10}$'
            return bool(re.match(pattern, text))
        
        return False
    
    @staticmethod
    def extract_plate_components(text: str) -> Dict[str, str]:
        """Extract components from license plate text"""
        if not text:
            return {}
        
        # Turkish format: 34ABC123
        tr_pattern = r'^([0-9]{2})([A-Z]{1,3})([0-9]{2,4})$'
        match = re.match(tr_pattern, text)
        
        if match:
            return {
                "province_code": match.group(1),
                "letters": match.group(2),
                "numbers": match.group(3),
                "format": "TR"
            }
        
        return {"raw_text": text, "format": "UNKNOWN"}

class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_unique_filename(base_path: Union[str, Path], 
                          suffix: str = "") -> Path:
        """Generate unique filename to avoid conflicts"""
        base_path = Path(base_path)
        counter = 1
        
        while base_path.exists():
            stem = base_path.stem
            if suffix:
                new_name = f"{stem}_{counter}{suffix}"
            else:
                new_name = f"{stem}_{counter}{base_path.suffix}"
            base_path = base_path.parent / new_name
            counter += 1
        
        return base_path
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], 
                          algorithm: str = "md5") -> str:
        """Calculate file hash for integrity checking"""
        try:
            hash_func = getattr(hashlib, algorithm)()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """Get file size in megabytes"""
        try:
            size_bytes = Path(file_path).stat().st_size
            return round(size_bytes / (1024 * 1024), 2)
        except Exception:
            return 0.0

class ValidationUtils:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_coordinates(box: List[float]) -> bool:
        """Validate bounding box coordinates"""
        if len(box) != 4:
            return False
        
        x1, y1, x2, y2 = box
        
        # Check if coordinates are numeric
        if not all(isinstance(coord, (int, float)) for coord in box):
            return False
        
        # Check if coordinates are valid
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Check if coordinates are non-negative
        if any(coord < 0 for coord in box):
            return False
        
        return True
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """Validate confidence score"""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def validate_image_array(img: np.ndarray) -> bool:
        """Validate image array"""
        if not isinstance(img, np.ndarray):
            return False
        
        if img.size == 0:
            return False
        
        if len(img.shape) < 2 or len(img.shape) > 3:
            return False
        
        return True

class PerformanceUtils:
    """Utility class for performance monitoring"""
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    @staticmethod
    def calculate_fps(processing_time: float, frame_count: int) -> float:
        """Calculate frames per second"""
        if processing_time <= 0 or frame_count <= 0:
            return 0.0
        return frame_count / processing_time

class VisualizationUtils:
    """Utility class for visualization and drawing"""
    
    @staticmethod
    def draw_bounding_box(image: np.ndarray, box: List[float], 
                         label: str = "", color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """Draw bounding box on image"""
        if not ValidationUtils.validate_coordinates(box):
            return image
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            # Calculate label size
            font_scale = 0.6
            font_thickness = 1
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(image, 
                         (x1, y1 - label_height - baseline - 5),
                         (x1 + label_width, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                       font_thickness)
        
        return image
    
    @staticmethod
    def create_processing_visualization(results: List[Dict], 
                                      output_path: Union[str, Path]) -> bool:
        """Create visualization of processing results"""
        try:
            # Group results by type
            images = [r for r in results if 'image_path' in r]
            videos = [r for r in results if 'video_path' in r]
            
            # Create summary visualization
            summary_data = {
                "Total Files Processed": len(results),
                "Images Processed": len(images),
                "Videos Processed": len(videos),
                "Total Plates Detected": sum(r.get('plates_detected', 0) for r in results),
                "Average Processing Time": np.mean([r.get('processing_time', 0) for r in results])
            }
            
            # Create simple text visualization
            vis_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            y_offset = 50
            for key, value in summary_data.items():
                text = f"{key}: {value}"
                cv2.putText(vis_img, text, (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y_offset += 40
            
            # Save visualization
            cv2.imwrite(str(output_path), vis_img)
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Test image utilities
    print("Testing ImageUtils...")
    test_path = "test.webp"
    if Path(test_path).exists():
        info = ImageUtils.get_image_info(test_path)
        print(f"Image info: {info}")
    
    # Test text utilities
    print("\nTesting TextUtils...")
    test_texts = ["34ABC123", "06XYZ789", "Invalid!@#"]
    for text in test_texts:
        cleaned = TextUtils.clean_plate_text(text)
        valid = TextUtils.validate_plate_format(cleaned)
        components = TextUtils.extract_plate_components(cleaned)
        print(f"'{text}' -> '{cleaned}' (valid: {valid}) -> {components}")
    
    # Test file utilities
    print("\nTesting FileUtils...")
    FileUtils.ensure_directory("test_output")
    unique_path = FileUtils.get_unique_filename("test_output/test.txt")
    print(f"Unique path: {unique_path}")
    
    print("\n✅ All utility tests completed!")
