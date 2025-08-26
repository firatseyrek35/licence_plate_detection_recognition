#!/usr/bin/env python3
"""
License Plate Detection & Recognition System
============================================

Advanced computer vision application combining YOLO object detection with OCR technology
for automatic license plate recognition from images and video files.

Author: Your Name
Version: 1.0.0
License: MIT
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import easyocr
import numpy as np
from tqdm import tqdm

# PyTorch 2.6+ güvenlik değişikliği için (Ultralytics .pt yüklenebilsin)
torch.serialization.add_safe_globals([DetectionModel])

# Global configuration
@dataclass
class Config:
    """Application configuration class"""
    model_path: str = "best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45  # IOU threshold for NMS
    max_detections: int = 50
    ocr_languages: List[str] = None
    output_format: str = "jpg"
    save_cropped_plates: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    debug_mode: bool = False  # Enable debug features
    
    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ['en']

# Performance metrics
@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    total_images: int = 0
    total_plates_detected: int = 0
    total_processing_time: float = 0.0
    average_detection_time: float = 0.0
    average_ocr_time: float = 0.0
    successful_ocr_count: int = 0
    failed_ocr_count: int = 0
    
    def update(self, detection_time: float, ocr_time: float, plates_detected: int, ocr_success: int):
        """Update metrics with new data"""
        self.total_images += 1
        self.total_plates_detected += plates_detected
        self.total_processing_time += detection_time + ocr_time
        self.average_detection_time = (self.average_detection_time * (self.total_images - 1) + detection_time) / self.total_images
        self.average_ocr_time = (self.average_ocr_time * (self.total_images - 1) + ocr_time) / self.total_images
        self.successful_ocr_count += ocr_success
        self.failed_ocr_count += (plates_detected - ocr_success)

class LicensePlateDetector:
    """Main license plate detection and recognition class"""
    
    def __init__(self, config: Config):
        """Initialize the detector with configuration"""
        self.config = config
        self.metrics = PerformanceMetrics()
        self.setup_logging()
        self.setup_models()
        
    def setup_logging(self):
        """Setup logging configuration"""
        if not self.config.enable_logging:
            return
            
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"plate_detection_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("License Plate Detection System initialized")
        
    def setup_models(self):
        """Initialize YOLO and OCR models"""
        try:
            self.logger.info("Loading YOLO model...")
            self.yolo_model = YOLO(self.config.model_path)
            self.logger.info(f"YOLO model loaded: {self.config.model_path}")
            
            self.logger.info("Initializing OCR reader...")
            self.ocr_reader = easyocr.Reader(self.config.ocr_languages)
            self.logger.info("OCR reader initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """Simple but effective image preprocessing for OCR"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Simple noise reduction with median blur
            gray = cv2.medianBlur(gray, 3)
            
            # Basic contrast enhancement with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Simple Otsu thresholding (more reliable than adaptive)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {str(e)}")
            return plate_img
    
    def extract_and_read_plate(self, img: np.ndarray, box: List[float], 
                              reader: easyocr.Reader) -> Tuple[str, float]:
        """Extract license plate region and perform OCR recognition"""
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding around the detected region
            padding = 10  # Increased padding for better plate extraction
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)
            
            # Extract plate region
            plate_img = img[y1:y2, x1:x2]
            
            if plate_img.size == 0:
                return "Plaka kesilemedi", 0.0
            
            # Save cropped plate if enabled
            if self.config.save_cropped_plates:
                self.save_cropped_plate(plate_img, f"plate_{x1}_{y1}_{x2}_{y2}")
            
            # Debug: Save both original and processed images (only if debug enabled)
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(debug_dir / f"original_{timestamp}.jpg"), plate_img)
                cv2.imwrite(str(debug_dir / f"processed_{timestamp}.jpg"), processed_img)
            
            # Preprocess image
            processed_img = self.preprocess_plate_image(plate_img)
            
            # OCR recognition - try original image first (often better)
            self.logger.info("Attempting OCR on original image...")
            results = reader.readtext(plate_img)
            
            if not results:
                self.logger.info("No results from original image, trying processed...")
                results = reader.readtext(processed_img)
            
            if results:
                self.logger.info(f"OCR found {len(results)} results: {results}")
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                self.logger.info(f"Best OCR result: '{text}' with confidence {confidence:.3f}")
                
                # Clean up text (remove special characters, normalize)
                cleaned_text = self.clean_plate_text(text)
                self.logger.info(f"Cleaned text: '{cleaned_text}'")
                
                # Only return if we have meaningful text
                if len(cleaned_text) >= 3:  # At least 3 characters
                    return cleaned_text, confidence
                else:
                    self.logger.warning(f"Text too short: '{cleaned_text}' (length: {len(cleaned_text)})")
                    return "Metin çok kısa", 0.0
            else:
                self.logger.warning("No OCR results found from either image")
                return "Metin okunamadı", 0.0
                
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            return f"OCR hatası: {str(e)}", 0.0
    
    def clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text"""
        import re
        
        # Remove special characters except alphanumeric
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        return cleaned
    
    def save_cropped_plate(self, plate_img: np.ndarray, filename: str):
        """Save cropped plate image for analysis"""
        try:
            cropped_dir = Path("cropped_plates")
            cropped_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = cropped_dir / f"{filename}_{timestamp}.jpg"
            cv2.imwrite(str(save_path), plate_img)
            
        except Exception as e:
            self.logger.warning(f"Could not save cropped plate: {str(e)}")
    
    def process_image(self, image_path: Path, output_dir: Path) -> Dict:
        """Process single image and return results"""
        start_time = time.time()
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # YOLO detection
            detection_start = time.time()
            results = self.yolo_model.predict(
                source=str(image_path), 
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,  # IOU threshold for NMS
                max_det=self.config.max_detections,
                verbose=False
            )[0]
            detection_time = time.time() - detection_start
            
            # Extract detection results
            boxes_xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            cls_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
            confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

            # Get class names
            names_map = results.names if hasattr(results, "names") else self.yolo_model.model.names
            cls_names = [names_map[c] if c in names_map else str(c) for c in cls_ids]

            # OCR processing
            ocr_start = time.time()
            ocr_results = []
            successful_ocr = 0
            
            if len(boxes_xyxy) > 0:
                self.logger.info(f"Detected {len(boxes_xyxy)} plates in {image_path.name}")
                
                for i, box in enumerate(boxes_xyxy):
                    ocr_text, ocr_conf = self.extract_and_read_plate(img, box, self.ocr_reader)
                    ocr_results.append((ocr_text, ocr_conf))
                    
                    if ocr_conf > 0.5:  # Consider successful if confidence > 50%
                        successful_ocr += 1
                    
                    self.logger.info(f"Plate {i+1}: {ocr_text} (Confidence: {ocr_conf:.2f})")
            
            ocr_time = time.time() - ocr_start
            total_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update(detection_time, ocr_time, len(boxes_xyxy), successful_ocr)
            
            # Save annotated image
            output_path = self.save_annotated_image(
                image_path, img, boxes_xyxy, cls_names, confs, ocr_results, output_dir
            )
            
            return {
                'image_path': str(image_path),
                'plates_detected': len(boxes_xyxy),
                'ocr_results': ocr_results,
                'processing_time': total_time,
                'detection_time': detection_time,
                'ocr_time': ocr_time,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def save_annotated_image(self, image_path: Path, img: np.ndarray, 
                           boxes: np.ndarray, names: List[str], 
                           confidences: List[float], ocr_results: List[Tuple[str, float]], 
                           output_dir: Path) -> Path:
        """Save image with annotations and OCR results"""
        try:
            annotated_img = img.copy()
            
            for i, ((x1, y1, x2, y2), cls_id, conf) in enumerate(zip(boxes, names, confidences)):
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                
                # Draw bounding box
                cv2.rectangle(annotated_img, p1, p2, (0, 255, 0), 2)
                
                # Prepare label
                if i < len(ocr_results):
                    ocr_text, ocr_conf = ocr_results[i]
                    label = f"{cls_id} {conf:.2f} | {ocr_text} ({ocr_conf:.2f})"
                else:
                    label = f"{cls_id} {conf:.2f}"
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_img, 
                            (p1[0], p1[1] - label_size[1] - 10),
                            (p1[0] + label_size[0], p1[1]),
                            (0, 255, 0), -1)
                
                cv2.putText(annotated_img, label, 
                           (p1[0], p1[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save image
            output_path = output_dir / f"{image_path.stem}_annotated.{self.config.output_format}"
            cv2.imwrite(str(output_path), annotated_img)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving annotated image: {str(e)}")
            raise
    
    def process_video(self, video_path: Path, output_dir: Path) -> Dict:
        """Process video file and extract frames with plates"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            self.logger.info(f"Processing video: {video_path.name}")
            self.logger.info(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
            
            frame_count = 0
            plates_found = 0
            start_time = time.time()
            
            # Create output video writer
            output_video_path = output_dir / f"{video_path.stem}_processed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, 
                                (int(cap.get(3)), int(cap.get(4))))
            
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process every 3rd frame for performance
                    if frame_count % 3 == 0:
                        # Run detection on frame
                        results = self.yolo_model.predict(
                            source=frame, 
                            conf=self.config.confidence_threshold,
                            iou=self.config.iou_threshold,  # IOU threshold for NMS
                            verbose=False
                        )[0]
                        
                        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
                        
                        if len(boxes) > 0:
                            plates_found += len(boxes)
                            # Draw boxes on frame
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    out.write(frame)
                    pbar.update(1)
            
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            return {
                'video_path': str(video_path),
                'total_frames': total_frames,
                'plates_found': plates_found,
                'processing_time': processing_time,
                'output_path': str(output_video_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            return {'video_path': str(video_path), 'error': str(e)}
    
    def generate_report(self, results: List[Dict], output_dir: Path):
        """Generate comprehensive processing report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'model_path': self.config.model_path,
                    'confidence_threshold': self.config.confidence_threshold,
                    'iou_threshold': self.config.iou_threshold
                },
                'performance_metrics': {
                    'total_images_processed': self.metrics.total_images,
                    'total_plates_detected': self.metrics.total_plates_detected,
                    'total_processing_time': self.metrics.total_processing_time,
                    'average_detection_time': self.metrics.average_detection_time,
                    'average_ocr_time': self.metrics.average_ocr_time,
                    'ocr_success_rate': (self.metrics.successful_ocr_count / 
                                       max(self.metrics.total_plates_detected, 1)) * 100
                },
                'results': results
            }
            
            # Save JSON report
            report_path = output_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Print summary
            self.print_summary(report)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None
    
    def print_summary(self, report: Dict):
        """Print processing summary to console"""
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        
        metrics = report['performance_metrics']
        print(f"🖼️  Total Images Processed: {metrics['total_images_processed']}")
        print(f"🚗 Total Plates Detected: {metrics['total_plates_detected']}")
        print(f"⏱️  Total Processing Time: {metrics['total_processing_time']:.2f}s")
        print(f"🔍 Average Detection Time: {metrics['average_detection_time']:.3f}s")
        print(f"📝 Average OCR Time: {metrics['average_ocr_time']:.3f}s")
        print(f"✅ OCR Success Rate: {metrics['ocr_success_rate']:.1f}%")
        print("="*60)

def is_image_file(p: Path) -> bool:
    """Check if file is a supported image format"""
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_video_file(p: Path) -> bool:
    """Check if file is a supported video format"""
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced License Plate Detection & Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source image.jpg
  python main.py --source video.mp4 --conf 0.5
  python main.py --source images_folder/ --save-cropped
  python main.py --source data/ --conf 0.3 --iou 0.4 --max-det 100
  python main.py --source test.webp --debug  # Enable debug mode
        """
    )
    
    parser.add_argument("--source", "-s", type=str, required=True,
                       help="Input image, video file, or directory path")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IOU threshold for NMS (default: 0.45)")
    parser.add_argument("--max-det", type=int, default=50,
                       help="Maximum detections per image (default: 50)")
    parser.add_argument("--out", "-o", type=str, default="runs/plate_detections",
                       help="Output directory path")
    parser.add_argument("--save-cropped", action="store_true",
                       help="Save individual cropped plate images")
    parser.add_argument("--no-logging", action="store_true",
                       help="Disable logging")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level (default: INFO)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (save debug images)")
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config(
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            max_detections=args.max_det,
            save_cropped_plates=args.save_cropped,
            enable_logging=not args.no_logging,
            log_level=args.log_level,
            debug_mode=args.debug
        )
        
        # Initialize detector
        detector = LicensePlateDetector(config)
        
        # Setup output directory
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process source
        source_path = Path(args.source)
        results = []
        
        if source_path.is_file():
            if is_image_file(source_path):
                detector.logger.info(f"Processing single image: {source_path}")
                result = detector.process_image(source_path, output_dir)
                results.append(result)
                
            elif is_video_file(source_path):
                detector.logger.info(f"Processing video file: {source_path}")
                result = detector.process_video(source_path, output_dir)
                results.append(result)
                
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")
                
        elif source_path.is_dir():
            detector.logger.info(f"Processing directory: {source_path}")
            
            # Get all supported files
            image_files = [p for p in source_path.rglob("*") if is_image_file(p)]
            video_files = [p for p in source_path.rglob("*") if is_video_file(p)]
            
            detector.logger.info(f"Found {len(image_files)} images and {len(video_files)} videos")
            
            # Process images
            for img_path in tqdm(image_files, desc="Processing images"):
                result = detector.process_image(img_path, output_dir)
                results.append(result)
            
            # Process videos
            for video_path in tqdm(video_files, desc="Processing videos"):
                result = detector.process_video(video_path, output_dir)
                results.append(result)
                
        else:
            raise ValueError(f"Source path does not exist: {source_path}")
        
        # Generate report
        if results:
            report_path = detector.generate_report(results, output_dir)
            if report_path:
                detector.logger.info(f"Report saved to: {report_path}")
        
        detector.logger.info("Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
