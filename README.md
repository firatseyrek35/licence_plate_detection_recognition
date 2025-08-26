# 🚗 License Plate Detection & Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**License Plate Detection & Recognition System** is an advanced computer vision application that combines state-of-the-art YOLO object detection with OCR (Optical Character Recognition) technology to automatically detect and read license plates from images and video files.

This project demonstrates expertise in:
- **Deep Learning**: YOLO-based object detection
- **Computer Vision**: Image processing and analysis
- **OCR Technology**: Text extraction from images
- **Python Development**: Professional software engineering practices

## ✨ Features

### 🔍 **Core Capabilities**
- **Real-time License Plate Detection**: YOLO-based detection with configurable confidence thresholds
- **High-Accuracy OCR**: EasyOCR integration for robust text recognition
- **Multi-format Support**: Handles JPG, PNG, WEBP, BMP, TIFF, and MP4 files
- **Batch Processing**: Process entire directories of images simultaneously
- **Intelligent Preprocessing**: Advanced image enhancement for optimal OCR results

### 🎨 **Advanced Image Processing**
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
- **Noise Reduction**: Bilateral filtering for cleaner images
- **Sharpening**: Kernel-based image enhancement
- **Adaptive Thresholding**: Optimal text extraction
- **Morphological Operations**: Image cleanup and optimization

### 📊 **Comprehensive Output**
- **Visual Annotations**: Bounding boxes with confidence scores
- **OCR Results**: Extracted text with confidence metrics
- **Detailed Logging**: Console output with processing statistics
- **Organized Storage**: Structured output directory management
- **Performance Reports**: JSON-based processing summaries

### 🎥 **Video Processing**
- **Frame-based Detection**: Intelligent frame skipping for performance
- **Real-time Processing**: Live video analysis capabilities
- **Output Generation**: Processed videos with detection overlays

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Image  │───▶│  YOLO Detection  │───▶│  Plate Cropping │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Output Image  │◀───│  OCR Processing  │◀───│ Image Preproc.  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Technology Stack**
- **Detection Engine**: Ultralytics YOLO
- **OCR Engine**: EasyOCR
- **Image Processing**: OpenCV
- **Deep Learning**: PyTorch
- **Development**: Python 3.8+

## 🚀 Installation

### **Prerequisites**
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 4GB+ RAM

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Manual Installation**
```bash
# Core dependencies
pip install torch torchvision ultralytics opencv-python numpy

# OCR engine
pip install easyocr

# Additional utilities
pip install pandas scipy tqdm Pillow
```

## 💻 Usage

### **Basic Usage**

```bash
# Single image processing
python main.py --source test.webp

# Video file processing
python main.py --source video.mp4 --conf 0.5

# Directory batch processing
python main.py --source images_folder/ --save-cropped

# Custom parameters
python main.py --source data/ --conf 0.3 --iou 0.4 --max-det 100
```

### **Command Line Arguments**

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--source` | `-s` | string | **Required** | Input image/video file or directory |
| `--conf` | - | float | 0.25 | Detection confidence threshold |
| `--iou` | - | float | 0.45 | IOU threshold for NMS |
| `--max-det` | - | int | 50 | Maximum detections per image |
| `--out` | `-o` | string | `runs/plate_detections` | Output directory path |
| `--save-cropped` | - | flag | False | Save individual cropped plate images |
| `--no-logging` | - | flag | False | Disable logging |
| `--log-level` | - | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |

### **Example Output**

```
🔍 2 adet plaka tespit edildi!
==================================================
📋 Plaka 1: 34ABC123
   Güvenilirlik: 0.85
   Konum: (150, 200) - (300, 250)
------------------------------
📋 Plaka 2: 06XYZ789
   Güvenilirlik: 0.92
   Konum: (400, 150) - (550, 200)
------------------------------

📊 ÖZET:
   Toplam plaka: 2
   Plaka 1: 34ABC123 (Güven: 0.85)
   Plaka 2: 06XYZ789 (Güven: 0.92)
==================================================
```

## 🔧 API Reference

### **Core Classes**

#### `LicensePlateDetector`
Main detection and recognition class.

**Methods:**
- `process_image(image_path, output_dir)`: Process single image
- `process_video(video_path, output_dir)`: Process video file
- `generate_report(results, output_dir)`: Generate processing report

#### `Config`
Configuration management class.

**Parameters:**
- `confidence_threshold`: Detection confidence (0.0-1.0)
- `iou_threshold`: IOU threshold for NMS (0.0-1.0)
- `max_detections`: Maximum detections per image
- `save_cropped_plates`: Save individual plate images

### **Utility Classes**

#### `ImageUtils`
Image processing utilities.

#### `TextUtils`
Text processing and validation.

#### `FileUtils`
File and directory operations.

## 📈 Performance

### **Detection Accuracy**
- **YOLO Model**: Custom-trained on license plate dataset
- **Confidence Range**: Configurable threshold (default: 0.25)
- **Processing Speed**: Real-time performance on GPU

### **OCR Performance**
- **Text Recognition**: High accuracy on clear images
- **Character Support**: Latin alphabet and numbers
- **Confidence Scoring**: Reliability metrics for each detection

### **System Requirements**
- **Minimum**: CPU-only processing
- **Recommended**: NVIDIA GPU with CUDA support
- **Memory**: 4GB+ RAM for large image processing

## 🧪 Testing

### **Run System Tests**
```bash
# Test all components
python test_system.py

# Test specific functionality
python -c "from main import Config; print('Config test passed')"
```

### **Test Coverage**
The system includes comprehensive testing for:
- Module imports
- Configuration management
- Model loading
- Utility functions
- Image processing capabilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Fork and clone
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection

# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_system.py

# Code formatting
black main.py utils.py config.py
flake8 main.py utils.py config.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO implementation
- **EasyOCR**: OCR engine
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/license-plate-detection](https://github.com/yourusername/license-plate-detection)
- **Issues**: [GitHub Issues](https://github.com/yourusername/license-plate-detection/issues)
- **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ for Computer Vision & AI**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/license-plate-detection?style=social)](https://github.com/yourusername/license-plate-detection)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/license-plate-detection?style=social)](https://github.com/yourusername/license-plate-detection)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/license-plate-detection)](https://github.com/yourusername/license-plate-detection/issues)

</div>
