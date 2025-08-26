#!/usr/bin/env python3
"""
Configuration file for License Plate Detection & Recognition System
=================================================================

This file contains all configurable parameters for the system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_path: str = "best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45  # IOU threshold for NMS
    max_detections: int = 50
    device: str = "auto"  # auto, cpu, cuda, mps
    half_precision: bool = True  # Use FP16 for faster inference

@dataclass
class OCRConfig:
    """OCR configuration parameters"""
    languages: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    model_storage_directory: str = "~/.cache/easyocr"
    download_enabled: bool = True
    recog_network: str = "standard"  # standard, chinese_sim, chinese_tra, japanese, korean, thai, arabic, devanagari, cyrillic, greek, latin
    detector_network: str = "craft"  # craft, dbnet18, dbnet50, dbnet50dcn, psenet, psenetv2, psenetv3, fcenet

@dataclass
class ImageProcessingConfig:
    """Image processing configuration parameters"""
    resize_factor: float = 1.0
    padding: int = 5
    save_cropped_plates: bool = True
    output_format: str = "jpg"
    quality: int = 95
    preprocessing_enabled: bool = True
    
    # Preprocessing parameters
    bilateral_filter_d: int = 9
    bilateral_filter_sigma_color: int = 75
    bilateral_filter_sigma_space: int = 75
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple = (8, 8)
    sharpening_strength: int = 9
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2

@dataclass
class VideoProcessingConfig:
    """Video processing configuration parameters"""
    frame_skip: int = 20  # Process every Nth frame
    output_codec: str = "mp4v"
    output_fps: int = 30
    resize_video: bool = False
    target_width: int = 1920
    target_height: int = 1080

@dataclass
class OutputConfig:
    """Output configuration parameters"""
    base_output_dir: str = "runs/plate_detections"
    save_annotated_images: bool = True
    save_cropped_plates: bool = True
    save_processing_report: bool = True
    save_performance_metrics: bool = True
    create_timestamped_folders: bool = True
    
    # Report formats
    report_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    
    # Output file naming
    use_timestamps: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"

@dataclass
class LoggingConfig:
    """Logging configuration parameters"""
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_to_console: bool = True
    log_directory: str = "logs"
    max_log_files: int = 10
    log_file_size_mb: int = 10
    
    # Log format
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration"""
    enable_progress_bars: bool = True
    enable_multiprocessing: bool = False
    max_workers: int = 4
    batch_size: int = 1
    memory_optimization: bool = True
    cache_models: bool = True
    
    # GPU optimization
    gpu_memory_fraction: float = 0.8
    allow_growth: bool = True

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Paths
    project_root: Path = Path(__file__).parent
    models_dir: Path = field(default_factory=lambda: Path("models"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug_mode: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security
    allow_unsafe_models: bool = False
    validate_model_checksums: bool = True

class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager"""
        self.config_file = config_file
        self.model_config = ModelConfig()
        self.ocr_config = OCRConfig()
        self.image_config = ImageProcessingConfig()
        self.video_config = VideoProcessingConfig()
        self.output_config = OutputConfig()
        self.logging_config = LoggingConfig()
        self.performance_config = PerformanceConfig()
        self.system_config = SystemConfig()
        
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        import json
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section, data in config_data.items():
                if hasattr(self, f"{section}_config"):
                    config_obj = getattr(self, f"{section}_config")
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                            
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        import json
        
        config_data = {
            'model': self.model_config.__dict__,
            'ocr': self.ocr_config.__dict__,
            'image': self.image_config.__dict__,
            'video': self.video_config.__dict__,
            'output': self.output_config.__dict__,
            'logging': self.logging_config.__dict__,
            'performance': self.performance_config.__dict__,
            'system': self.system_config.__dict__
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            'model': self.model_config.__dict__,
            'ocr': self.ocr_config.__dict__,
            'image': self.image_config.__dict__,
            'video': self.video_config.__dict__,
            'output': self.output_config.__dict__,
            'logging': self.logging_config.__dict__,
            'performance': self.performance_config.__dict__,
            'system': self.system_config.__dict__
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Model validation
        if not Path(self.model_config.model_path).exists():
            warnings.append(f"Model file not found: {self.model_config.model_path}")
        
        # Confidence threshold validation
        if not 0.0 <= self.model_config.confidence_threshold <= 1.0:
            warnings.append("Confidence threshold must be between 0.0 and 1.0")
        
        # IOU threshold validation
        if not 0.0 <= self.model_config.iou_threshold <= 1.0:
            warnings.append("IOU threshold must be between 0.0 and 1.0")
        
        # Output directory validation
        if not self.output_config.base_output_dir:
            warnings.append("Output directory cannot be empty")
        
        return warnings

# Default configuration instance
default_config = ConfigManager()

# Environment-specific configurations
def get_production_config() -> ConfigManager:
    """Get production-optimized configuration"""
    config = ConfigManager()
    config.logging_config.level = "WARNING"
    config.performance_config.enable_multiprocessing = True
    config.performance_config.max_workers = 8
    config.performance_config.batch_size = 4
    return config

def get_development_config() -> ConfigManager:
    """Get development configuration with debug features"""
    config = ConfigManager()
    config.logging_config.level = "DEBUG"
    config.logging_config.enabled = True
    config.performance_config.enable_progress_bars = True
    config.output_config.save_processing_report = True
    return config

def get_testing_config() -> ConfigManager:
    """Get testing configuration"""
    config = ConfigManager()
    config.logging_config.level = "DEBUG"
    config.output_config.base_output_dir = "test_outputs"
    config.performance_config.enable_progress_bars = False
    return config

if __name__ == "__main__":
    # Example usage
    config = ConfigManager()
    
    # Print current configuration
    print("Current Configuration:")
    print("=" * 50)
    for section, data in config.get_all_configs().items():
        print(f"\n{section.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Validate configuration
    warnings = config.validate_config()
    if warnings:
        print("\nConfiguration Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ Configuration is valid!")
    
    # Save configuration
    config.save_to_file("config_backup.json")
    print("\n💾 Configuration saved to config_backup.json")
