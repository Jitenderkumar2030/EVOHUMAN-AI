"""
Shared utilities for EvoHuman.AI platform
"""
import hashlib
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
import structlog


# Logging setup
def setup_logging(service_name: str) -> structlog.BoundLogger:
    """Setup structured logging for a service"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger(service_name)
    return logger


# ID Generation
def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """Generate a short unique ID"""
    return str(uuid.uuid4())[:8]


# Encryption utilities
class DataEncryption:
    """Handle data encryption for privacy"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self.key = key
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt string data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data back to string"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> bytes:
        """Encrypt dictionary data"""
        json_str = json.dumps(data, default=str)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt data back to dictionary"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


# Hash utilities
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed


def hash_data(data: str) -> str:
    """Create a hash of arbitrary data"""
    return hashlib.sha256(data.encode()).hexdigest()


# Time utilities
def utc_now() -> datetime:
    """Get current UTC datetime"""
    return datetime.utcnow()


def days_from_now(days: int) -> datetime:
    """Get datetime N days from now"""
    return utc_now() + timedelta(days=days)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# Data validation
def validate_bio_metric(value: float, metric_type: str) -> bool:
    """Validate biological metric values"""
    ranges = {
        "heart_rate": (30, 220),
        "blood_pressure_systolic": (70, 200),
        "blood_pressure_diastolic": (40, 120),
        "body_temperature": (35.0, 42.0),
        "glucose": (70, 400),
        "cholesterol": (100, 400),
        "bmi": (10, 50),
        "body_fat_percentage": (3, 50),
    }
    
    if metric_type in ranges:
        min_val, max_val = ranges[metric_type]
        return min_val <= value <= max_val
    
    return True  # Unknown metrics pass validation


# Configuration helpers
def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    import yaml
    
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    result = {}
    for config in configs:
        result.update(config)
    return result


# Health check utilities
def create_health_check_response(service_name: str, dependencies: Dict[str, bool]) -> Dict[str, Any]:
    """Create standardized health check response"""
    all_healthy = all(dependencies.values())
    
    return {
        "service": service_name,
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": utc_now().isoformat(),
        "dependencies": dependencies,
        "version": "1.0.0"  # TODO: Get from package
    }
