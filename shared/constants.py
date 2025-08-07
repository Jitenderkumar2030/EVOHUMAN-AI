"""
Shared constants for EvoHuman.AI platform
"""

# Model Versions
MODEL_VERSIONS = {
    "esm3": "1.2.0",
    "bio_twin": "2.1.0", 
    "symbiotic": "1.5.0",
    "proteus": "1.0.0",
    "aice": "1.1.0"
}

# Service Names
SERVICE_NAMES = {
    "ESM3": "esm3-service",
    "BIO_TWIN": "bio-twin",
    "SYMBIOTIC": "symbiotic-service", 
    "GATEWAY": "gateway",
    "NOTIFICATIONS": "notifications"
}

# Timeline Event Types
TIMELINE_EVENT_TYPES = {
    "INSIGHT_GENERATED": "insight_generated",
    "INTERVENTION_APPLIED": "intervention_applied", 
    "FEEDBACK_RECEIVED": "feedback_received",
    "METRIC_UPDATED": "metric_updated",
    "ANALYSIS_COMPLETED": "analysis_completed",
    "GOAL_ACHIEVED": "goal_achieved",
    "MUTATION_PREDICTED": "mutation_predicted",
    "EVOLUTION_STEP": "evolution_step"
}

# Feedback Quality Thresholds
FEEDBACK_QUALITY = {
    "HIGH_THRESHOLD": 0.8,
    "MEDIUM_THRESHOLD": 0.6,
    "LOW_THRESHOLD": 0.4,
    "MIN_CONSISTENCY_SAMPLES": 3
}

# Risk Levels
RISK_LEVELS = {
    "LOW": "low",
    "MEDIUM": "medium", 
    "HIGH": "high"
}

# Notification Priorities
NOTIFICATION_PRIORITY = {
    "LOW": "low",
    "MEDIUM": "medium",
    "HIGH": "high", 
    "URGENT": "urgent"
}

# Default Configuration
DEFAULT_CONFIG = {
    "explainability_enabled": True,
    "feedback_scoring_enabled": True,
    "timeline_generation_enabled": True,
    "batch_processing_enabled": True,
    "notifications_enabled": True,
    "max_timeline_events": 1000,
    "feedback_retention_days": 365,
    "notification_retention_days": 30
}

# Cache Keys
CACHE_KEYS = {
    "USER_TIMELINE": "timeline:{user_id}",
    "USER_FEEDBACK_HISTORY": "feedback_history:{user_id}",
    "USER_INSIGHTS": "insights:{user_id}",
    "USER_NOTIFICATIONS": "notifications:{user_id}",
    "MODEL_METADATA": "model_metadata:{service}",
    "FEEDBACK_QUALITY_SCORES": "feedback_quality:{user_id}"
}

# API Response Formats
API_SUCCESS_FORMAT = {
    "status": "success",
    "data": None,
    "metadata": {
        "timestamp": None,
        "version": None,
        "request_id": None
    }
}

API_ERROR_FORMAT = {
    "status": "error",
    "error": {
        "code": None,
        "message": None,
        "details": None
    },
    "metadata": {
        "timestamp": None,
        "request_id": None
    }
}
