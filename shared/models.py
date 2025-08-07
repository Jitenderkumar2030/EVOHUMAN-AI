"""
Shared data models for EvoHuman.AI platform
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class EvolutionState(str, Enum):
    BASELINE = "baseline"
    ANALYZING = "analyzing"
    EVOLVING = "evolving"
    OPTIMIZED = "optimized"


class BioMetricType(str, Enum):
    GENETIC = "genetic"
    PHYSIOLOGICAL = "physiological"
    COGNITIVE = "cognitive"
    BEHAVIORAL = "behavioral"


# User Models
class UserBase(BaseModel):
    email: str
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: str
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# BioDigital Twin Models
class BioMetric(BaseModel):
    type: BioMetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # ESM3, Proteus, AiCE, etc.


class EvolutionGoal(BaseModel):
    id: str
    user_id: str
    category: str  # longevity, cognitive, physical, spiritual
    target_metric: str
    current_value: float
    target_value: float
    timeline_days: int
    priority: int = Field(ge=1, le=10)
    created_at: datetime


class BioTwinSnapshot(BaseModel):
    id: str
    user_id: str
    state: EvolutionState
    metrics: List[BioMetric]
    insights: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime


# AI Service Models
class AIServiceRequest(BaseModel):
    user_id: str
    service_type: str  # esm3, proteus, aice, symbiotic
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = {}
    priority: int = 1


class AIServiceResponse(BaseModel):
    request_id: str
    service_type: str
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime


# ExoStack Models
class ComputeNode(BaseModel):
    id: str
    name: str
    host: str
    port: int
    capabilities: List[str]  # cpu, gpu, memory_gb, etc.
    status: str  # online, offline, busy
    last_heartbeat: datetime


class ComputeJob(BaseModel):
    id: str
    user_id: str
    job_type: str
    config: Dict[str, Any]
    node_requirements: Dict[str, Any]
    status: str  # queued, running, completed, failed
    assigned_node: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Evolution Engine Models
class FeedbackLoop(BaseModel):
    id: str
    user_id: str
    loop_type: str  # reinforcement, correction, enhancement
    input_data: Dict[str, Any]
    human_feedback: Dict[str, Any]
    ai_response: Dict[str, Any]
    improvement_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime


class EvolutionPlan(BaseModel):
    id: str
    user_id: str
    goals: List[EvolutionGoal]
    current_phase: str
    phases: List[Dict[str, Any]]
    estimated_duration: int  # days
    success_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime


# ESM3 Protein Analysis Models
class ProteinSequence(BaseModel):
    sequence: str = Field(min_length=10, max_length=2048)
    name: Optional[str] = None
    description: Optional[str] = None


class MutationRequest(BaseModel):
    position: int = Field(ge=0)
    from_aa: str = Field(min_length=1, max_length=1)
    to_aa: str = Field(min_length=1, max_length=1)


class ProteinAnalysisRequest(BaseModel):
    sequence: str = Field(min_length=10, max_length=2048)
    analysis_type: str = "structure_prediction"
    include_mutations: bool = False
    include_evolution: bool = False
    user_id: Optional[str] = None


class ProteinAnalysisResult(BaseModel):
    sequence_id: str
    sequence: str
    sequence_length: int
    predicted_structure: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    analysis_type: str
    contact_map: Optional[Dict[str, Any]] = None
    mutation_analysis: Optional[Dict[str, Any]] = None
    evolution_suggestion: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: datetime
    status: str
    mock: bool = False


class MutationEffect(BaseModel):
    mutation: str
    stability_change: float
    confidence: float = Field(ge=0.0, le=1.0)
    effect_category: str  # stabilizing, destabilizing, neutral
    recommendation: str


class EvolutionPathway(BaseModel):
    pathway_id: str
    pathway_type: str
    target_positions: List[int]
    suggested_mutations: List[str]
    predicted_improvement: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    estimated_steps: int
    priority: str  # high, medium, low


# New Models for Enhanced Features
class ModelMetadata(BaseModel):
    """Model version and audit metadata"""
    model_name: str
    model_version: str
    timestamp: datetime
    request_id: Optional[str] = None
    processing_time: Optional[float] = None


class ExplanationData(BaseModel):
    """Explainability information for AI insights"""
    reason: str
    model_metadata: ModelMetadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(pattern="^(low|medium|high)$")  # low, medium, high
    biological_relevance: str
    contributing_factors: List[str] = []
    alternative_explanations: Optional[List[str]] = None


class TwinInsight(BaseModel):
    """Enhanced twin insight with explainability"""
    id: str
    user_id: str
    insight_type: str  # mutation, recommendation, warning, optimization
    content: str
    priority: int = Field(ge=1, le=10)
    explanation: ExplanationData
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: str = "active"  # active, archived, dismissed


class ESM3AnalysisResult(BaseModel):
    """Enhanced ESM3 analysis result with explainability"""
    sequence_id: str
    sequence: str
    sequence_length: int
    predicted_structure: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    analysis_type: str
    contact_map: Optional[Dict[str, Any]] = None
    mutation_analysis: Optional[Dict[str, Any]] = None
    evolution_suggestion: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: datetime
    status: str
    mock: bool = False
    # New explainability fields
    explanation: ExplanationData
    model_metadata: ModelMetadata


class FeedbackQualityScore(BaseModel):
    """Quality assessment for user feedback"""
    score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    reliability_factors: List[str] = []
    timestamp: datetime


class EnhancedFeedback(BaseModel):
    """Enhanced feedback with quality scoring"""
    id: str
    user_id: str
    insight_id: Optional[str] = None
    intervention_id: Optional[str] = None
    feedback_type: str  # rating, text, behavioral, physiological
    content: Dict[str, Any]
    quality_score: FeedbackQualityScore
    created_at: datetime
    context: Optional[Dict[str, Any]] = None


class TimelineEvent(BaseModel):
    """Timeline event for user evolution tracking"""
    id: str
    user_id: str
    event_type: str  # insight, intervention, feedback, metric_update, analysis
    timestamp: datetime
    title: str
    description: str
    data: Dict[str, Any]
    impact_score: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    tags: List[str] = []
    source_service: str  # bio-twin, esm3, symbiotic


class UserEvolutionTimeline(BaseModel):
    """Complete user evolution timeline"""
    user_id: str
    timeline_id: str
    events: List[TimelineEvent]
    summary_stats: Dict[str, Any]
    generated_at: datetime
    last_updated: datetime
    total_events: int
    date_range: Dict[str, datetime]  # start_date, end_date


class BatchFeedbackRequest(BaseModel):
    """Batch feedback processing request"""
    user_id: str
    feedback_entries: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None
    priority: int = Field(ge=1, le=10, default=5)


class BatchFeedbackResponse(BaseModel):
    """Batch feedback processing response"""
    batch_id: str
    user_id: str
    total_entries: int
    processed: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime


class NotificationMessage(BaseModel):
    """User notification message"""
    id: str
    user_id: str
    message_type: str  # insight, reminder, alert, achievement
    title: str
    content: str
    priority: str = Field(pattern="^(low|medium|high|urgent)$")
    created_at: datetime
    expires_at: Optional[datetime] = None
    read: bool = False
    actions: Optional[List[Dict[str, str]]] = None  # clickable actions


class NotificationQueue(BaseModel):
    """User notification queue"""
    user_id: str
    messages: List[NotificationMessage]
    last_updated: datetime
    unread_count: int


# Enhanced existing models
class EnhancedBioTwinSnapshot(BaseModel):
    """Enhanced bio-twin snapshot with timeline"""
    id: str
    user_id: str
    state: EvolutionState
    metrics: List[BioMetric]
    insights: List[TwinInsight]  # Changed from Dict to List of TwinInsight
    recommendations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    # New fields
    timeline: Optional[UserEvolutionTimeline] = None
    model_metadata: ModelMetadata
    explanation: Optional[ExplanationData] = None
