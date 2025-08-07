"""
EvoHuman.AI AiCE (AI Cognitive Enhancer) Service
Memory, reasoning, meditative alignment, and wisdom engine
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import httpx
from typing import Dict, Any, List, Optional
import structlog
import asyncio
import numpy as np
from datetime import datetime, timedelta

from shared.models import (
    ModelMetadata, ExplanationData, BioMetric, TwinInsight
)
from shared.constants import MODEL_VERSIONS, RISK_LEVELS
from shared.utils import setup_logging, create_health_check_response, generate_id, utc_now
from .memory_graph import MemoryGraphEngine
from .intuition_module import IntuitionModule
from .wisdom_engine import WisdomEngine
from .meditation_aligner import MeditationAligner


# Setup logging
logger = setup_logging("aice-service")

# Global components
memory_graph: Optional[MemoryGraphEngine] = None
intuition_module: Optional[IntuitionModule] = None
wisdom_engine: Optional[WisdomEngine] = None
meditation_aligner: Optional[MeditationAligner] = None
service_clients: Dict[str, httpx.AsyncClient] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global memory_graph, intuition_module, wisdom_engine, meditation_aligner, service_clients
    
    logger.info("Starting AiCE (AI Cognitive Enhancer) Service")
    
    # Initialize service clients
    services = {
        "bio_twin": os.getenv("BIO_TWIN_SERVICE_URL", "http://bio-twin:8000"),
        "symbiotic": os.getenv("SYMBIOTIC_SERVICE_URL", "http://symbiotic-service:8000"),
        "notifications": os.getenv("NOTIFICATIONS_SERVICE_URL", "http://notifications:8000")
    }
    
    for name, url in services.items():
        service_clients[name] = httpx.AsyncClient(base_url=url, timeout=60.0)
    
    # Initialize AiCE components
    try:
        memory_graph = MemoryGraphEngine()
        await memory_graph.initialize()
        
        intuition_module = IntuitionModule(memory_graph)
        await intuition_module.initialize()
        
        wisdom_engine = WisdomEngine(memory_graph, service_clients)
        await wisdom_engine.initialize()
        
        meditation_aligner = MeditationAligner(wisdom_engine)
        await meditation_aligner.initialize()
        
        logger.info("AiCE service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize AiCE service", error=str(e))
        logger.warning("Running in development mode")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AiCE service")
    for client in service_clients.values():
        await client.aclose()
    
    if memory_graph:
        await memory_graph.cleanup()


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI AiCE (AI Cognitive Enhancer) Service",
    description="Memory, reasoning, meditative alignment, and wisdom engine",
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {
        "memory_graph": memory_graph is not None,
        "intuition_module": intuition_module is not None,
        "wisdom_engine": wisdom_engine is not None,
        "meditation_aligner": meditation_aligner is not None
    }
    
    # Check service connections
    for name, client in service_clients.items():
        try:
            response = await client.get("/health", timeout=5.0)
            dependencies[f"service_{name}"] = response.status_code == 200
        except Exception:
            dependencies[f"service_{name}"] = False
    
    return create_health_check_response("aice-service", dependencies)


# MEMORY GRAPH OPERATIONS

@app.post("/memory/store")
async def store_memory(
    memory_request: Dict[str, Any]
):
    """Store information in the memory graph"""
    
    if not memory_graph:
        return await _mock_memory_storage(memory_request)
    
    try:
        user_id = memory_request.get("user_id")
        memory_type = memory_request.get("type", "experience")
        content = memory_request.get("content", {})
        importance = memory_request.get("importance", 0.5)
        tags = memory_request.get("tags", [])
        
        memory_id = await memory_graph.store_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags
        )
        
        return {
            "memory_id": memory_id,
            "status": "stored",
            "user_id": user_id,
            "type": memory_type,
            "importance": importance,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Memory storage failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store memory")


@app.get("/memory/recall/{user_id}")
async def recall_memories(
    user_id: str,
    query: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 10,
    importance_threshold: float = 0.3
):
    """Recall memories from the memory graph"""
    
    if not memory_graph:
        return await _mock_memory_recall(user_id, query, memory_type, limit)
    
    try:
        memories = await memory_graph.recall_memories(
            user_id=user_id,
            query=query,
            memory_type=memory_type,
            limit=limit,
            importance_threshold=importance_threshold
        )
        
        return {
            "user_id": user_id,
            "query": query,
            "memories_found": len(memories),
            "memories": memories,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Memory recall failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to recall memories")


@app.post("/memory/connect")
async def create_memory_connections(
    connection_request: Dict[str, Any]
):
    """Create connections between memories"""
    
    if not memory_graph:
        return await _mock_memory_connection(connection_request)
    
    try:
        user_id = connection_request.get("user_id")
        memory_ids = connection_request.get("memory_ids", [])
        connection_type = connection_request.get("connection_type", "related")
        strength = connection_request.get("strength", 0.5)
        
        connections = await memory_graph.create_connections(
            user_id=user_id,
            memory_ids=memory_ids,
            connection_type=connection_type,
            strength=strength
        )
        
        return {
            "user_id": user_id,
            "connections_created": len(connections),
            "connections": connections,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Memory connection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create memory connections")


# INTUITION MODULE

@app.post("/intuition/analyze")
async def analyze_intuition(
    intuition_request: Dict[str, Any]
):
    """Analyze patterns and generate intuitive insights"""
    
    if not intuition_module:
        return await _mock_intuition_analysis(intuition_request)
    
    try:
        user_id = intuition_request.get("user_id")
        context = intuition_request.get("context", {})
        focus_areas = intuition_request.get("focus_areas", [])
        
        intuition_results = await intuition_module.analyze_patterns(
            user_id=user_id,
            context=context,
            focus_areas=focus_areas
        )
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_name="aice-intuition",
            model_version=MODEL_VERSIONS["aice"],
            timestamp=utc_now(),
            processing_time=intuition_results.get("processing_time", 1.5)
        )
        
        # Create explanation for insights
        explanation = ExplanationData(
            reason="Intuitive insights generated through pattern analysis of user's memory graph, historical decisions, and subconscious indicators.",
            model_metadata=model_metadata,
            confidence_score=intuition_results.get("confidence", 0.75),
            risk_level=RISK_LEVELS["LOW"],
            biological_relevance="Insights based on cognitive patterns and decision-making history that may reflect subconscious processing.",
            contributing_factors=intuition_results.get("contributing_factors", [])
        )
        
        return {
            "user_id": user_id,
            "analysis_id": generate_id(),
            "insights": intuition_results.get("insights", []),
            "patterns_detected": intuition_results.get("patterns", []),
            "confidence_score": intuition_results.get("confidence", 0.75),
            "explanation": explanation.model_dump(),
            "recommendations": intuition_results.get("recommendations", []),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Intuition analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to analyze intuition")


@app.post("/intuition/validate")
async def validate_intuition(
    validation_request: Dict[str, Any]
):
    """Validate intuitive insights against data and outcomes"""
    
    if not intuition_module:
        return await _mock_intuition_validation(validation_request)
    
    try:
        user_id = validation_request.get("user_id")
        intuition_id = validation_request.get("intuition_id")
        actual_outcome = validation_request.get("outcome", {})
        feedback_score = validation_request.get("feedback_score", 0.5)
        
        validation_results = await intuition_module.validate_insight(
            user_id=user_id,
            intuition_id=intuition_id,
            actual_outcome=actual_outcome,
            feedback_score=feedback_score
        )
        
        return {
            "user_id": user_id,
            "intuition_id": intuition_id,
            "validation_results": validation_results,
            "accuracy_score": validation_results.get("accuracy", 0.0),
            "learning_update": "Intuition model updated based on validation",
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Intuition validation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to validate intuition")


# WISDOM ENGINE

@app.post("/wisdom/seek")
async def seek_wisdom(
    wisdom_request: Dict[str, Any]
):
    """Seek wisdom on complex life decisions and situations"""
    
    if not wisdom_engine:
        return await _mock_wisdom_seeking(wisdom_request)
    
    try:
        user_id = wisdom_request.get("user_id")
        situation = wisdom_request.get("situation", "")
        dilemma = wisdom_request.get("dilemma", {})
        context = wisdom_request.get("context", {})
        
        wisdom_response = await wisdom_engine.provide_wisdom(
            user_id=user_id,
            situation=situation,
            dilemma=dilemma,
            context=context
        )
        
        return {
            "user_id": user_id,
            "wisdom_id": generate_id(),
            "situation": situation,
            "wisdom_response": wisdom_response,
            "confidence": wisdom_response.get("confidence", 0.8),
            "perspectives": wisdom_response.get("perspectives", []),
            "recommendations": wisdom_response.get("recommendations", []),
            "philosophical_framework": wisdom_response.get("framework", "integrated"),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Wisdom seeking failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to provide wisdom")


@app.post("/wisdom/reflect")
async def reflect_on_experience(
    reflection_request: Dict[str, Any]
):
    """Reflect on experiences to extract wisdom and learning"""
    
    if not wisdom_engine:
        return await _mock_wisdom_reflection(reflection_request)
    
    try:
        user_id = reflection_request.get("user_id")
        experience = reflection_request.get("experience", {})
        emotions = reflection_request.get("emotions", [])
        outcomes = reflection_request.get("outcomes", {})
        
        reflection_results = await wisdom_engine.reflect_on_experience(
            user_id=user_id,
            experience=experience,
            emotions=emotions,
            outcomes=outcomes
        )
        
        return {
            "user_id": user_id,
            "reflection_id": generate_id(),
            "experience_summary": reflection_results.get("summary", ""),
            "lessons_learned": reflection_results.get("lessons", []),
            "wisdom_extracted": reflection_results.get("wisdom", []),
            "growth_opportunities": reflection_results.get("growth", []),
            "emotional_insights": reflection_results.get("emotional_insights", []),
            "integration_suggestions": reflection_results.get("integration", []),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Wisdom reflection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reflect on experience")


# MEDITATION ALIGNER

@app.post("/meditation/align")
async def align_meditation_practice(
    alignment_request: Dict[str, Any]
):
    """Align meditation practice with personal growth goals"""
    
    if not meditation_aligner:
        return await _mock_meditation_alignment(alignment_request)
    
    try:
        user_id = alignment_request.get("user_id")
        current_practice = alignment_request.get("current_practice", {})
        goals = alignment_request.get("goals", [])
        challenges = alignment_request.get("challenges", [])
        
        alignment_results = await meditation_aligner.create_alignment(
            user_id=user_id,
            current_practice=current_practice,
            goals=goals,
            challenges=challenges
        )
        
        return {
            "user_id": user_id,
            "alignment_id": generate_id(),
            "personalized_practice": alignment_results.get("practice", {}),
            "meditation_techniques": alignment_results.get("techniques", []),
            "progression_plan": alignment_results.get("progression", []),
            "integration_guidance": alignment_results.get("integration", []),
            "progress_metrics": alignment_results.get("metrics", []),
            "estimated_duration": alignment_results.get("duration", "4-6 weeks"),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Meditation alignment failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to align meditation practice")


@app.get("/meditation/progress/{user_id}")
async def track_meditation_progress(
    user_id: str,
    timeframe_days: int = 30
):
    """Track meditation and mindfulness progress"""
    
    if not meditation_aligner:
        return await _mock_meditation_progress(user_id, timeframe_days)
    
    try:
        progress_data = await meditation_aligner.track_progress(
            user_id=user_id,
            timeframe_days=timeframe_days
        )
        
        return {
            "user_id": user_id,
            "timeframe_days": timeframe_days,
            "progress_summary": progress_data,
            "consistency_score": progress_data.get("consistency", 0.0),
            "depth_improvement": progress_data.get("depth_improvement", 0.0),
            "integration_level": progress_data.get("integration", 0.0),
            "insights_generated": len(progress_data.get("insights", [])),
            "next_recommendations": progress_data.get("next_steps", []),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Meditation progress tracking failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to track meditation progress")


# COGNITIVE ENHANCEMENT

@app.post("/cognitive/enhance")
async def enhance_cognitive_abilities(
    enhancement_request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Enhance cognitive abilities through personalized training"""
    
    try:
        user_id = enhancement_request.get("user_id")
        target_abilities = enhancement_request.get("abilities", ["memory", "focus", "creativity"])
        current_level = enhancement_request.get("current_level", {})
        training_intensity = enhancement_request.get("intensity", "moderate")
        
        enhancement_id = generate_id()
        
        # Start enhancement process in background
        background_tasks.add_task(
            _run_cognitive_enhancement,
            enhancement_id,
            user_id,
            target_abilities,
            current_level,
            training_intensity
        )
        
        return {
            "enhancement_id": enhancement_id,
            "user_id": user_id,
            "status": "started",
            "target_abilities": target_abilities,
            "training_intensity": training_intensity,
            "estimated_duration": "2-4 weeks",
            "message": "Cognitive enhancement program initiated"
        }
        
    except Exception as e:
        logger.error("Cognitive enhancement failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start cognitive enhancement")


@app.get("/cognitive/assessment/{user_id}")
async def assess_cognitive_state(user_id: str):
    """Assess current cognitive state and capabilities"""
    
    try:
        # Simulate cognitive assessment
        assessment_results = await _perform_cognitive_assessment(user_id)
        
        return {
            "user_id": user_id,
            "assessment_id": generate_id(),
            "cognitive_profile": assessment_results,
            "strengths": assessment_results.get("strengths", []),
            "improvement_areas": assessment_results.get("improvement_areas", []),
            "overall_score": assessment_results.get("overall_score", 0.75),
            "recommendations": assessment_results.get("recommendations", []),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Cognitive assessment failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to assess cognitive state")


# MOCK IMPLEMENTATIONS FOR DEVELOPMENT

async def _mock_memory_storage(memory_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock memory storage"""
    await asyncio.sleep(0.1)
    return {
        "memory_id": generate_id(),
        "status": "stored_mock",
        "user_id": memory_request.get("user_id"),
        "type": memory_request.get("type", "experience"),
        "mock": True
    }


async def _mock_memory_recall(user_id: str, query: str, memory_type: str, limit: int) -> Dict[str, Any]:
    """Mock memory recall"""
    await asyncio.sleep(0.2)
    
    mock_memories = [
        {
            "memory_id": generate_id(),
            "type": "insight",
            "content": "Previous breakthrough in protein stability optimization",
            "importance": 0.9,
            "timestamp": (utc_now() - timedelta(days=5)).isoformat()
        },
        {
            "memory_id": generate_id(),
            "type": "experience",
            "content": "Successful meditation session with deep focus state",
            "importance": 0.7,
            "timestamp": (utc_now() - timedelta(days=2)).isoformat()
        }
    ]
    
    return {
        "user_id": user_id,
        "query": query,
        "memories_found": len(mock_memories),
        "memories": mock_memories[:limit],
        "mock": True
    }


async def _mock_memory_connection(connection_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock memory connections"""
    await asyncio.sleep(0.15)
    
    mock_connections = [
        {
            "connection_id": generate_id(),
            "from_memory": connection_request.get("memory_ids", [])[0] if connection_request.get("memory_ids") else generate_id(),
            "to_memory": connection_request.get("memory_ids", [])[1] if len(connection_request.get("memory_ids", [])) > 1 else generate_id(),
            "type": connection_request.get("connection_type", "related"),
            "strength": connection_request.get("strength", 0.5),
            "timestamp": utc_now().isoformat()
        }
    ]
    
    return {
        "user_id": connection_request.get("user_id"),
        "connections_created": len(mock_connections),
        "connections": mock_connections,
        "mock": True
    }


async def _mock_intuition_analysis(intuition_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock intuition analysis"""
    await asyncio.sleep(0.5)
    
    return {
        "insights": [
            "Your recent stress patterns suggest a breakthrough is imminent",
            "The timing alignment between your meditation practice and creative work is optimal",
            "Consider exploring the connection between your nutritional changes and energy levels"
        ],
        "patterns": [
            "Cyclical creativity peaks every 3-4 days",
            "Stress reduction correlates with meditation consistency",
            "Decision confidence improves with adequate sleep"
        ],
        "confidence": 0.78,
        "contributing_factors": [
            "Historical decision patterns",
            "Biorhythm analysis",
            "Subconscious indicator processing"
        ],
        "processing_time": 0.5,
        "mock": True
    }


async def _mock_intuition_validation(validation_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock intuition validation"""
    await asyncio.sleep(0.3)
    
    return {
        "validation_results": {
            "accuracy": 0.73,
            "precision": 0.68,
            "relevance": 0.81,
            "learning_impact": 0.62
        },
        "accuracy": 0.73,
        "improvement_suggestions": [
            "Consider longer observation periods for pattern detection",
            "Include more contextual factors in analysis"
        ],
        "mock": True
    }


async def _mock_wisdom_seeking(wisdom_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock wisdom seeking"""
    await asyncio.sleep(0.3)
    
    return {
        "wisdom_response": {
            "primary_insight": "The path of growth often requires embracing temporary discomfort for long-term flourishing",
            "perspectives": [
                "Stoic: Focus on what you can control, accept what you cannot",
                "Buddhist: Suffering arises from attachment; practice non-attachment",
                "Modern Psychology: Growth mindset transforms challenges into opportunities"
            ],
            "recommendations": [
                "Take time for deep reflection before major decisions",
                "Seek diverse perspectives from trusted advisors",
                "Consider both short-term and long-term consequences",
                "Trust your intuition while validating with rational analysis"
            ],
            "framework": "integrated_wisdom",
            "confidence": 0.85
        },
        "mock": True
    }


async def _mock_wisdom_reflection(reflection_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock wisdom reflection"""
    await asyncio.sleep(0.4)
    
    return {
        "summary": "A challenging experience that provided valuable growth opportunities",
        "lessons": [
            "Patience and persistence lead to breakthrough moments",
            "Emotional regulation is key during difficult periods",
            "Support systems are crucial for maintaining perspective"
        ],
        "wisdom": [
            "Every challenge contains the seeds of wisdom",
            "Growth happens at the edge of comfort zones",
            "Integration requires both understanding and practice"
        ],
        "growth": [
            "Develop stronger emotional resilience",
            "Practice mindful response over reactive patterns",
            "Cultivate deeper self-compassion"
        ],
        "emotional_insights": [
            "Fear often masks excitement for growth",
            "Vulnerability creates deeper connections"
        ],
        "integration": [
            "Daily reflection practice",
            "Apply lessons in similar future situations",
            "Share wisdom with others facing similar challenges"
        ],
        "mock": True
    }


async def _mock_meditation_alignment(alignment_request: Dict[str, Any]) -> Dict[str, Any]:
    """Mock meditation alignment"""
    await asyncio.sleep(0.4)
    
    return {
        "practice": {
            "morning_session": {
                "duration": 20,
                "technique": "mindfulness_breathing",
                "focus": "presence_cultivation"
            },
            "evening_session": {
                "duration": 15,
                "technique": "body_scan",
                "focus": "tension_release"
            }
        },
        "techniques": [
            "Focused attention meditation for concentration improvement",
            "Open awareness practice for creativity enhancement",
            "Loving-kindness meditation for emotional balance"
        ],
        "progression": [
            "Week 1-2: Establish consistent daily practice",
            "Week 3-4: Deepen focus and extend duration",
            "Week 5-6: Integrate mindfulness into daily activities"
        ],
        "duration": "6 weeks",
        "mock": True
    }


async def _mock_meditation_progress(user_id: str, timeframe_days: int) -> Dict[str, Any]:
    """Mock meditation progress"""
    await asyncio.sleep(0.2)
    
    return {
        "consistency": 0.78,
        "depth_improvement": 0.65,
        "integration": 0.72,
        "insights": ["Increased emotional awareness", "Better stress response"],
        "next_steps": ["Extend morning session to 25 minutes", "Add walking meditation"]
    }


async def _run_cognitive_enhancement(
    enhancement_id: str,
    user_id: str,
    abilities: List[str],
    current_level: Dict[str, Any],
    intensity: str
):
    """Run cognitive enhancement program (background task)"""
    # Mock implementation - would run actual cognitive training
    await asyncio.sleep(2.0)
    logger.info(f"Cognitive enhancement {enhancement_id} completed for user {user_id}")


async def _perform_cognitive_assessment(user_id: str) -> Dict[str, Any]:
    """Perform cognitive assessment"""
    await asyncio.sleep(0.8)
    
    return {
        "memory": {"score": 0.82, "percentile": 78},
        "attention": {"score": 0.75, "percentile": 68},
        "processing_speed": {"score": 0.88, "percentile": 85},
        "creativity": {"score": 0.79, "percentile": 72},
        "emotional_intelligence": {"score": 0.84, "percentile": 80},
        "overall_score": 0.816,
        "strengths": ["Processing Speed", "Emotional Intelligence"],
        "improvement_areas": ["Sustained Attention", "Working Memory"],
        "recommendations": [
            "Practice focused attention meditation daily",
            "Engage in memory palace techniques",
            "Use spaced repetition for learning"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
