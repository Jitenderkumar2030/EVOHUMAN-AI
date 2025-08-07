import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List, Any
import redis
import json
import faiss
import numpy as np
from datetime import datetime
import structlog

from shared.models import (
    BatchFeedbackRequest, BatchFeedbackResponse, EnhancedFeedback,
    FeedbackQualityScore, ModelMetadata
)
from shared.constants import MODEL_VERSIONS, FEEDBACK_QUALITY
from shared.utils import setup_logging, generate_id, utc_now
from .feedback_scorer import FeedbackQualityScorer
from .multi_agent_system import MultiAgentSystem, AgentObservation, AgentReward

# Models
class UserInput(BaseModel):
    user_id: str
    context: str
    bio_metrics: Dict[str, float]
    esm3_insight: str
    goal: str

class SymbioticResponse(BaseModel):
    suggested_intervention: str
    confidence: float
    learning_mode: str

class TrainingSignal(BaseModel):
    user_id: str
    intervention_id: str
    feedback_score: float
    context: Dict[str, any]

# Setup logging
logger = setup_logging("symbiotic-ais")

# Global components
multi_agent_systems: Dict[str, MultiAgentSystem] = {}  # Per-user systems
feedback_scorer: Optional[FeedbackQualityScorer] = None

# Initialize FastAPI
app = FastAPI(
    title="EvoHuman.AI SymbioticAIS Service",
    description="Enhanced feedback processing with quality scoring and batch operations",
    version="1.5.0"
)

# Initialize Redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Initialize FAISS vector store
VECTOR_DIM = 128  # Dimension of our embedding space
index = faiss.IndexFlatL2(VECTOR_DIM)

# Initialize feedback quality scorer
feedback_scorer = FeedbackQualityScorer(redis_client)

def get_embedding(context: dict) -> np.ndarray:
    """Convert context to vector embedding"""
    # Placeholder: Replace with actual embedding logic
    return np.random.rand(VECTOR_DIM).astype('float32')

@app.post("/symbiotic/analyze")
async def analyze_input(user_input: UserInput) -> SymbioticResponse:
    try:
        # Get user history from Redis
        user_history = redis_client.get(f"history:{user_input.user_id}")
        if user_history:
            user_history = json.loads(user_history)
        else:
            user_history = []

        # Create context vector
        context_vector = get_embedding({
            "bio_metrics": user_input.bio_metrics,
            "esm3_insight": user_input.esm3_insight,
            "goal": user_input.goal,
            "history": user_history[-5:]  # Last 5 interactions
        })

        # Search similar patterns in FAISS
        if index.ntotal > 0:
            D, I = index.search(context_vector.reshape(1, -1), 1)
            confidence = 1.0 / (1.0 + D[0][0])  # Convert distance to confidence
        else:
            confidence = 0.5  # Default confidence for cold start

        # Generate intervention based on context
        intervention = generate_intervention(user_input, confidence)

        # Store interaction
        user_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": user_input.dict(),
            "intervention": intervention
        })
        redis_client.set(
            f"history:{user_input.user_id}",
            json.dumps(user_history[-100:])  # Keep last 100 interactions
        )

        return SymbioticResponse(
            suggested_intervention=intervention,
            confidence=float(confidence),
            learning_mode="active" if confidence < 0.7 else "stable"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbiotic/state/{user_id}")
async def get_state(user_id: str):
    try:
        history = redis_client.get(f"history:{user_id}")
        if not history:
            return {"state": "new_user", "interactions": 0}

        history = json.loads(history)
        return {
            "state": "learning" if len(history) < 10 else "trained",
            "interactions": len(history),
            "last_interaction": history[-1]["timestamp"] if history else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/symbiotic/train")
async def train_model(signal: TrainingSignal):
    try:
        # Get context vector
        context_vector = get_embedding(signal.context)

        # Update FAISS index with feedback
        index.add(context_vector.reshape(1, -1))

        # Store feedback in Redis
        feedback_key = f"feedback:{signal.user_id}:{signal.intervention_id}"
        redis_client.set(feedback_key, signal.feedback_score)

        return {"status": "success", "message": "Feedback incorporated"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_intervention(user_input: UserInput, confidence: float) -> str:
    """Generate personalized intervention based on context and confidence"""
    if user_input.bio_metrics.get("energy", 0) < 0.5:
        return "Take a 20-min power nap followed by light exercise"
    elif user_input.bio_metrics.get("focus", 0) < 0.6:
        return "Take 15-min cognitive pause. Avoid glucose spike."
    else:
        return "Maintain current routine. Optimal state detected."


# ENHANCED FEEDBACK PROCESSING ENDPOINTS

@app.post("/feedback/enhanced")
async def submit_enhanced_feedback(
    user_id: str,
    feedback_content: Dict[str, Any],
    feedback_type: str,
    insight_id: Optional[str] = None,
    intervention_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> EnhancedFeedback:
    """Submit feedback with quality scoring"""
    
    try:
        # Score feedback quality
        quality_score = feedback_scorer.score_feedback(
            user_id=user_id,
            feedback_content=feedback_content,
            feedback_type=feedback_type,
            context=context
        )
        
        # Create enhanced feedback
        enhanced_feedback = EnhancedFeedback(
            id=generate_id(),
            user_id=user_id,
            insight_id=insight_id,
            intervention_id=intervention_id,
            feedback_type=feedback_type,
            content=feedback_content,
            quality_score=quality_score,
            created_at=utc_now(),
            context=context
        )
        
        # Store enhanced feedback
        feedback_key = f"enhanced_feedback:{user_id}:{enhanced_feedback.id}"
        redis_client.setex(
            feedback_key,
            86400 * 30,  # 30 days
            json.dumps(enhanced_feedback.model_dump(), default=str)
        )
        
        # Update learning system with quality-weighted feedback
        if quality_score.score >= FEEDBACK_QUALITY["MEDIUM_THRESHOLD"]:
            # Only use high-quality feedback for learning
            await _update_learning_system(user_id, enhanced_feedback, quality_score.score)
        
        logger.info(
            "Enhanced feedback processed",
            user_id=user_id,
            feedback_type=feedback_type,
            quality_score=quality_score.score,
            feedback_id=enhanced_feedback.id
        )
        
        return enhanced_feedback
        
    except Exception as e:
        logger.error("Failed to process enhanced feedback", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@app.post("/feedback/batch")
async def submit_batch_feedback(
    batch_request: BatchFeedbackRequest
) -> BatchFeedbackResponse:
    """Process multiple feedback entries in a single request"""
    
    start_time = utc_now()
    batch_id = generate_id()
    results = []
    processed_count = 0
    failed_count = 0
    
    try:
        logger.info(
            "Starting batch feedback processing",
            user_id=batch_request.user_id,
            batch_size=len(batch_request.feedback_entries),
            batch_id=batch_id
        )
        
        for i, feedback_entry in enumerate(batch_request.feedback_entries):
            try:
                # Extract feedback details
                feedback_content = feedback_entry.get("content", {})
                feedback_type = feedback_entry.get("feedback_type", "text")
                insight_id = feedback_entry.get("insight_id")
                intervention_id = feedback_entry.get("intervention_id")
                
                # Score feedback quality
                quality_score = feedback_scorer.score_feedback(
                    user_id=batch_request.user_id,
                    feedback_content=feedback_content,
                    feedback_type=feedback_type,
                    context=batch_request.context
                )
                
                # Create enhanced feedback
                enhanced_feedback = EnhancedFeedback(
                    id=generate_id(),
                    user_id=batch_request.user_id,
                    insight_id=insight_id,
                    intervention_id=intervention_id,
                    feedback_type=feedback_type,
                    content=feedback_content,
                    quality_score=quality_score,
                    created_at=utc_now(),
                    context=batch_request.context
                )
                
                # Store enhanced feedback
                feedback_key = f"enhanced_feedback:{batch_request.user_id}:{enhanced_feedback.id}"
                redis_client.setex(
                    feedback_key,
                    86400 * 30,  # 30 days
                    json.dumps(enhanced_feedback.model_dump(), default=str)
                )
                
                # Update learning system with quality-weighted feedback
                if quality_score.score >= FEEDBACK_QUALITY["MEDIUM_THRESHOLD"]:
                    await _update_learning_system(batch_request.user_id, enhanced_feedback, quality_score.score)
                
                results.append({
                    "index": i,
                    "feedback_id": enhanced_feedback.id,
                    "status": "success",
                    "quality_score": quality_score.score,
                    "processing_notes": f"Quality: {quality_score.score:.2f}, Used for learning: {quality_score.score >= FEEDBACK_QUALITY['MEDIUM_THRESHOLD']}"
                })
                
                processed_count += 1
                
            except Exception as e:
                logger.error(
                    "Failed to process feedback entry in batch",
                    user_id=batch_request.user_id,
                    batch_id=batch_id,
                    entry_index=i,
                    error=str(e)
                )
                
                results.append({
                    "index": i,
                    "status": "failed",
                    "error": str(e)
                })
                
                failed_count += 1
        
        processing_time = (utc_now() - start_time).total_seconds()
        
        response = BatchFeedbackResponse(
            batch_id=batch_id,
            user_id=batch_request.user_id,
            total_entries=len(batch_request.feedback_entries),
            processed=processed_count,
            failed=failed_count,
            results=results,
            processing_time=processing_time,
            timestamp=utc_now()
        )
        
        logger.info(
            "Batch feedback processing completed",
            user_id=batch_request.user_id,
            batch_id=batch_id,
            processed=processed_count,
            failed=failed_count,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Batch feedback processing failed",
            user_id=batch_request.user_id,
            batch_id=batch_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Batch feedback processing failed")


@app.get("/feedback/quality/{user_id}")
async def get_feedback_quality_summary(user_id: str):
    """Get summary of user's feedback quality metrics"""
    
    try:
        # Get feedback history from scorer
        history = feedback_scorer._get_feedback_history(user_id)
        
        if not history:
            return {
                "user_id": user_id,
                "total_feedback": 0,
                "average_quality_score": 0.0,
                "quality_distribution": {}
            }
        
        # Calculate quality metrics
        quality_scores = [item["quality_score"]["score"] for item in history]
        average_quality = sum(quality_scores) / len(quality_scores)
        
        # Quality distribution
        high_quality = len([s for s in quality_scores if s >= FEEDBACK_QUALITY["HIGH_THRESHOLD"]])
        medium_quality = len([s for s in quality_scores if FEEDBACK_QUALITY["MEDIUM_THRESHOLD"] <= s < FEEDBACK_QUALITY["HIGH_THRESHOLD"]])
        low_quality = len([s for s in quality_scores if s < FEEDBACK_QUALITY["MEDIUM_THRESHOLD"]])
        
        return {
            "user_id": user_id,
            "total_feedback": len(history),
            "average_quality_score": average_quality,
            "quality_distribution": {
                "high": high_quality,
                "medium": medium_quality,
                "low": low_quality
            },
            "recent_quality_trend": quality_scores[-10:] if len(quality_scores) >= 10 else quality_scores,
            "feedback_reliability": "high" if average_quality >= FEEDBACK_QUALITY["HIGH_THRESHOLD"] else "medium" if average_quality >= FEEDBACK_QUALITY["MEDIUM_THRESHOLD"] else "low"
        }
        
    except Exception as e:
        logger.error("Failed to get feedback quality summary", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve quality summary")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {
        "redis": True,
        "faiss_index": index.ntotal >= 0,
        "feedback_scorer": feedback_scorer is not None
    }
    
    # Test Redis connection
    try:
        redis_client.ping()
    except Exception:
        dependencies["redis"] = False
    
    from shared.utils import create_health_check_response
    return create_health_check_response("symbiotic-ais", dependencies)


async def _update_learning_system(
    user_id: str, 
    enhanced_feedback: EnhancedFeedback, 
    quality_weight: float
):
    """Update the learning system with quality-weighted feedback"""
    
    try:
        # Create weighted training signal
        training_signal = {
            "user_id": user_id,
            "feedback_id": enhanced_feedback.id,
            "feedback_content": enhanced_feedback.content,
            "quality_weight": quality_weight,
            "timestamp": enhanced_feedback.created_at.isoformat()
        }
        
        # Convert to context vector with quality weighting
        context_vector = get_embedding(enhanced_feedback.content) * quality_weight
        
        # Update FAISS index
        index.add(context_vector.reshape(1, -1))
        
        # Store weighted training signal
        signal_key = f"training_signal:{user_id}:{enhanced_feedback.id}"
        redis_client.setex(
            signal_key,
            86400 * 7,  # 7 days
            json.dumps(training_signal, default=str)
        )
        
        logger.debug(
            "Learning system updated with quality-weighted feedback",
            user_id=user_id,
            feedback_id=enhanced_feedback.id,
            quality_weight=quality_weight
        )
        
    except Exception as e:
        logger.error(
            "Failed to update learning system",
            user_id=user_id,
            feedback_id=enhanced_feedback.id,
            error=str(e)
        )


@app.post("/multi_agent/initialize")
async def initialize_multi_agent_system(user_id: str):
    """Initialize multi-agent system for user"""
    global multi_agent_systems

    try:
        if user_id not in multi_agent_systems:
            multi_agent_systems[user_id] = MultiAgentSystem(user_id)
            logger.info("Multi-agent system initialized", user_id=user_id)

        system_status = await multi_agent_systems[user_id].get_system_status()

        return {
            "user_id": user_id,
            "status": "initialized",
            "system_status": system_status,
            "timestamp": utc_now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to initialize multi-agent system", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/multi_agent/step")
async def execute_multi_agent_step(
    user_id: str,
    human_input: Optional[Dict[str, Any]] = None
):
    """Execute one step of the multi-agent system"""
    global multi_agent_systems

    try:
        # Initialize system if not exists
        if user_id not in multi_agent_systems:
            multi_agent_systems[user_id] = MultiAgentSystem(user_id)

        # Execute step
        step_results = await multi_agent_systems[user_id].step(human_input)

        logger.info("Multi-agent step executed",
                   user_id=user_id,
                   actions_taken=step_results["step_results"]["actions_taken"])

        return {
            "user_id": user_id,
            "step_results": step_results,
            "timestamp": utc_now().isoformat()
        }

    except Exception as e:
        logger.error("Multi-agent step failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Step execution failed: {str(e)}")


@app.get("/multi_agent/status/{user_id}")
async def get_multi_agent_status(user_id: str):
    """Get multi-agent system status"""
    global multi_agent_systems

    try:
        if user_id not in multi_agent_systems:
            return {
                "user_id": user_id,
                "status": "not_initialized",
                "message": "Multi-agent system not initialized for this user"
            }

        system_status = await multi_agent_systems[user_id].get_system_status()

        return {
            "user_id": user_id,
            "status": "active",
            "system_status": system_status,
            "timestamp": utc_now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get multi-agent status", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@app.post("/multi_agent/human_feedback")
async def process_human_feedback(
    user_id: str,
    feedback_data: Dict[str, Any]
):
    """Process human feedback through multi-agent system"""
    global multi_agent_systems

    try:
        # Initialize system if not exists
        if user_id not in multi_agent_systems:
            multi_agent_systems[user_id] = MultiAgentSystem(user_id)

        # Process feedback through human proxy agent
        human_input = {
            "satisfaction": feedback_data.get("satisfaction", 0.5),
            "feedback": feedback_data,
            "goals": feedback_data.get("goals", [])
        }

        # Execute step with human feedback
        step_results = await multi_agent_systems[user_id].step(human_input)

        # Generate response based on agent actions
        response = await _generate_symbiotic_response(user_id, step_results, feedback_data)

        return {
            "user_id": user_id,
            "feedback_processed": True,
            "symbiotic_response": response,
            "step_results": step_results,
            "timestamp": utc_now().isoformat()
        }

    except Exception as e:
        logger.error("Human feedback processing failed", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")


async def _generate_symbiotic_response(
    user_id: str,
    step_results: Dict[str, Any],
    feedback_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate symbiotic response based on multi-agent system results"""

    # Analyze agent actions to generate response
    agent_actions = step_results.get("agent_actions", {})
    system_performance = step_results["step_results"]["system_performance"]

    # Generate intervention suggestions
    interventions = []

    if "evolution_optimizer_1" in agent_actions:
        if agent_actions["evolution_optimizer_1"] == "optimize_evolution_path":
            interventions.append("Optimize your daily routine for longevity")
        elif agent_actions["evolution_optimizer_1"] == "suggest_intervention":
            interventions.append("Consider lifestyle modifications for better health")

    if "goal_coordinator_1" in agent_actions:
        if agent_actions["goal_coordinator_1"] == "coordinate_goals":
            interventions.append("Align your short-term actions with long-term goals")

    if "adaptation_specialist_1" in agent_actions:
        interventions.append("Adapt your approach based on recent progress")

    # Default intervention if none generated
    if not interventions:
        interventions.append("Continue with current approach and monitor progress")

    return {
        "suggested_interventions": interventions,
        "confidence": min(1.0, system_performance + 0.2),
        "learning_mode": "collaborative" if len(agent_actions) > 2 else "individual",
        "system_performance": system_performance,
        "agent_collaboration": len(agent_actions)
    }
