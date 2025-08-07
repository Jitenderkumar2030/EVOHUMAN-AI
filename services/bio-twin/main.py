"""
EvoHuman.AI BioDigital Twin Engine
Orchestrates biological and cognitive modeling
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

from shared.models import (
    BioTwinSnapshot, BioMetric, EvolutionGoal, EvolutionState,
    EnhancedBioTwinSnapshot, TwinInsight, ModelMetadata, UserEvolutionTimeline
)
from shared.constants import MODEL_VERSIONS, SERVICE_NAMES
from shared.utils import setup_logging, create_health_check_response, generate_id, utc_now
from .twin_engine import BioTwinEngine
from .evolution_planner import EvolutionPlanner
from .timeline_engine import TimelineEngine
from .explainability_engine import ExplainabilityEngine
from .evolution_orchestrator import EvolutionOrchestrator


# Setup logging
logger = setup_logging("bio-twin")

# Global components
twin_engine: Optional[BioTwinEngine] = None
evolution_planner: Optional[EvolutionPlanner] = None
timeline_engine: Optional[TimelineEngine] = None
explainability_engine: Optional[ExplainabilityEngine] = None
evolution_orchestrator: Optional[EvolutionOrchestrator] = None
service_clients: Dict[str, httpx.AsyncClient] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global twin_engine, evolution_planner, timeline_engine, explainability_engine, evolution_orchestrator, service_clients
    
    logger.info("Starting BioDigital Twin Engine")
    
    # Initialize service clients
    services = {
        "esm3": os.getenv("ESM3_SERVICE_URL", "http://esm3-service:8000"),
        "proteus": os.getenv("PROTEUS_SERVICE_URL", "http://proteus-service:8000"),
        "aice": os.getenv("AICE_SERVICE_URL", "http://aice-service:8000"),
        "symbiotic": os.getenv("SYMBIOTIC_SERVICE_URL", "http://symbiotic-service:8000")
    }
    
    for name, url in services.items():
        service_clients[name] = httpx.AsyncClient(base_url=url, timeout=60.0)
    
    # Initialize engines
    twin_engine = BioTwinEngine(service_clients)
    evolution_planner = EvolutionPlanner(service_clients)
    timeline_engine = TimelineEngine()
    explainability_engine = ExplainabilityEngine()
    evolution_orchestrator = EvolutionOrchestrator()
    
    logger.info("BioDigital Twin Engine initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down BioDigital Twin Engine")
    for client in service_clients.values():
        await client.aclose()


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI BioDigital Twin Engine",
    description="Orchestrates biological and cognitive modeling for human evolution",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {}
    
    # Check AI services
    for name, client in service_clients.items():
        try:
            response = await client.get("/health", timeout=5.0)
            dependencies[f"service_{name}"] = response.status_code == 200
        except Exception:
            dependencies[f"service_{name}"] = False
    
    return create_health_check_response("bio-twin", dependencies)


@app.get("/snapshot/{user_id}")
async def get_bio_twin_snapshot(user_id: str) -> BioTwinSnapshot:
    """Get user's current bio-digital twin snapshot"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    try:
        snapshot = await twin_engine.get_current_snapshot(user_id)
        return snapshot
    except Exception as e:
        logger.error("Failed to get bio-twin snapshot", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bio-twin snapshot")


@app.post("/analyze")
async def analyze_bio_twin(
    analysis_request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Request comprehensive bio-twin analysis"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    user_id = analysis_request.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    try:
        # Start analysis in background
        analysis_id = generate_id()
        background_tasks.add_task(
            twin_engine.run_comprehensive_analysis,
            user_id,
            analysis_request,
            analysis_id
        )
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Comprehensive bio-twin analysis initiated"
        }
    except Exception as e:
        logger.error("Failed to start bio-twin analysis", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start analysis")


@app.get("/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get status of running analysis"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    status = await twin_engine.get_analysis_status(analysis_id)
    if not status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return status


@app.post("/metrics/update")
async def update_bio_metrics(
    user_id: str,
    metrics: List[BioMetric]
):
    """Update user's biological metrics"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    try:
        await twin_engine.update_bio_metrics(user_id, metrics)
        return {"status": "success", "updated_metrics": len(metrics)}
    except Exception as e:
        logger.error("Failed to update bio metrics", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update metrics")


@app.post("/evolution/plan")
async def create_evolution_plan(
    user_id: str,
    goals: List[EvolutionGoal]
):
    """Create personalized evolution plan"""
    if not evolution_planner:
        raise HTTPException(status_code=500, detail="Evolution planner not initialized")
    
    try:
        plan = await evolution_planner.create_evolution_plan(user_id, goals)
        return plan
    except Exception as e:
        logger.error("Failed to create evolution plan", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create evolution plan")


@app.get("/evolution/plan/{user_id}")
async def get_evolution_plan(user_id: str):
    """Get user's current evolution plan"""
    if not evolution_planner:
        raise HTTPException(status_code=500, detail="Evolution planner not initialized")
    
    try:
        plan = await evolution_planner.get_evolution_plan(user_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Evolution plan not found")
        return plan
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get evolution plan", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve evolution plan")


@app.post("/evolution/feedback")
async def submit_evolution_feedback(
    user_id: str,
    feedback_data: Dict[str, Any]
):
    """Submit feedback on evolution recommendations and update digital twin state"""
    if not evolution_orchestrator:
        raise HTTPException(status_code=500, detail="Evolution orchestrator not initialized")
    
    try:
        # Extract required fields
        intervention_id = feedback_data.get("intervention_id")
        insight_id = feedback_data.get("insight_id")
        context = feedback_data.get("context", {})
        
        if not intervention_id:
            raise HTTPException(status_code=400, detail="intervention_id is required")
        
        # Process evolution feedback through orchestrator
        result = await evolution_orchestrator.process_evolution_feedback(
            user_id=user_id,
            feedback_data=feedback_data,
            intervention_id=intervention_id,
            insight_id=insight_id,
            context=context
        )
        
        return {
            "status": "success",
            "feedback_processed": True,
            "evolution_updated": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process evolution feedback", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@app.post("/simulate/longevity")
async def simulate_longevity_scenarios(
    user_id: str,
    scenarios: List[Dict[str, Any]]
):
    """Simulate longevity scenarios using integrated AI models"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    try:
        results = await twin_engine.simulate_longevity_scenarios(user_id, scenarios)
        return results
    except Exception as e:
        logger.error("Failed to simulate longevity scenarios", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to run simulations")


@app.get("/insights/{user_id}")
async def get_personalized_insights(user_id: str):
    """Get AI-generated personalized insights"""
    if not twin_engine:
        raise HTTPException(status_code=500, detail="Twin engine not initialized")
    
    try:
        insights = await twin_engine.generate_personalized_insights(user_id)
        return insights
    except Exception as e:
        logger.error("Failed to generate insights", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate insights")


# NEW ENHANCED FEATURES

@app.get("/state/{user_id}")
async def get_enhanced_twin_state(user_id: str) -> EnhancedBioTwinSnapshot:
    """Get enhanced bio-twin state with timeline and explainability"""
    if not twin_engine or not timeline_engine:
        raise HTTPException(status_code=500, detail="Engines not initialized")
    
    try:
        # Get basic snapshot
        snapshot = await twin_engine.get_current_snapshot(user_id)
        
        # Get timeline
        timeline = await timeline_engine.get_timeline(user_id, limit=50)
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_name="bio-twin-engine",
            model_version=MODEL_VERSIONS["bio_twin"],
            timestamp=utc_now(),
            processing_time=0.1
        )
        
        # Convert to enhanced snapshot
        enhanced_snapshot = EnhancedBioTwinSnapshot(
            id=snapshot.id,
            user_id=snapshot.user_id,
            state=snapshot.state,
            metrics=snapshot.metrics,
            insights=[],  # Will be populated with TwinInsight objects
            recommendations=snapshot.recommendations,
            confidence_score=snapshot.confidence_score,
            created_at=snapshot.created_at,
            timeline=timeline,
            model_metadata=model_metadata
        )
        
        return enhanced_snapshot
        
    except Exception as e:
        logger.error("Failed to get enhanced twin state", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve enhanced state")


@app.get("/timeline/{user_id}")
async def get_evolution_timeline(
    user_id: str,
    limit: int = 100,
    event_types: Optional[str] = None,
    format: str = "structured"  # structured or json
) -> Dict[str, Any]:
    """Get user's evolution timeline"""
    if not timeline_engine:
        raise HTTPException(status_code=500, detail="Timeline engine not initialized")
    
    try:
        # Parse event types filter
        event_type_list = event_types.split(",") if event_types else None
        
        if format == "json":
            # Return timeline in JSON format for UI
            timeline_json = await timeline_engine.export_timeline_json(user_id)
            return timeline_json
        else:
            # Return structured timeline object
            timeline = await timeline_engine.get_timeline(
                user_id=user_id,
                limit=limit,
                event_types=event_type_list
            )
            return timeline.model_dump()
            
    except Exception as e:
        logger.error("Failed to get evolution timeline", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve timeline")


@app.post("/timeline/{user_id}/event")
async def add_timeline_event(
    user_id: str,
    event_data: Dict[str, Any]
):
    """Add custom event to user's timeline"""
    if not timeline_engine:
        raise HTTPException(status_code=500, detail="Timeline engine not initialized")
    
    try:
        # Extract required fields
        event_type = event_data.get("event_type")
        title = event_data.get("title")
        description = event_data.get("description")
        data = event_data.get("data", {})
        
        if not all([event_type, title, description]):
            raise HTTPException(
                status_code=400, 
                detail="event_type, title, and description are required"
            )
        
        # Add event
        event = await timeline_engine.add_event(
            user_id=user_id,
            event_type=event_type,
            title=title,
            description=description,
            data=data,
            impact_score=event_data.get("impact_score"),
            tags=event_data.get("tags", [])
        )
        
        return {
            "status": "success",
            "event_id": event.id,
            "message": "Timeline event added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add timeline event", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add timeline event")


@app.post("/insights/{user_id}/explain")
async def explain_insight(
    user_id: str,
    insight_request: Dict[str, Any]
):
    """Generate explanation for an AI insight"""
    if not explainability_engine:
        raise HTTPException(status_code=500, detail="Explainability engine not initialized")
    
    try:
        # Extract insight details
        insight_content = insight_request.get("insight_content", "")
        confidence_score = insight_request.get("confidence_score", 0.5)
        contributing_factors = insight_request.get("contributing_factors", [])
        biological_context = insight_request.get("biological_context", "general")
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_name="explainability-engine",
            model_version=MODEL_VERSIONS["bio_twin"],
            timestamp=utc_now()
        )
        
        # Generate explanation
        explanation = explainability_engine.create_explanation(
            insight_content=insight_content,
            confidence_score=confidence_score,
            model_metadata=model_metadata,
            contributing_factors=contributing_factors,
            biological_context=biological_context,
            data_sources=insight_request.get("data_sources", [])
        )
        
        return explanation.model_dump()
        
    except Exception as e:
        logger.error("Failed to generate insight explanation", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate explanation")


@app.get("/timeline/{user_id}/cleanup")
async def cleanup_timeline(
    user_id: str,
    retention_days: int = 365
):
    """Clean up old timeline events"""
    if not timeline_engine:
        raise HTTPException(status_code=500, detail="Timeline engine not initialized")
    
    try:
        await timeline_engine.cleanup_old_events(user_id, retention_days)
        return {
            "status": "success",
            "message": f"Timeline cleaned up with {retention_days} days retention"
        }
        
    except Exception as e:
        logger.error("Failed to cleanup timeline", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cleanup timeline")


# EVOLUTION ORCHESTRATOR ENDPOINTS

@app.get("/evolution/status/{user_id}")
async def get_evolution_status(user_id: str):
    """Get comprehensive evolution status for user"""
    if not evolution_orchestrator:
        raise HTTPException(status_code=500, detail="Evolution orchestrator not initialized")
    
    try:
        status = await evolution_orchestrator.get_evolution_status(user_id)
        return status
    except Exception as e:
        logger.error("Failed to get evolution status", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get evolution status")


@app.get("/evolution/graph/{user_id}")
async def get_evolution_graph(user_id: str):
    """Get evolution graph data for visualization"""
    if not evolution_orchestrator:
        raise HTTPException(status_code=500, detail="Evolution orchestrator not initialized")
    
    try:
        # Get evolution graph from Redis
        graph_key = f"evolution_graph:{user_id}"
        graph_data = evolution_orchestrator.redis_client.get(graph_key)
        
        if graph_data:
            import json
            graph = json.loads(graph_data)
            return {
                "user_id": user_id,
                "graph_data": graph,
                "timeline_length": len(graph.get("timeline", [])),
                "nodes_count": len(graph.get("nodes", [])),
                "edges_count": len(graph.get("edges", []))
            }
        else:
            return {
                "user_id": user_id,
                "graph_data": {"nodes": [], "edges": [], "timeline": []},
                "timeline_length": 0,
                "nodes_count": 0,
                "edges_count": 0
            }
            
    except Exception as e:
        logger.error("Failed to get evolution graph", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get evolution graph")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
