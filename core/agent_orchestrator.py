"""
EvoHuman.AI Agent Orchestrator
Central coordinator for cognitive, biological and feedback systems
"""
import sys
import os
from typing import Dict, List, Optional, Union
import asyncio
import httpx
import json
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field
import redis
import structlog

# Configure logging
logger = structlog.get_logger("agent_orchestrator")

class EvolutionRequest(BaseModel):
    user_id: str
    goal: str
    focus_area: List[str]
    feedback: Optional[str] = None
    context: Optional[Dict[str, any]] = Field(default_factory=dict)

class SimulationInsights(BaseModel):
    cell_healing_potential: float
    brain_oxidative_resilience: float
    metabolic_efficiency: float = Field(default=0.0)
    telomere_stability: float = Field(default=0.0)
    immune_resilience: float = Field(default=0.0)

class EvolutionResponse(BaseModel):
    longevity_score: float
    daily_recommendations: List[str]
    simulation_insights: SimulationInsights
    wisdom_quote: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AgentOrchestrator:
    def __init__(self):
        """Initialize the orchestrator with service connections"""
        self.redis = redis.Redis(
            host='redis',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Initialize service clients
        self.clients = {
            "aice": httpx.AsyncClient(base_url="http://aice-service:8000"),
            "proteus": httpx.AsyncClient(base_url="http://proteus-service:8000"),
            "esm3": httpx.AsyncClient(base_url="http://esm3-service:8000"),
            "symbiotic": httpx.AsyncClient(base_url="http://symbiotic-service:8000"),
            "exostack": httpx.AsyncClient(base_url="http://exostack-service:8000")
        }

    async def close(self):
        """Cleanup connections"""
        for client in self.clients.values():
            await client.aclose()

    async def get_cached_insights(self, user_id: str) -> Optional[Dict]:
        """Retrieve cached insights if available and fresh"""
        cache_key = f"insights:{user_id}"
        cached = self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            # Check if cache is less than 24 hours old
            cached_time = datetime.fromisoformat(data["updated_at"])
            if (datetime.utcnow() - cached_time).total_seconds() < 86400:
                return data
        return None

    async def store_cached_insights(self, user_id: str, insights: Dict):
        """Store insights in cache"""
        cache_key = f"insights:{user_id}"
        insights["updated_at"] = datetime.utcnow().isoformat()
        self.redis.setex(cache_key, 86400, json.dumps(insights))

    async def get_cognitive_insights(self, user_id: str, goal: str) -> Dict:
        """Get cognitive insights from AiCE"""
        try:
            response = await self.clients["aice"].post("/analyze", json={
                "user_id": user_id,
                "goal": goal,
                "insight_type": "longevity_wisdom"
            })
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("AiCE insight error", error=str(e))
            return {
                "wisdom_quote": "Wisdom comes from listening to your body's signals.",
                "cognitive_score": 0.65
            }

    async def get_biological_simulation(
        self,
        user_id: str,
        focus_areas: List[str]
    ) -> Dict:
        """Run biological simulation using Proteus + ESM3"""
        try:
            # Get protein analysis from ESM3
            esm3_response = await self.clients["esm3"].post("/analyze", json={
                "user_id": user_id,
                "analysis_type": "longevity_markers",
                "focus_areas": focus_areas
            })
            esm3_response.raise_for_status()
            esm3_data = esm3_response.json()

            # Run Proteus simulation
            proteus_response = await self.clients["proteus"].post("/simulate", json={
                "user_id": user_id,
                "esm3_markers": esm3_data["markers"],
                "focus_areas": focus_areas,
                "simulation_type": "longevity"
            })
            proteus_response.raise_for_status()
            return proteus_response.json()

        except Exception as e:
            logger.error("Biological simulation error", error=str(e))
            return {
                "cell_healing_potential": 0.7,
                "brain_oxidative_resilience": 0.65,
                "simulation_score": 0.75
            }

    async def get_feedback_insights(
        self,
        user_id: str,
        feedback: Optional[str]
    ) -> Dict:
        """Get personalized insights from SymbioticAIS"""
        try:
            response = await self.clients["symbiotic"].post("/symbiotic/analyze", json={
                "user_id": user_id,
                "feedback": feedback,
                "analysis_type": "longevity_patterns"
            })
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Feedback analysis error", error=str(e))
            return {
                "feedback_score": 0.7,
                "recommendations": [
                    "Listen to your body's response to interventions",
                    "Start with small changes and observe effects"
                ]
            }

    def calculate_longevity_score(self, insights: Dict[str, float]) -> float:
        """Calculate overall longevity score from various insights"""
        weights = {
            "cognitive_score": 0.3,
            "simulation_score": 0.4,
            "feedback_score": 0.3
        }
        
        score = sum(
            weights.get(k, 0) * v 
            for k, v in insights.items() 
            if k in weights
        )
        return round(score * 100)

    async def orchestrate_evolution(
        self,
        request: EvolutionRequest
    ) -> EvolutionResponse:
        """Main orchestration method to coordinate all services"""
        # Check cache first
        cached = await self.get_cached_insights(request.user_id)
        if cached:
            return EvolutionResponse(**cached)

        # Gather insights from all services concurrently
        cognitive_task = self.get_cognitive_insights(
            request.user_id,
            request.goal
        )
        biological_task = self.get_biological_simulation(
            request.user_id,
            request.focus_area
        )
        feedback_task = self.get_feedback_insights(
            request.user_id,
            request.feedback
        )

        cognitive, biological, feedback = await asyncio.gather(
            cognitive_task,
            biological_task,
            feedback_task
        )

        # Combine all insights
        longevity_score = self.calculate_longevity_score({
            "cognitive_score": cognitive.get("cognitive_score", 0.7),
            "simulation_score": biological.get("simulation_score", 0.7),
            "feedback_score": feedback.get("feedback_score", 0.7)
        })

        # Combine recommendations
        recommendations = (
            feedback.get("recommendations", []) +
            ["Optimize sleep schedule for cellular repair"] +
            ["Practice stress management for telomere protection"]
        )

        response = EvolutionResponse(
            longevity_score=longevity_score,
            daily_recommendations=recommendations[:5],  # Top 5 recommendations
            simulation_insights=SimulationInsights(
                cell_healing_potential=biological.get("cell_healing_potential", 0.7),
                brain_oxidative_resilience=biological.get("brain_oxidative_resilience", 0.65),
                metabolic_efficiency=biological.get("metabolic_efficiency", 0.72),
                telomere_stability=biological.get("telomere_stability", 0.68),
                immune_resilience=biological.get("immune_resilience", 0.75)
            ),
            wisdom_quote=cognitive.get("wisdom_quote", "Your body's wisdom guides your evolution.")
        )

        # Cache the insights
        await self.store_cached_insights(request.user_id, response.dict())

        return response

# Global orchestrator instance
orchestrator: Optional[AgentOrchestrator] = None

async def get_orchestrator() -> AgentOrchestrator:
    """Get or create the global orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator()
    return orchestrator