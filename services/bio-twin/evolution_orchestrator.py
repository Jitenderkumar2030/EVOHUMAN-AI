"""
Evolution Orchestrator for EvoHuman.AI
Connects user feedback into the digital twin evolution state and coordinates
parameter adjustments across AiCE and Proteus services
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import asyncio
import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import structlog
import redis

from shared.models import (
    BioTwinSnapshot, EvolutionGoal, FeedbackLoop, 
    BioMetric, BioMetricType, EvolutionState
)
from shared.utils import setup_logging, generate_id, utc_now
from shared.constants import FEEDBACK_QUALITY


class EvolutionOrchestrator:
    """Orchestrates evolution feedback loops and parameter coordination"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.logger = setup_logging("evolution_orchestrator")
        self.redis_client = redis_client or redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        
        # Service clients
        self.service_clients = {
            "aice": httpx.AsyncClient(
                base_url=os.getenv("AICE_SERVICE_URL", "http://aice-service:8000"),
                timeout=30.0
            ),
            "proteus": httpx.AsyncClient(
                base_url=os.getenv("PROTEUS_SERVICE_URL", "http://proteus-service:8000"),
                timeout=30.0
            ),
            "symbiotic": httpx.AsyncClient(
                base_url=os.getenv("SYMBIOTIC_SERVICE_URL", "http://symbiotic-service:8000"),
                timeout=30.0
            ),
            "esm3": httpx.AsyncClient(
                base_url=os.getenv("ESM3_SERVICE_URL", "http://esm3-service:8000"),
                timeout=30.0
            )
        }
        
        # Evolution parameters
        self.evolution_weights = {
            "cognitive": 0.3,
            "physiological": 0.3,
            "genetic": 0.2,
            "behavioral": 0.2
        }
        
        # Feedback impact factors
        self.feedback_impact_multipliers = {
            "high_quality": 1.0,
            "medium_quality": 0.7,
            "low_quality": 0.3
        }
    
    async def process_evolution_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        intervention_id: str,
        insight_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and update evolution state
        
        Args:
            user_id: User identifier
            feedback_data: Feedback content and ratings
            intervention_id: ID of intervention being evaluated
            insight_id: Optional ID of insight being evaluated
            context: Additional context for feedback processing
            
        Returns:
            Dict with evolution updates and parameter adjustments
        """
        try:
            self.logger.info(
                "Processing evolution feedback",
                user_id=user_id,
                intervention_id=intervention_id,
                insight_id=insight_id
            )
            
            # 1. Get current bio-twin snapshot
            current_snapshot = await self._get_current_biotwin_snapshot(user_id)
            
            # 2. Score feedback quality via SymbioticAIS
            feedback_quality = await self._score_feedback_quality(
                user_id, feedback_data, context
            )
            
            # 3. Calculate evolution delta
            evolution_delta = await self._calculate_evolution_delta(
                user_id, feedback_data, feedback_quality, current_snapshot
            )
            
            # 4. Update bio-twin evolution state
            updated_snapshot = await self._update_biotwin_evolution_state(
                user_id, current_snapshot, evolution_delta, feedback_data
            )
            
            # 5. Calculate parameter adjustments
            parameter_adjustments = await self._calculate_parameter_adjustments(
                user_id, evolution_delta, feedback_quality, updated_snapshot
            )
            
            # 6. Apply parameter adjustments to services
            adjustment_results = await self._apply_parameter_adjustments(
                user_id, parameter_adjustments
            )
            
            # 7. Store feedback loop record
            feedback_loop = await self._store_feedback_loop(
                user_id, feedback_data, evolution_delta, 
                parameter_adjustments, intervention_id, insight_id
            )
            
            # 8. Update evolution graph
            await self._update_evolution_graph(
                user_id, evolution_delta, feedback_loop.id
            )
            
            result = {
                "feedback_loop_id": feedback_loop.id,
                "evolution_delta": evolution_delta,
                "parameter_adjustments": parameter_adjustments,
                "adjustment_results": adjustment_results,
                "updated_bio_twin": {
                    "state": updated_snapshot.state.value,
                    "confidence_score": updated_snapshot.confidence_score,
                    "evolution_progress": self._calculate_evolution_progress(updated_snapshot)
                },
                "quality_impact": {
                    "feedback_quality_score": feedback_quality["score"],
                    "impact_multiplier": self._get_quality_multiplier(feedback_quality["score"]),
                    "learning_weight": feedback_quality["score"] * 0.8 + 0.2
                }
            }
            
            self.logger.info(
                "Evolution feedback processed successfully",
                user_id=user_id,
                feedback_loop_id=feedback_loop.id,
                evolution_delta_magnitude=np.linalg.norm(list(evolution_delta.values())),
                parameter_adjustments_count=len(parameter_adjustments)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Failed to process evolution feedback",
                user_id=user_id,
                intervention_id=intervention_id,
                error=str(e)
            )
            raise
    
    async def _get_current_biotwin_snapshot(self, user_id: str) -> BioTwinSnapshot:
        """Get user's current bio-twin snapshot"""
        
        # Try to get from cache first
        snapshot_key = f"biotwin_snapshot:{user_id}:current"
        cached_snapshot = self.redis_client.get(snapshot_key)
        
        if cached_snapshot:
            snapshot_data = json.loads(cached_snapshot)
            return BioTwinSnapshot(**snapshot_data)
        
        # Create initial snapshot if none exists
        initial_metrics = [
            BioMetric(
                type=BioMetricType.COGNITIVE,
                name="baseline_cognitive",
                value=0.5,
                unit="score",
                timestamp=utc_now(),
                confidence=0.7,
                source="initial_assessment"
            ),
            BioMetric(
                type=BioMetricType.PHYSIOLOGICAL,
                name="baseline_physiological",
                value=0.5,
                unit="score",
                timestamp=utc_now(),
                confidence=0.7,
                source="initial_assessment"
            ),
            BioMetric(
                type=BioMetricType.BEHAVIORAL,
                name="baseline_behavioral",
                value=0.5,
                unit="score",
                timestamp=utc_now(),
                confidence=0.7,
                source="initial_assessment"
            )
        ]
        
        initial_snapshot = BioTwinSnapshot(
            id=generate_id(),
            user_id=user_id,
            state=EvolutionState.BASELINE,
            metrics=initial_metrics,
            insights={
                "evolution_phase": "initialization",
                "learning_status": "collecting_baseline_data",
                "adaptation_rate": 0.1
            },
            recommendations=[
                "Continue providing feedback to establish baseline patterns",
                "Engage with recommendations to enable personalization"
            ],
            confidence_score=0.5,
            created_at=utc_now()
        )
        
        # Cache the initial snapshot
        self.redis_client.setex(
            snapshot_key,
            3600,  # 1 hour cache
            json.dumps(initial_snapshot.model_dump(), default=str)
        )
        
        return initial_snapshot
    
    async def _score_feedback_quality(
        self, 
        user_id: str, 
        feedback_data: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Score feedback quality using SymbioticAIS service"""
        
        try:
            response = await self.service_clients["symbiotic"].post(
                "/feedback/enhanced",
                params={
                    "user_id": user_id,
                    "feedback_type": feedback_data.get("type", "mixed"),
                    "context": json.dumps(context) if context else None
                },
                json={"feedback_content": feedback_data}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "score": result["quality_score"]["score"],
                    "consistency_score": result["quality_score"]["consistency_score"],
                    "relevance_score": result["quality_score"]["relevance_score"],
                    "sentiment_score": result["quality_score"]["sentiment_score"],
                    "reliability_factors": result["quality_score"]["reliability_factors"]
                }
            else:
                self.logger.warning(
                    "Failed to score feedback quality",
                    user_id=user_id,
                    status_code=response.status_code
                )
                return {"score": 0.5, "consistency_score": 0.5, "relevance_score": 0.5}
                
        except Exception as e:
            self.logger.error(
                "Error scoring feedback quality",
                user_id=user_id,
                error=str(e)
            )
            return {"score": 0.5, "consistency_score": 0.5, "relevance_score": 0.5}
    
    async def _calculate_evolution_delta(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        feedback_quality: Dict[str, Any],
        current_snapshot: BioTwinSnapshot
    ) -> Dict[str, float]:
        """Calculate evolution delta based on feedback"""
        
        # Base delta calculation
        delta = {
            "cognitive": 0.0,
            "physiological": 0.0,
            "genetic": 0.0,
            "behavioral": 0.0
        }
        
        # Extract feedback signals
        if "effectiveness_rating" in feedback_data:
            effectiveness = feedback_data["effectiveness_rating"]
            base_delta = (effectiveness - 3.0) / 5.0  # Normalize to -0.4 to 0.4 range
            
            # Apply to relevant domains based on intervention type
            intervention_type = feedback_data.get("intervention_type", "general")
            
            if intervention_type in ["cognitive", "mental", "focus", "memory"]:
                delta["cognitive"] += base_delta * 0.8
                delta["behavioral"] += base_delta * 0.2
            elif intervention_type in ["exercise", "nutrition", "sleep"]:
                delta["physiological"] += base_delta * 0.7
                delta["behavioral"] += base_delta * 0.3
            elif intervention_type in ["meditation", "stress", "mindfulness"]:
                delta["cognitive"] += base_delta * 0.5
                delta["behavioral"] += base_delta * 0.5
            else:
                # General intervention
                for domain in delta:
                    delta[domain] += base_delta * 0.25
        
        # Apply feedback quality multiplier
        quality_multiplier = self._get_quality_multiplier(feedback_quality["score"])
        for domain in delta:
            delta[domain] *= quality_multiplier
        
        # Apply learning rate based on current evolution state
        learning_rate = self._get_learning_rate(current_snapshot.state)
        for domain in delta:
            delta[domain] *= learning_rate
        
        # Sentiment-based adjustments
        sentiment_score = feedback_quality.get("sentiment_score", 0.0)
        if abs(sentiment_score) > 0.3:  # Strong sentiment
            sentiment_bonus = abs(sentiment_score) * 0.1
            for domain in delta:
                if delta[domain] > 0:
                    delta[domain] += sentiment_bonus
                elif delta[domain] < 0:
                    delta[domain] -= sentiment_bonus
        
        # Clip deltas to reasonable bounds
        for domain in delta:
            delta[domain] = max(-0.2, min(0.2, delta[domain]))
        
        return delta
    
    async def _update_biotwin_evolution_state(
        self,
        user_id: str,
        current_snapshot: BioTwinSnapshot,
        evolution_delta: Dict[str, float],
        feedback_data: Dict[str, Any]
    ) -> BioTwinSnapshot:
        """Update bio-twin snapshot with evolution changes"""
        
        # Update metrics with evolution delta
        updated_metrics = []
        for metric in current_snapshot.metrics:
            domain = metric.type.value.lower()
            if domain in evolution_delta:
                new_value = metric.value + evolution_delta[domain]
                new_value = max(0.0, min(1.0, new_value))  # Clamp to [0, 1]
                
                updated_metric = BioMetric(
                    type=metric.type,
                    name=metric.name,
                    value=new_value,
                    unit=metric.unit,
                    timestamp=utc_now(),
                    confidence=min(1.0, metric.confidence + 0.05),  # Slightly increase confidence
                    source="feedback_evolution"
                )
                updated_metrics.append(updated_metric)
            else:
                updated_metrics.append(metric)
        
        # Update evolution state
        new_state = self._calculate_new_evolution_state(
            current_snapshot.state, evolution_delta, updated_metrics
        )
        
        # Update insights
        updated_insights = current_snapshot.insights.copy()
        updated_insights.update({
            "last_evolution_delta": evolution_delta,
            "feedback_incorporation_count": updated_insights.get("feedback_incorporation_count", 0) + 1,
            "last_feedback_timestamp": utc_now().isoformat(),
            "evolution_momentum": self._calculate_evolution_momentum(evolution_delta),
            "adaptation_rate": self._update_adaptation_rate(
                updated_insights.get("adaptation_rate", 0.1), evolution_delta
            )
        })
        
        # Generate new recommendations based on evolution
        new_recommendations = await self._generate_evolution_recommendations(
            user_id, evolution_delta, updated_metrics, feedback_data
        )
        
        # Calculate confidence score
        new_confidence = self._calculate_updated_confidence(
            current_snapshot.confidence_score,
            evolution_delta,
            updated_insights.get("feedback_incorporation_count", 1)
        )
        
        updated_snapshot = BioTwinSnapshot(
            id=generate_id(),
            user_id=user_id,
            state=new_state,
            metrics=updated_metrics,
            insights=updated_insights,
            recommendations=new_recommendations,
            confidence_score=new_confidence,
            created_at=utc_now()
        )
        
        # Cache updated snapshot
        snapshot_key = f"biotwin_snapshot:{user_id}:current"
        self.redis_client.setex(
            snapshot_key,
            3600,  # 1 hour cache
            json.dumps(updated_snapshot.model_dump(), default=str)
        )
        
        return updated_snapshot
    
    async def _calculate_parameter_adjustments(
        self,
        user_id: str,
        evolution_delta: Dict[str, float],
        feedback_quality: Dict[str, Any],
        updated_snapshot: BioTwinSnapshot
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate parameter adjustments for AiCE and Proteus services"""
        
        adjustments = {
            "aice": {},
            "proteus": {},
            "esm3": {}
        }
        
        # AiCE cognitive parameter adjustments
        cognitive_delta = evolution_delta.get("cognitive", 0.0)
        if abs(cognitive_delta) > 0.05:
            adjustments["aice"] = {
                "learning_rate": max(0.001, min(0.1, 0.01 + cognitive_delta * 0.1)),
                "focus_enhancement_weight": max(0.0, min(1.0, 0.5 + cognitive_delta)),
                "memory_consolidation_strength": max(0.0, min(1.0, 0.6 + cognitive_delta * 0.5)),
                "cognitive_load_threshold": max(0.3, min(0.9, 0.7 - abs(cognitive_delta))),
                "adaptation_sensitivity": feedback_quality["score"] * 0.8 + 0.2
            }
        
        # Proteus physiological parameter adjustments
        physiological_delta = evolution_delta.get("physiological", 0.0)
        if abs(physiological_delta) > 0.05:
            adjustments["proteus"] = {
                "regeneration_rate": max(0.1, min(2.0, 1.0 + physiological_delta * 2.0)),
                "cellular_adaptation_speed": max(0.5, min(1.5, 1.0 + physiological_delta)),
                "metabolic_efficiency_target": max(0.6, min(1.0, 0.8 + physiological_delta * 0.4)),
                "tissue_optimization_priority": "high" if physiological_delta > 0.1 else "normal",
                "recovery_enhancement_factor": max(1.0, min(3.0, 1.5 + physiological_delta * 2.0))
            }
        
        # ESM3 genetic/protein parameter adjustments
        genetic_delta = evolution_delta.get("genetic", 0.0)
        if abs(genetic_delta) > 0.03:  # More conservative threshold for genetic changes
            adjustments["esm3"] = {
                "mutation_analysis_sensitivity": max(0.3, min(0.9, 0.6 + genetic_delta)),
                "evolution_pathway_exploration": max(0.1, min(0.8, 0.4 + abs(genetic_delta))),
                "protein_stability_weight": max(0.5, min(1.0, 0.8 - abs(genetic_delta) * 0.3)),
                "optimization_target_intensity": feedback_quality["score"] * 0.6 + 0.4
            }
        
        # Global parameters based on overall evolution momentum
        evolution_momentum = np.linalg.norm(list(evolution_delta.values()))
        if evolution_momentum > 0.1:
            for service in adjustments:
                if adjustments[service]:
                    adjustments[service]["global_adaptation_rate"] = min(1.0, evolution_momentum * 2.0)
                    adjustments[service]["feedback_integration_weight"] = feedback_quality["score"]
        
        return adjustments
    
    async def _apply_parameter_adjustments(
        self,
        user_id: str,
        parameter_adjustments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply parameter adjustments to respective services"""
        
        results = {}
        
        for service_name, adjustments in parameter_adjustments.items():
            if not adjustments:
                continue
                
            try:
                if service_name == "aice":
                    response = await self.service_clients["aice"].post(
                        f"/parameters/update/{user_id}",
                        json=adjustments
                    )
                elif service_name == "proteus":
                    response = await self.service_clients["proteus"].post(
                        f"/parameters/adjust/{user_id}",
                        json=adjustments
                    )
                elif service_name == "esm3":
                    response = await self.service_clients["esm3"].post(
                        f"/model/adjust_parameters",
                        json={"user_id": user_id, "parameters": adjustments}
                    )
                
                if response.status_code == 200:
                    results[service_name] = {
                        "status": "success",
                        "applied_parameters": adjustments,
                        "service_response": response.json()
                    }
                else:
                    results[service_name] = {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}",
                        "attempted_parameters": adjustments
                    }
                    
            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "attempted_parameters": adjustments
                }
                
                self.logger.error(
                    f"Failed to apply parameters to {service_name}",
                    user_id=user_id,
                    service=service_name,
                    error=str(e)
                )
        
        return results
    
    async def _store_feedback_loop(
        self,
        user_id: str,
        feedback_data: Dict[str, Any],
        evolution_delta: Dict[str, float],
        parameter_adjustments: Dict[str, Dict[str, Any]],
        intervention_id: str,
        insight_id: Optional[str]
    ) -> FeedbackLoop:
        """Store feedback loop record"""
        
        feedback_loop = FeedbackLoop(
            id=generate_id(),
            user_id=user_id,
            loop_type="evolution_feedback",
            input_data={
                "feedback": feedback_data,
                "intervention_id": intervention_id,
                "insight_id": insight_id
            },
            human_feedback=feedback_data,
            ai_response={
                "evolution_delta": evolution_delta,
                "parameter_adjustments": parameter_adjustments
            },
            improvement_score=np.linalg.norm(list(evolution_delta.values())),
            created_at=utc_now()
        )
        
        # Store in Redis
        loop_key = f"feedback_loop:{user_id}:{feedback_loop.id}"
        self.redis_client.setex(
            loop_key,
            86400 * 30,  # 30 days
            json.dumps(feedback_loop.model_dump(), default=str)
        )
        
        return feedback_loop
    
    async def _update_evolution_graph(
        self,
        user_id: str,
        evolution_delta: Dict[str, float],
        feedback_loop_id: str
    ):
        """Update the evolution graph with new data point"""
        
        graph_key = f"evolution_graph:{user_id}"
        
        # Get current graph data
        graph_data = self.redis_client.get(graph_key)
        if graph_data:
            graph = json.loads(graph_data)
        else:
            graph = {"nodes": [], "edges": [], "timeline": []}
        
        # Add new timeline entry
        timeline_entry = {
            "timestamp": utc_now().isoformat(),
            "feedback_loop_id": feedback_loop_id,
            "evolution_delta": evolution_delta,
            "delta_magnitude": np.linalg.norm(list(evolution_delta.values())),
            "dominant_domain": max(evolution_delta.items(), key=lambda x: abs(x[1]))[0]
        }
        
        graph["timeline"].append(timeline_entry)
        
        # Keep last 1000 entries
        graph["timeline"] = graph["timeline"][-1000:]
        
        # Update evolution graph
        self.redis_client.setex(
            graph_key,
            86400 * 365,  # 1 year
            json.dumps(graph)
        )
    
    def _get_quality_multiplier(self, quality_score: float) -> float:
        """Get feedback quality multiplier"""
        if quality_score >= FEEDBACK_QUALITY.get("HIGH_THRESHOLD", 0.8):
            return self.feedback_impact_multipliers["high_quality"]
        elif quality_score >= FEEDBACK_QUALITY.get("MEDIUM_THRESHOLD", 0.5):
            return self.feedback_impact_multipliers["medium_quality"]
        else:
            return self.feedback_impact_multipliers["low_quality"]
    
    def _get_learning_rate(self, evolution_state: EvolutionState) -> float:
        """Get learning rate based on evolution state"""
        rates = {
            EvolutionState.BASELINE: 1.0,
            EvolutionState.ANALYZING: 0.8,
            EvolutionState.EVOLVING: 1.2,
            EvolutionState.OPTIMIZED: 0.6
        }
        return rates.get(evolution_state, 1.0)
    
    def _calculate_new_evolution_state(
        self,
        current_state: EvolutionState,
        evolution_delta: Dict[str, float],
        updated_metrics: List[BioMetric]
    ) -> EvolutionState:
        """Calculate new evolution state based on progress"""
        
        delta_magnitude = np.linalg.norm(list(evolution_delta.values()))
        avg_metric_value = np.mean([m.value for m in updated_metrics])
        
        if current_state == EvolutionState.BASELINE:
            if delta_magnitude > 0.1:
                return EvolutionState.EVOLVING
            else:
                return EvolutionState.ANALYZING
                
        elif current_state == EvolutionState.ANALYZING:
            if delta_magnitude > 0.15:
                return EvolutionState.EVOLVING
            else:
                return EvolutionState.ANALYZING
                
        elif current_state == EvolutionState.EVOLVING:
            if avg_metric_value > 0.8 and delta_magnitude < 0.05:
                return EvolutionState.OPTIMIZED
            else:
                return EvolutionState.EVOLVING
                
        elif current_state == EvolutionState.OPTIMIZED:
            if delta_magnitude > 0.1:
                return EvolutionState.EVOLVING
            else:
                return EvolutionState.OPTIMIZED
        
        return current_state
    
    def _calculate_evolution_momentum(self, evolution_delta: Dict[str, float]) -> float:
        """Calculate evolution momentum"""
        return float(np.linalg.norm(list(evolution_delta.values())))
    
    def _update_adaptation_rate(
        self, 
        current_rate: float, 
        evolution_delta: Dict[str, float]
    ) -> float:
        """Update adaptation rate based on evolution delta"""
        delta_magnitude = np.linalg.norm(list(evolution_delta.values()))
        
        # Increase adaptation rate if evolution is happening
        if delta_magnitude > 0.1:
            new_rate = min(1.0, current_rate * 1.1)
        else:
            new_rate = max(0.01, current_rate * 0.95)
        
        return new_rate
    
    async def _generate_evolution_recommendations(
        self,
        user_id: str,
        evolution_delta: Dict[str, float],
        updated_metrics: List[BioMetric],
        feedback_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on evolution progress"""
        
        recommendations = []
        
        # Analyze dominant evolution direction
        dominant_domain = max(evolution_delta.items(), key=lambda x: abs(x[1]))
        domain_name, delta_value = dominant_domain
        
        if delta_value > 0.1:
            recommendations.append(
                f"Positive {domain_name} evolution detected. Continue current interventions."
            )
        elif delta_value < -0.1:
            recommendations.append(
                f"Consider adjusting {domain_name} interventions for better outcomes."
            )
        
        # Metric-based recommendations
        for metric in updated_metrics:
            if metric.value > 0.8:
                recommendations.append(
                    f"Excellent progress in {metric.name}. Maintain current approach."
                )
            elif metric.value < 0.3:
                recommendations.append(
                    f"Focus needed on {metric.name} improvement through targeted interventions."
                )
        
        # Feedback-based recommendations
        if "effectiveness_rating" in feedback_data:
            rating = feedback_data["effectiveness_rating"]
            if rating >= 4:
                recommendations.append("High satisfaction detected. Consider similar interventions.")
            elif rating <= 2:
                recommendations.append("Low satisfaction. Exploring alternative approaches.")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_updated_confidence(
        self,
        current_confidence: float,
        evolution_delta: Dict[str, float],
        feedback_count: int
    ) -> float:
        """Calculate updated confidence score"""
        
        # Base confidence increase from feedback incorporation
        confidence_boost = min(0.1, feedback_count * 0.01)
        
        # Confidence adjustment based on evolution consistency
        delta_magnitude = np.linalg.norm(list(evolution_delta.values()))
        consistency_factor = 1.0 - min(0.2, delta_magnitude)  # Large changes reduce confidence slightly
        
        new_confidence = (current_confidence + confidence_boost) * consistency_factor
        return max(0.1, min(1.0, new_confidence))
    
    def _calculate_evolution_progress(self, snapshot: BioTwinSnapshot) -> Dict[str, float]:
        """Calculate evolution progress across domains"""
        
        progress = {}
        for metric in snapshot.metrics:
            domain = metric.type.value.lower()
            if domain not in progress:
                progress[domain] = []
            progress[domain].append(metric.value)
        
        # Average progress per domain
        return {domain: np.mean(values) for domain, values in progress.items()}
    
    async def get_evolution_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive evolution status for user"""
        
        try:
            # Get current snapshot
            current_snapshot = await self._get_current_biotwin_snapshot(user_id)
            
            # Get evolution graph
            graph_key = f"evolution_graph:{user_id}"
            graph_data = self.redis_client.get(graph_key)
            
            if graph_data:
                graph = json.loads(graph_data)
                timeline = graph.get("timeline", [])
            else:
                timeline = []
            
            # Calculate evolution metrics
            progress = self._calculate_evolution_progress(current_snapshot)
            
            # Recent evolution trend
            recent_deltas = []
            if len(timeline) >= 5:
                for entry in timeline[-5:]:
                    recent_deltas.append(entry.get("delta_magnitude", 0.0))
            
            evolution_trend = "stable"
            if recent_deltas:
                trend_slope = np.polyfit(range(len(recent_deltas)), recent_deltas, 1)[0]
                if trend_slope > 0.02:
                    evolution_trend = "accelerating"
                elif trend_slope < -0.02:
                    evolution_trend = "decelerating"
            
            return {
                "user_id": user_id,
                "current_state": current_snapshot.state.value,
                "confidence_score": current_snapshot.confidence_score,
                "evolution_progress": progress,
                "evolution_trend": evolution_trend,
                "total_feedback_loops": len(timeline),
                "last_evolution": timeline[-1] if timeline else None,
                "recommendations": current_snapshot.recommendations,
                "insights": current_snapshot.insights
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to get evolution status",
                user_id=user_id,
                error=str(e)
            )
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        for client in self.service_clients.values():
            await client.aclose()
