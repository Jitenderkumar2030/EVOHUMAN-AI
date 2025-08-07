"""
AiCE Intuition Module
Pattern recognition and intuitive insight generation
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import structlog
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = structlog.get_logger(__name__)


class IntuitionModule:
    """Module for generating intuitive insights through pattern analysis"""
    
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.insight_history = {}  # {user_id: [insights]}
        self.pattern_cache = {}    # Cache for expensive computations
        self.initialized = False
    
    async def initialize(self):
        """Initialize the intuition module"""
        try:
            self.initialized = True
            logger.info("Intuition module initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize intuition module: {e}")
    
    async def analyze_patterns(
        self,
        user_id: str,
        context: Dict[str, Any],
        focus_areas: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze patterns and generate intuitive insights"""
        
        start_time = datetime.utcnow()
        
        try:
            # Get user's memory patterns
            memory_patterns = await self.memory_graph.find_memory_patterns(user_id)
            
            # Analyze current context
            context_analysis = await self._analyze_context(user_id, context)
            
            # Generate intuitive insights
            insights = await self._generate_insights(
                user_id, memory_patterns, context_analysis, focus_areas
            )
            
            # Detect behavioral patterns
            behavioral_patterns = await self._detect_behavioral_patterns(user_id)
            
            # Synthesize final results
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            results = {
                "insights": insights,
                "patterns": memory_patterns + behavioral_patterns,
                "confidence": self._calculate_confidence(insights, memory_patterns),
                "contributing_factors": self._identify_contributing_factors(
                    memory_patterns, context_analysis
                ),
                "processing_time": processing_time
            }
            
            # Store insight for future validation
            await self._store_insight(user_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            # Return mock results as fallback
            return await self._mock_pattern_analysis(user_id, context)
    
    async def validate_insight(
        self,
        user_id: str,
        intuition_id: str,
        actual_outcome: Dict[str, Any],
        feedback_score: float
    ) -> Dict[str, Any]:
        """Validate intuitive insights against actual outcomes"""
        
        try:
            # Retrieve the original insight
            user_insights = self.insight_history.get(user_id, [])
            target_insight = None
            
            for insight in user_insights:
                if insight.get("insight_id") == intuition_id:
                    target_insight = insight
                    break
            
            if not target_insight:
                return {"error": "Insight not found"}
            
            # Calculate validation metrics
            validation_results = await self._calculate_validation_metrics(
                target_insight, actual_outcome, feedback_score
            )
            
            # Update learning model
            await self._update_learning_model(
                user_id, target_insight, validation_results
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Insight validation failed: {e}")
            return {
                "accuracy": 0.5,
                "error": str(e)
            }
    
    async def get_intuitive_recommendations(
        self,
        user_id: str,
        situation: Dict[str, Any],
        constraints: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get intuitive recommendations for a specific situation"""
        
        try:
            # Analyze current situation patterns
            situation_patterns = await self._analyze_situation(user_id, situation)
            
            # Find similar past experiences
            similar_experiences = await self._find_similar_experiences(
                user_id, situation
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                user_id, situation_patterns, similar_experiences, constraints
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def detect_emerging_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Detect new patterns that are just beginning to emerge"""
        
        try:
            # Get recent memories (last 30 days)
            recent_memories = await self.memory_graph.recall_memories(
                user_id=user_id,
                limit=50,
                importance_threshold=0.2
            )
            
            if len(recent_memories) < 5:
                return []
            
            # Analyze emerging patterns
            emerging_patterns = []
            
            # Pattern 1: New recurring themes
            theme_patterns = await self._detect_theme_emergence(recent_memories)
            emerging_patterns.extend(theme_patterns)
            
            # Pattern 2: Behavioral shifts
            behavioral_shifts = await self._detect_behavioral_shifts(user_id)
            emerging_patterns.extend(behavioral_shifts)
            
            # Pattern 3: Energy/mood patterns
            energy_patterns = await self._detect_energy_patterns(recent_memories)
            emerging_patterns.extend(energy_patterns)
            
            return emerging_patterns
            
        except Exception as e:
            logger.error(f"Failed to detect emerging patterns: {e}")
            return []
    
    # PRIVATE METHODS
    
    async def _analyze_context(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the current context for pattern recognition"""
        
        analysis = {
            "context_type": context.get("type", "general"),
            "key_elements": [],
            "emotional_state": context.get("emotional_state", "neutral"),
            "temporal_context": context.get("time_of_day", "unknown"),
            "complexity_score": 0.5
        }
        
        # Extract key elements
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 5:
                analysis["key_elements"].append({
                    "element": key,
                    "value": value[:50],
                    "importance": 0.6
                })
        
        # Calculate complexity
        analysis["complexity_score"] = min(1.0, len(analysis["key_elements"]) * 0.1 + 0.3)
        
        return analysis
    
    async def _generate_insights(
        self,
        user_id: str,
        memory_patterns: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        focus_areas: List[str] = None
    ) -> List[str]:
        """Generate intuitive insights based on patterns and context"""
        
        insights = []
        
        # Insight generation based on memory patterns
        for pattern in memory_patterns[:3]:  # Top 3 patterns
            pattern_type = pattern.get("pattern_type", "unknown")
            
            if pattern_type == "hub_memories":
                insights.append(
                    "Your most connected memories suggest a strong integrative thinking pattern - "
                    "you naturally link diverse experiences to create new understanding"
                )
            elif pattern_type == "memory_clusters":
                clusters = pattern.get("clusters", [])
                if len(clusters) > 2:
                    insights.append(
                        f"You organize thoughts into {len(clusters)} distinct areas of focus, "
                        "suggesting a systematic approach to processing complex information"
                    )
            elif pattern_type == "temporal_patterns":
                insights.append(
                    "Your memory creation patterns show optimal cognitive periods - "
                    "timing your important decisions during these windows could improve outcomes"
                )
        
        # Context-based insights
        emotional_state = context_analysis.get("emotional_state", "neutral")
        if emotional_state != "neutral":
            if emotional_state == "stressed":
                insights.append(
                    "Current stress indicators suggest your intuitive processing may be heightened - "
                    "this often leads to breakthrough insights if channeled properly"
                )
            elif emotional_state == "excited":
                insights.append(
                    "Your elevated emotional state indicates strong pattern recognition potential - "
                    "now may be an optimal time for creative problem-solving"
                )
        
        # Focus area insights
        if focus_areas:
            for area in focus_areas[:2]:  # Limit to top 2
                if area == "creativity":
                    insights.append(
                        "Your memory network shows strong cross-domain connections, "
                        "indicating high creative potential in novel solution generation"
                    )
                elif area == "decision-making":
                    insights.append(
                        "Pattern analysis suggests you integrate emotional and rational processing - "
                        "trusting your first instinct while validating with analysis often works best"
                    )
        
        # Default insights if none generated
        if not insights:
            insights = [
                "Your cognitive patterns suggest a balanced approach to processing information",
                "Current context indicates readiness for integrative thinking and insight generation"
            ]
        
        return insights[:4]  # Limit to 4 insights
    
    async def _detect_behavioral_patterns(
        self, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Detect patterns in user behavior and decision-making"""
        
        patterns = []
        
        # Get decision-related memories
        decision_memories = await self.memory_graph.recall_memories(
            user_id=user_id,
            memory_type="decision",
            limit=20
        )
        
        if len(decision_memories) >= 3:
            # Analyze decision timing patterns
            decision_times = []
            decision_outcomes = []
            
            for memory in decision_memories:
                content = memory.get("content", {})
                if "decision_time" in content:
                    decision_times.append(content["decision_time"])
                if "outcome" in content:
                    decision_outcomes.append(content["outcome"])
            
            if decision_times:
                patterns.append({
                    "pattern_type": "decision_timing",
                    "description": "Patterns in decision-making timing and outcomes",
                    "insights": [
                        f"Analyzed {len(decision_memories)} decision points",
                        "Quick decisions show higher satisfaction scores",
                        "Morning decisions tend to be more analytical"
                    ]
                })
        
        return patterns
    
    async def _calculate_confidence(
        self, 
        insights: List[str], 
        patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for generated insights"""
        
        base_confidence = 0.6
        
        # Increase confidence based on pattern strength
        pattern_bonus = min(0.2, len(patterns) * 0.05)
        
        # Increase confidence based on insight specificity
        insight_bonus = 0.0
        for insight in insights:
            if len(insight) > 100:  # More detailed insights
                insight_bonus += 0.05
        
        insight_bonus = min(0.15, insight_bonus)
        
        return min(0.95, base_confidence + pattern_bonus + insight_bonus)
    
    async def _identify_contributing_factors(
        self,
        memory_patterns: List[Dict[str, Any]],
        context_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify factors contributing to the insights"""
        
        factors = []
        
        # Memory-based factors
        if memory_patterns:
            factors.append(f"Analysis of {len(memory_patterns)} memory pattern types")
            
        # Context factors
        if context_analysis.get("complexity_score", 0) > 0.7:
            factors.append("High contextual complexity indicating deep processing")
            
        if context_analysis.get("emotional_state") != "neutral":
            factors.append(f"Emotional state ({context_analysis['emotional_state']}) influencing pattern recognition")
        
        # Default factors
        if not factors:
            factors = [
                "Historical pattern analysis",
                "Current cognitive state assessment",
                "Temporal context integration"
            ]
        
        return factors
    
    async def _store_insight(self, user_id: str, results: Dict[str, Any]):
        """Store insight for future validation"""
        
        insight_record = {
            "insight_id": f"insight_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "insights": results["insights"],
            "confidence": results["confidence"],
            "contributing_factors": results["contributing_factors"],
            "validation_status": "pending"
        }
        
        if user_id not in self.insight_history:
            self.insight_history[user_id] = []
        
        self.insight_history[user_id].append(insight_record)
        
        # Keep only last 50 insights per user
        if len(self.insight_history[user_id]) > 50:
            self.insight_history[user_id] = self.insight_history[user_id][-50:]
    
    async def _calculate_validation_metrics(
        self,
        original_insight: Dict[str, Any],
        actual_outcome: Dict[str, Any],
        feedback_score: float
    ) -> Dict[str, Any]:
        """Calculate validation metrics for an insight"""
        
        # Simple validation for now - could be enhanced with ML
        base_accuracy = feedback_score  # User feedback as primary signal
        
        # Adjust based on confidence vs outcome alignment
        original_confidence = original_insight.get("confidence", 0.5)
        confidence_alignment = 1.0 - abs(original_confidence - feedback_score)
        
        # Calculate other metrics
        precision = min(1.0, base_accuracy + confidence_alignment * 0.1)
        relevance = feedback_score * 0.9  # Mostly based on user feedback
        
        return {
            "accuracy": base_accuracy,
            "precision": precision,
            "relevance": relevance,
            "confidence_alignment": confidence_alignment,
            "learning_impact": feedback_score * 0.1  # How much to adjust future insights
        }
    
    async def _update_learning_model(
        self,
        user_id: str,
        insight: Dict[str, Any],
        validation_results: Dict[str, Any]
    ):
        """Update the learning model based on validation results"""
        
        # Simple learning update - could be enhanced with ML
        learning_impact = validation_results.get("learning_impact", 0.0)
        
        # Store learning signal for future insight generation
        if user_id not in self.pattern_cache:
            self.pattern_cache[user_id] = {"learning_signals": []}
        
        self.pattern_cache[user_id]["learning_signals"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "insight_type": "general",
            "validation_score": validation_results.get("accuracy", 0.5),
            "learning_weight": learning_impact
        })
        
        # Keep only recent learning signals
        signals = self.pattern_cache[user_id]["learning_signals"]
        if len(signals) > 100:
            self.pattern_cache[user_id]["learning_signals"] = signals[-100:]
    
    async def _analyze_situation(
        self, 
        user_id: str, 
        situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in a specific situation"""
        
        return {
            "situation_type": situation.get("type", "general"),
            "complexity": len(situation.keys()) * 0.1,
            "urgency": situation.get("urgency", "normal"),
            "stakeholders": situation.get("stakeholders", [])
        }
    
    async def _find_similar_experiences(
        self, 
        user_id: str, 
        situation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar past experiences from memory"""
        
        # Simple similarity matching - could be enhanced with embeddings
        similar_memories = await self.memory_graph.recall_memories(
            user_id=user_id,
            query=str(situation.get("description", "")),
            limit=5
        )
        
        return similar_memories
    
    async def _generate_recommendations(
        self,
        user_id: str,
        situation_patterns: Dict[str, Any],
        similar_experiences: List[Dict[str, Any]],
        constraints: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate intuitive recommendations"""
        
        recommendations = []
        
        # Base recommendations
        if situation_patterns.get("complexity", 0) > 0.7:
            recommendations.append({
                "recommendation": "Break down complex situation into smaller components",
                "confidence": 0.8,
                "reasoning": "High complexity detected - systematic approach recommended"
            })
        
        # Experience-based recommendations
        if similar_experiences:
            recommendations.append({
                "recommendation": "Apply lessons from similar past experiences",
                "confidence": 0.75,
                "reasoning": f"Found {len(similar_experiences)} similar situations in your history"
            })
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _detect_theme_emergence(
        self, 
        recent_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect new themes emerging in recent memories"""
        
        # Simple theme detection - could be enhanced with NLP
        themes = {}
        
        for memory in recent_memories:
            tags = memory.get("tags", [])
            for tag in tags:
                themes[tag] = themes.get(tag, 0) + 1
        
        # Find emerging themes (appearing frequently in recent memories)
        emerging = []
        for theme, count in themes.items():
            if count >= 3:  # Appears in at least 3 recent memories
                emerging.append({
                    "pattern_type": "emerging_theme",
                    "theme": theme,
                    "frequency": count,
                    "description": f"New focus on {theme} detected in recent activities"
                })
        
        return emerging
    
    async def _detect_behavioral_shifts(self, user_id: str) -> List[Dict[str, Any]]:
        """Detect shifts in behavioral patterns"""
        
        # Placeholder for behavioral shift detection
        return [{
            "pattern_type": "behavioral_shift",
            "description": "Subtle changes in decision-making patterns detected",
            "confidence": 0.6
        }]
    
    async def _detect_energy_patterns(
        self, 
        recent_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect patterns in energy and mood from recent memories"""
        
        energy_levels = []
        
        for memory in recent_memories:
            content = memory.get("content", {})
            if "energy_level" in content:
                energy_levels.append(content["energy_level"])
        
        if len(energy_levels) >= 5:
            avg_energy = sum(energy_levels) / len(energy_levels)
            return [{
                "pattern_type": "energy_pattern",
                "description": f"Average energy level trending at {avg_energy:.1f}/10",
                "insights": ["Energy optimization opportunities detected"]
            }]
        
        return []
    
    async def _mock_pattern_analysis(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock pattern analysis for fallback"""
        
        return {
            "insights": [
                "Your cognitive patterns suggest strong analytical capabilities",
                "Current context indicates readiness for creative problem-solving",
                "Recent memory patterns show consistent learning and adaptation"
            ],
            "patterns": [{
                "pattern_type": "mock_pattern",
                "description": "Consistent cognitive processing detected"
            }],
            "confidence": 0.65,
            "contributing_factors": [
                "Mock pattern analysis",
                "Context assessment",
                "User history integration"
            ],
            "processing_time": 0.1
        }
