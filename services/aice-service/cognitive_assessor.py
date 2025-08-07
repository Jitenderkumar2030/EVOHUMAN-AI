"""
AiCE Cognitive Assessment Engine
Comprehensive cognitive function evaluation and enhancement tracking
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .memory_graph import MemoryGraphEngine


logger = structlog.get_logger("cognitive-assessor")


class CognitiveFunction(Enum):
    MEMORY = "memory"
    ATTENTION = "attention"
    PROCESSING_SPEED = "processing_speed"
    EXECUTIVE_FUNCTION = "executive_function"
    LANGUAGE = "language"
    VISUOSPATIAL = "visuospatial"
    REASONING = "reasoning"
    LEARNING = "learning"


@dataclass
class CognitiveScore:
    function: CognitiveFunction
    raw_score: float
    percentile: float
    z_score: float
    interpretation: str
    confidence: float


@dataclass
class CognitiveProfile:
    user_id: str
    assessment_date: datetime
    overall_score: float
    function_scores: Dict[CognitiveFunction, CognitiveScore]
    strengths: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]


class CognitiveAssessor:
    """Advanced cognitive assessment and enhancement tracking"""
    
    def __init__(self, memory_graph: MemoryGraphEngine):
        self.memory_graph = memory_graph
        self.assessment_history: Dict[str, List[CognitiveProfile]] = {}
        
        # Normative data (simplified - in production would use real norms)
        self.normative_data = {
            CognitiveFunction.MEMORY: {"mean": 100, "std": 15},
            CognitiveFunction.ATTENTION: {"mean": 100, "std": 15},
            CognitiveFunction.PROCESSING_SPEED: {"mean": 100, "std": 15},
            CognitiveFunction.EXECUTIVE_FUNCTION: {"mean": 100, "std": 15},
            CognitiveFunction.LANGUAGE: {"mean": 100, "std": 15},
            CognitiveFunction.VISUOSPATIAL: {"mean": 100, "std": 15},
            CognitiveFunction.REASONING: {"mean": 100, "std": 15},
            CognitiveFunction.LEARNING: {"mean": 100, "std": 15}
        }
        
        logger.info("Cognitive Assessor initialized")
    
    async def conduct_comprehensive_assessment(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveProfile:
        """Conduct comprehensive cognitive assessment"""
        
        logger.info("Starting comprehensive cognitive assessment", user_id=user_id)
        
        try:
            # Analyze different cognitive functions
            function_scores = {}
            
            # Memory assessment
            memory_score = await self._assess_memory_function(user_id, assessment_data)
            function_scores[CognitiveFunction.MEMORY] = memory_score
            
            # Attention assessment
            attention_score = await self._assess_attention_function(user_id, assessment_data)
            function_scores[CognitiveFunction.ATTENTION] = attention_score
            
            # Processing speed assessment
            speed_score = await self._assess_processing_speed(user_id, assessment_data)
            function_scores[CognitiveFunction.PROCESSING_SPEED] = speed_score
            
            # Executive function assessment
            exec_score = await self._assess_executive_function(user_id, assessment_data)
            function_scores[CognitiveFunction.EXECUTIVE_FUNCTION] = exec_score
            
            # Language assessment
            language_score = await self._assess_language_function(user_id, assessment_data)
            function_scores[CognitiveFunction.LANGUAGE] = language_score
            
            # Visuospatial assessment
            visuospatial_score = await self._assess_visuospatial_function(user_id, assessment_data)
            function_scores[CognitiveFunction.VISUOSPATIAL] = visuospatial_score
            
            # Reasoning assessment
            reasoning_score = await self._assess_reasoning_function(user_id, assessment_data)
            function_scores[CognitiveFunction.REASONING] = reasoning_score
            
            # Learning assessment
            learning_score = await self._assess_learning_function(user_id, assessment_data)
            function_scores[CognitiveFunction.LEARNING] = learning_score
            
            # Calculate overall score
            overall_score = np.mean([score.raw_score for score in function_scores.values()])
            
            # Identify strengths and weaknesses
            strengths, areas_for_improvement = self._identify_cognitive_patterns(function_scores)
            
            # Generate recommendations
            recommendations = await self._generate_enhancement_recommendations(
                user_id, function_scores, assessment_data
            )
            
            # Analyze trends if previous assessments exist
            trend_analysis = await self._analyze_cognitive_trends(user_id, function_scores)
            
            # Create cognitive profile
            profile = CognitiveProfile(
                user_id=user_id,
                assessment_date=datetime.utcnow(),
                overall_score=overall_score,
                function_scores=function_scores,
                strengths=strengths,
                areas_for_improvement=areas_for_improvement,
                recommendations=recommendations,
                trend_analysis=trend_analysis
            )
            
            # Store assessment history
            if user_id not in self.assessment_history:
                self.assessment_history[user_id] = []
            self.assessment_history[user_id].append(profile)
            
            # Store in memory graph
            await self._store_assessment_in_memory(user_id, profile)
            
            logger.info("Cognitive assessment completed", 
                       user_id=user_id,
                       overall_score=overall_score)
            
            return profile
            
        except Exception as e:
            logger.error("Cognitive assessment failed", user_id=user_id, error=str(e))
            raise
    
    async def _assess_memory_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess memory function"""
        
        # Analyze memory graph statistics
        memory_stats = await self.memory_graph.find_memory_patterns(user_id)
        
        # Calculate memory metrics
        memory_recall_accuracy = assessment_data.get("memory_recall_accuracy", 0.75)
        memory_retention_rate = assessment_data.get("memory_retention_rate", 0.80)
        working_memory_span = assessment_data.get("working_memory_span", 7)
        
        # Memory graph connectivity (proxy for associative memory)
        graph_connectivity = len(memory_stats) / max(1, len(memory_stats)) if memory_stats else 0.5
        
        # Composite memory score
        raw_score = (
            memory_recall_accuracy * 30 +
            memory_retention_rate * 30 +
            (working_memory_span / 9) * 25 +  # Normalize to 0-1
            graph_connectivity * 15
        ) * 100 / 100  # Scale to 100
        
        return self._create_cognitive_score(CognitiveFunction.MEMORY, raw_score)
    
    async def _assess_attention_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess attention and focus"""
        
        sustained_attention = assessment_data.get("sustained_attention_score", 0.75)
        selective_attention = assessment_data.get("selective_attention_score", 0.80)
        divided_attention = assessment_data.get("divided_attention_score", 0.70)
        attention_switching = assessment_data.get("attention_switching_score", 0.75)
        
        raw_score = (
            sustained_attention * 25 +
            selective_attention * 25 +
            divided_attention * 25 +
            attention_switching * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.ATTENTION, raw_score)
    
    async def _assess_processing_speed(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess cognitive processing speed"""
        
        reaction_time = assessment_data.get("reaction_time_ms", 500)  # milliseconds
        symbol_coding_speed = assessment_data.get("symbol_coding_speed", 60)  # symbols per minute
        pattern_matching_speed = assessment_data.get("pattern_matching_speed", 45)  # matches per minute
        
        # Convert to normalized scores (lower reaction time = better)
        reaction_score = max(0, (1000 - reaction_time) / 1000)
        coding_score = min(1.0, symbol_coding_speed / 80)  # 80 is excellent
        matching_score = min(1.0, pattern_matching_speed / 60)  # 60 is excellent
        
        raw_score = (reaction_score * 40 + coding_score * 30 + matching_score * 30) * 100
        
        return self._create_cognitive_score(CognitiveFunction.PROCESSING_SPEED, raw_score)
    
    async def _assess_executive_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess executive function"""
        
        planning_ability = assessment_data.get("planning_ability_score", 0.75)
        cognitive_flexibility = assessment_data.get("cognitive_flexibility_score", 0.80)
        inhibitory_control = assessment_data.get("inhibitory_control_score", 0.75)
        working_memory_updating = assessment_data.get("working_memory_updating_score", 0.70)
        
        raw_score = (
            planning_ability * 25 +
            cognitive_flexibility * 25 +
            inhibitory_control * 25 +
            working_memory_updating * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.EXECUTIVE_FUNCTION, raw_score)
    
    async def _assess_language_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess language abilities"""
        
        vocabulary_size = assessment_data.get("vocabulary_size", 15000)
        verbal_fluency = assessment_data.get("verbal_fluency_score", 0.75)
        comprehension_accuracy = assessment_data.get("comprehension_accuracy", 0.85)
        naming_speed = assessment_data.get("naming_speed_score", 0.80)
        
        # Normalize vocabulary (average adult has ~20,000-35,000 words)
        vocab_score = min(1.0, vocabulary_size / 25000)
        
        raw_score = (
            vocab_score * 25 +
            verbal_fluency * 25 +
            comprehension_accuracy * 25 +
            naming_speed * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.LANGUAGE, raw_score)
    
    async def _assess_visuospatial_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess visuospatial abilities"""
        
        spatial_rotation = assessment_data.get("spatial_rotation_score", 0.75)
        visual_memory = assessment_data.get("visual_memory_score", 0.80)
        spatial_navigation = assessment_data.get("spatial_navigation_score", 0.75)
        pattern_recognition = assessment_data.get("pattern_recognition_score", 0.85)
        
        raw_score = (
            spatial_rotation * 25 +
            visual_memory * 25 +
            spatial_navigation * 25 +
            pattern_recognition * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.VISUOSPATIAL, raw_score)
    
    async def _assess_reasoning_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess reasoning abilities"""
        
        logical_reasoning = assessment_data.get("logical_reasoning_score", 0.75)
        abstract_reasoning = assessment_data.get("abstract_reasoning_score", 0.80)
        analogical_reasoning = assessment_data.get("analogical_reasoning_score", 0.75)
        problem_solving = assessment_data.get("problem_solving_score", 0.80)
        
        raw_score = (
            logical_reasoning * 25 +
            abstract_reasoning * 25 +
            analogical_reasoning * 25 +
            problem_solving * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.REASONING, raw_score)
    
    async def _assess_learning_function(
        self,
        user_id: str,
        assessment_data: Dict[str, Any]
    ) -> CognitiveScore:
        """Assess learning abilities"""
        
        learning_rate = assessment_data.get("learning_rate_score", 0.75)
        transfer_learning = assessment_data.get("transfer_learning_score", 0.70)
        skill_acquisition = assessment_data.get("skill_acquisition_score", 0.80)
        knowledge_retention = assessment_data.get("knowledge_retention_score", 0.85)
        
        raw_score = (
            learning_rate * 25 +
            transfer_learning * 25 +
            skill_acquisition * 25 +
            knowledge_retention * 25
        ) * 100
        
        return self._create_cognitive_score(CognitiveFunction.LEARNING, raw_score)
    
    def _create_cognitive_score(
        self,
        function: CognitiveFunction,
        raw_score: float
    ) -> CognitiveScore:
        """Create standardized cognitive score"""
        
        norms = self.normative_data[function]
        z_score = (raw_score - norms["mean"]) / norms["std"]
        percentile = self._z_to_percentile(z_score)
        
        # Interpretation
        if percentile >= 90:
            interpretation = "Superior"
        elif percentile >= 75:
            interpretation = "Above Average"
        elif percentile >= 25:
            interpretation = "Average"
        elif percentile >= 10:
            interpretation = "Below Average"
        else:
            interpretation = "Impaired"
        
        # Confidence based on score consistency
        confidence = max(0.7, min(0.95, 0.8 + (abs(z_score) * 0.05)))
        
        return CognitiveScore(
            function=function,
            raw_score=raw_score,
            percentile=percentile,
            z_score=z_score,
            interpretation=interpretation,
            confidence=confidence
        )
    
    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile"""
        # Simplified conversion (in production would use proper statistical functions)
        if z_score >= 2.0:
            return 97.5
        elif z_score >= 1.5:
            return 93.3
        elif z_score >= 1.0:
            return 84.1
        elif z_score >= 0.5:
            return 69.1
        elif z_score >= 0.0:
            return 50.0
        elif z_score >= -0.5:
            return 30.9
        elif z_score >= -1.0:
            return 15.9
        elif z_score >= -1.5:
            return 6.7
        elif z_score >= -2.0:
            return 2.5
        else:
            return 1.0
    
    def _identify_cognitive_patterns(
        self,
        function_scores: Dict[CognitiveFunction, CognitiveScore]
    ) -> Tuple[List[str], List[str]]:
        """Identify cognitive strengths and weaknesses"""
        
        strengths = []
        areas_for_improvement = []
        
        for function, score in function_scores.items():
            if score.percentile >= 75:
                strengths.append(f"{function.value.replace('_', ' ').title()}: {score.interpretation}")
            elif score.percentile <= 25:
                areas_for_improvement.append(f"{function.value.replace('_', ' ').title()}: {score.interpretation}")
        
        return strengths, areas_for_improvement
    
    async def _generate_enhancement_recommendations(
        self,
        user_id: str,
        function_scores: Dict[CognitiveFunction, CognitiveScore],
        assessment_data: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized cognitive enhancement recommendations"""
        
        recommendations = []
        
        # Analyze each function and provide targeted recommendations
        for function, score in function_scores.items():
            if score.percentile <= 50:  # Below average or average
                if function == CognitiveFunction.MEMORY:
                    recommendations.append("Practice spaced repetition and memory palace techniques")
                    recommendations.append("Engage in regular memory training exercises")
                elif function == CognitiveFunction.ATTENTION:
                    recommendations.append("Practice mindfulness meditation to improve sustained attention")
                    recommendations.append("Use attention training apps and exercises")
                elif function == CognitiveFunction.PROCESSING_SPEED:
                    recommendations.append("Engage in speed-based cognitive training games")
                    recommendations.append("Practice rapid decision-making exercises")
                elif function == CognitiveFunction.EXECUTIVE_FUNCTION:
                    recommendations.append("Practice planning and organization skills")
                    recommendations.append("Engage in strategy games and puzzles")
                elif function == CognitiveFunction.LANGUAGE:
                    recommendations.append("Expand vocabulary through reading and word games")
                    recommendations.append("Practice verbal fluency exercises")
                elif function == CognitiveFunction.VISUOSPATIAL:
                    recommendations.append("Practice spatial rotation and navigation tasks")
                    recommendations.append("Engage in visual-spatial puzzles and games")
                elif function == CognitiveFunction.REASONING:
                    recommendations.append("Practice logical reasoning and problem-solving")
                    recommendations.append("Engage in abstract thinking exercises")
                elif function == CognitiveFunction.LEARNING:
                    recommendations.append("Practice active learning techniques")
                    recommendations.append("Engage in diverse learning experiences")
        
        # General recommendations
        recommendations.extend([
            "Maintain regular physical exercise for cognitive health",
            "Ensure adequate sleep (7-9 hours) for optimal cognitive function",
            "Practice stress management techniques",
            "Engage in social activities to maintain cognitive stimulation"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _analyze_cognitive_trends(
        self,
        user_id: str,
        current_scores: Dict[CognitiveFunction, CognitiveScore]
    ) -> Dict[str, Any]:
        """Analyze cognitive trends over time"""
        
        if user_id not in self.assessment_history or len(self.assessment_history[user_id]) < 2:
            return {"trend_available": False, "message": "Insufficient data for trend analysis"}
        
        previous_assessment = self.assessment_history[user_id][-2]  # Second to last
        
        trends = {}
        overall_change = 0.0
        
        for function, current_score in current_scores.items():
            if function in previous_assessment.function_scores:
                previous_score = previous_assessment.function_scores[function]
                change = current_score.raw_score - previous_score.raw_score
                trends[function.value] = {
                    "change": change,
                    "direction": "improved" if change > 2 else "declined" if change < -2 else "stable",
                    "magnitude": abs(change)
                }
                overall_change += change
        
        return {
            "trend_available": True,
            "overall_change": overall_change / len(current_scores),
            "function_trends": trends,
            "assessment_interval_days": (datetime.utcnow() - previous_assessment.assessment_date).days
        }
    
    async def _store_assessment_in_memory(
        self,
        user_id: str,
        profile: CognitiveProfile
    ):
        """Store cognitive assessment in memory graph"""
        
        try:
            memory_content = {
                "assessment_type": "cognitive_assessment",
                "overall_score": profile.overall_score,
                "strengths": profile.strengths,
                "areas_for_improvement": profile.areas_for_improvement,
                "recommendations": profile.recommendations[:5],  # Top 5
                "assessment_date": profile.assessment_date.isoformat()
            }
            
            await self.memory_graph.store_memory(
                user_id=user_id,
                memory_type="cognitive_assessment",
                content=memory_content,
                importance=0.8,  # High importance
                tags=["cognitive", "assessment", "enhancement"]
            )
            
        except Exception as e:
            logger.warning("Failed to store assessment in memory", error=str(e))
    
    async def get_assessment_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's cognitive assessment history"""
        
        if user_id not in self.assessment_history:
            return []
        
        history = []
        for profile in self.assessment_history[user_id]:
            history.append({
                "assessment_date": profile.assessment_date.isoformat(),
                "overall_score": profile.overall_score,
                "strengths_count": len(profile.strengths),
                "improvements_count": len(profile.areas_for_improvement),
                "recommendations_count": len(profile.recommendations)
            })
        
        return history
