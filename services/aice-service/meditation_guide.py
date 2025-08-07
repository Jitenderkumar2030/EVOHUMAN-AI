"""
AiCE Meditation Guide
Advanced meditation guidance and mindfulness training system
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from .memory_graph import MemoryGraphEngine
from .wisdom_engine import WisdomEngine


logger = structlog.get_logger("meditation-guide")


class MeditationType(Enum):
    MINDFULNESS = "mindfulness"
    CONCENTRATION = "concentration"
    LOVING_KINDNESS = "loving_kindness"
    BODY_SCAN = "body_scan"
    BREATHING = "breathing"
    WALKING = "walking"
    VISUALIZATION = "visualization"
    MANTRA = "mantra"


class MeditationLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class MeditationSession:
    session_id: str
    user_id: str
    meditation_type: MeditationType
    duration_minutes: int
    guidance_level: MeditationLevel
    start_time: datetime
    end_time: Optional[datetime]
    quality_score: Optional[float]
    insights: List[str]
    challenges: List[str]
    progress_notes: str


@dataclass
class MeditationProgram:
    program_id: str
    name: str
    description: str
    duration_weeks: int
    sessions_per_week: int
    meditation_types: List[MeditationType]
    target_outcomes: List[str]
    difficulty_progression: List[MeditationLevel]


class MeditationGuide:
    """Advanced meditation guidance and mindfulness training"""
    
    def __init__(self, memory_graph: MemoryGraphEngine, wisdom_engine: Optional['WisdomEngine'] = None):
        self.memory_graph = memory_graph
        self.wisdom_engine = wisdom_engine
        self.session_history: Dict[str, List[MeditationSession]] = {}
        self.user_programs: Dict[str, MeditationProgram] = {}
        
        # Meditation programs library
        self.programs_library = self._initialize_programs_library()
        
        # Guidance templates
        self.guidance_templates = self._initialize_guidance_templates()
        
        logger.info("Meditation Guide initialized")
    
    async def initialize(self):
        """Initialize meditation guide"""
        logger.info("Meditation Guide ready for guidance")
    
    async def assess_meditation_readiness(
        self,
        user_id: str,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess user's readiness for meditation"""
        
        try:
            # Analyze current mental state
            stress_level = current_state.get("stress_level", 5)  # 1-10 scale
            energy_level = current_state.get("energy_level", 5)  # 1-10 scale
            focus_ability = current_state.get("focus_ability", 5)  # 1-10 scale
            emotional_state = current_state.get("emotional_state", "neutral")
            
            # Get meditation history
            history = self.session_history.get(user_id, [])
            experience_level = self._assess_experience_level(history)
            
            # Determine optimal meditation type
            recommended_type = self._recommend_meditation_type(
                stress_level, energy_level, focus_ability, emotional_state, experience_level
            )
            
            # Determine session duration
            recommended_duration = self._recommend_session_duration(
                experience_level, stress_level, focus_ability
            )
            
            # Generate personalized guidance
            guidance = await self._generate_personalized_guidance(
                user_id, recommended_type, experience_level, current_state
            )
            
            return {
                "readiness_score": self._calculate_readiness_score(current_state),
                "recommended_meditation": {
                    "type": recommended_type.value,
                    "duration_minutes": recommended_duration,
                    "guidance_level": experience_level.value
                },
                "personalized_guidance": guidance,
                "preparation_suggestions": self._get_preparation_suggestions(current_state),
                "optimal_conditions": self._suggest_optimal_conditions(current_state)
            }
            
        except Exception as e:
            logger.error("Meditation readiness assessment failed", user_id=user_id, error=str(e))
            raise
    
    async def start_guided_session(
        self,
        user_id: str,
        meditation_type: MeditationType,
        duration_minutes: int,
        guidance_level: MeditationLevel = MeditationLevel.INTERMEDIATE
    ) -> Dict[str, Any]:
        """Start a guided meditation session"""
        
        try:
            session_id = f"med_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create session
            session = MeditationSession(
                session_id=session_id,
                user_id=user_id,
                meditation_type=meditation_type,
                duration_minutes=duration_minutes,
                guidance_level=guidance_level,
                start_time=datetime.utcnow(),
                end_time=None,
                quality_score=None,
                insights=[],
                challenges=[],
                progress_notes=""
            )
            
            # Get guidance script
            guidance_script = self._get_guidance_script(meditation_type, duration_minutes, guidance_level)
            
            # Generate real-time guidance
            real_time_guidance = await self._generate_real_time_guidance(
                user_id, meditation_type, guidance_level
            )
            
            # Store session start
            if user_id not in self.session_history:
                self.session_history[user_id] = []
            self.session_history[user_id].append(session)
            
            logger.info("Meditation session started", 
                       user_id=user_id,
                       session_id=session_id,
                       type=meditation_type.value)
            
            return {
                "session_id": session_id,
                "meditation_type": meditation_type.value,
                "duration_minutes": duration_minutes,
                "guidance_script": guidance_script,
                "real_time_guidance": real_time_guidance,
                "session_structure": self._get_session_structure(meditation_type, duration_minutes)
            }
            
        except Exception as e:
            logger.error("Failed to start meditation session", user_id=user_id, error=str(e))
            raise
    
    async def complete_session(
        self,
        user_id: str,
        session_id: str,
        session_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete meditation session and process feedback"""
        
        try:
            # Find session
            session = None
            if user_id in self.session_history:
                for s in self.session_history[user_id]:
                    if s.session_id == session_id:
                        session = s
                        break
            
            if not session:
                raise ValueError("Session not found")
            
            # Update session with completion data
            session.end_time = datetime.utcnow()
            session.quality_score = session_feedback.get("quality_score", 7.0)
            session.insights = session_feedback.get("insights", [])
            session.challenges = session_feedback.get("challenges", [])
            session.progress_notes = session_feedback.get("progress_notes", "")
            
            # Analyze session quality
            session_analysis = await self._analyze_session_quality(session, session_feedback)
            
            # Generate insights and recommendations
            insights = await self._generate_session_insights(user_id, session, session_analysis)
            
            # Update user's meditation profile
            await self._update_meditation_profile(user_id, session)
            
            # Store session memory
            await self._store_session_memory(user_id, session, insights)
            
            logger.info("Meditation session completed", 
                       user_id=user_id,
                       session_id=session_id,
                       quality_score=session.quality_score)
            
            return {
                "session_summary": {
                    "duration_actual": (session.end_time - session.start_time).total_seconds() / 60,
                    "quality_score": session.quality_score,
                    "meditation_type": session.meditation_type.value
                },
                "session_analysis": session_analysis,
                "insights": insights,
                "progress_update": await self._get_progress_update(user_id),
                "next_session_recommendations": await self._recommend_next_session(user_id)
            }
            
        except Exception as e:
            logger.error("Failed to complete meditation session", 
                        user_id=user_id, session_id=session_id, error=str(e))
            raise
    
    async def create_personalized_program(
        self,
        user_id: str,
        goals: List[str],
        available_time_per_week: int,
        experience_level: MeditationLevel,
        preferences: Dict[str, Any]
    ) -> MeditationProgram:
        """Create personalized meditation program"""
        
        try:
            # Analyze user's meditation history
            history = self.session_history.get(user_id, [])
            
            # Determine program parameters
            program_duration = self._determine_program_duration(goals, experience_level)
            sessions_per_week = min(7, max(2, available_time_per_week // 15))  # 15 min minimum
            
            # Select meditation types based on goals
            meditation_types = self._select_meditation_types_for_goals(goals, preferences)
            
            # Create program structure
            program_id = f"prog_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            program = MeditationProgram(
                program_id=program_id,
                name=f"Personalized Program for {user_id}",
                description=f"Custom program targeting: {', '.join(goals)}",
                duration_weeks=program_duration,
                sessions_per_week=sessions_per_week,
                meditation_types=meditation_types,
                target_outcomes=goals,
                difficulty_progression=self._create_difficulty_progression(
                    experience_level, program_duration
                )
            )
            
            # Store program
            self.user_programs[user_id] = program
            
            # Generate weekly schedule
            weekly_schedule = self._generate_weekly_schedule(program)
            
            logger.info("Personalized meditation program created", 
                       user_id=user_id,
                       program_id=program_id,
                       duration_weeks=program_duration)
            
            return {
                "program": program,
                "weekly_schedule": weekly_schedule,
                "estimated_outcomes": self._estimate_program_outcomes(program, history)
            }
            
        except Exception as e:
            logger.error("Failed to create personalized program", user_id=user_id, error=str(e))
            raise
    
    def _assess_experience_level(self, history: List[MeditationSession]) -> MeditationLevel:
        """Assess user's meditation experience level"""
        
        if not history:
            return MeditationLevel.BEGINNER
        
        total_sessions = len(history)
        total_minutes = sum(s.duration_minutes for s in history)
        avg_quality = np.mean([s.quality_score for s in history if s.quality_score])
        
        if total_sessions >= 100 and total_minutes >= 1000 and avg_quality >= 8.0:
            return MeditationLevel.EXPERT
        elif total_sessions >= 50 and total_minutes >= 500 and avg_quality >= 7.0:
            return MeditationLevel.ADVANCED
        elif total_sessions >= 10 and total_minutes >= 150:
            return MeditationLevel.INTERMEDIATE
        else:
            return MeditationLevel.BEGINNER
    
    def _recommend_meditation_type(
        self,
        stress_level: int,
        energy_level: int,
        focus_ability: int,
        emotional_state: str,
        experience_level: MeditationLevel
    ) -> MeditationType:
        """Recommend optimal meditation type"""
        
        # High stress - breathing or body scan
        if stress_level >= 7:
            return MeditationType.BREATHING if experience_level == MeditationLevel.BEGINNER else MeditationType.BODY_SCAN
        
        # Low energy - gentle mindfulness or loving kindness
        if energy_level <= 3:
            return MeditationType.LOVING_KINDNESS
        
        # Poor focus - concentration practice
        if focus_ability <= 4:
            return MeditationType.CONCENTRATION
        
        # Negative emotions - loving kindness
        if emotional_state in ["sad", "angry", "anxious"]:
            return MeditationType.LOVING_KINDNESS
        
        # High energy - walking or visualization
        if energy_level >= 8:
            return MeditationType.WALKING
        
        # Default to mindfulness
        return MeditationType.MINDFULNESS
    
    def _recommend_session_duration(
        self,
        experience_level: MeditationLevel,
        stress_level: int,
        focus_ability: int
    ) -> int:
        """Recommend session duration in minutes"""
        
        base_duration = {
            MeditationLevel.BEGINNER: 10,
            MeditationLevel.INTERMEDIATE: 20,
            MeditationLevel.ADVANCED: 30,
            MeditationLevel.EXPERT: 45
        }[experience_level]
        
        # Adjust for stress and focus
        if stress_level >= 8 or focus_ability <= 3:
            base_duration = max(5, base_duration - 10)
        elif stress_level <= 3 and focus_ability >= 8:
            base_duration += 10
        
        return base_duration
    
    def _calculate_readiness_score(self, current_state: Dict[str, Any]) -> float:
        """Calculate meditation readiness score (0-10)"""
        
        stress_level = current_state.get("stress_level", 5)
        energy_level = current_state.get("energy_level", 5)
        focus_ability = current_state.get("focus_ability", 5)
        motivation = current_state.get("motivation", 5)
        
        # Higher stress increases readiness (need for meditation)
        stress_factor = min(10, stress_level * 1.2)
        
        # Moderate energy is optimal
        energy_factor = 10 - abs(energy_level - 6)
        
        # Higher focus ability increases readiness
        focus_factor = focus_ability
        
        # Higher motivation increases readiness
        motivation_factor = motivation
        
        readiness = (stress_factor * 0.3 + energy_factor * 0.2 + 
                    focus_factor * 0.3 + motivation_factor * 0.2)
        
        return min(10.0, max(1.0, readiness))
    
    async def _generate_personalized_guidance(
        self,
        user_id: str,
        meditation_type: MeditationType,
        experience_level: MeditationLevel,
        current_state: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized meditation guidance"""
        
        guidance = []
        
        # Get user's meditation history for personalization
        history = self.session_history.get(user_id, [])
        
        # Base guidance for meditation type
        base_guidance = self.guidance_templates.get(meditation_type, [])
        guidance.extend(base_guidance[:3])  # First 3 items
        
        # Personalize based on experience level
        if experience_level == MeditationLevel.BEGINNER:
            guidance.append("Remember, it's normal for your mind to wander - gently return focus when you notice")
            guidance.append("Start with shorter sessions and gradually increase duration")
        elif experience_level == MeditationLevel.EXPERT:
            guidance.append("Explore subtle aspects of your practice today")
            guidance.append("Consider investigating the nature of awareness itself")
        
        # Personalize based on current state
        stress_level = current_state.get("stress_level", 5)
        if stress_level >= 7:
            guidance.append("Focus extra attention on releasing tension with each exhale")
        
        # Add wisdom-based guidance if available
        if self.wisdom_engine:
            try:
                wisdom_guidance = await self.wisdom_engine.generate_contextual_wisdom(
                    user_id, f"meditation guidance for {meditation_type.value}"
                )
                if wisdom_guidance:
                    guidance.append(wisdom_guidance[:100])  # Truncate if too long
            except Exception:
                pass  # Wisdom engine not available
        
        return guidance[:7]  # Limit to 7 guidance points
    
    def _get_preparation_suggestions(self, current_state: Dict[str, Any]) -> List[str]:
        """Get meditation preparation suggestions"""
        
        suggestions = [
            "Find a quiet, comfortable space where you won't be disturbed",
            "Sit in a comfortable position with your spine naturally upright",
            "Turn off or silence electronic devices"
        ]
        
        stress_level = current_state.get("stress_level", 5)
        if stress_level >= 7:
            suggestions.append("Take a few deep breaths to begin settling your nervous system")
            suggestions.append("Consider doing some gentle stretching before sitting")
        
        energy_level = current_state.get("energy_level", 5)
        if energy_level <= 3:
            suggestions.append("Ensure you're not too tired - consider meditating at a different time")
        elif energy_level >= 8:
            suggestions.append("Consider doing some light movement to settle excess energy")
        
        return suggestions
    
    def _suggest_optimal_conditions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal conditions for meditation"""
        
        return {
            "lighting": "Soft, dim lighting or natural light",
            "temperature": "Comfortable room temperature (68-72°F)",
            "posture": "Seated with spine upright, hands resting comfortably",
            "duration": "Start with what feels manageable, gradually increase",
            "timing": "Choose a consistent time when you're naturally alert",
            "environment": "Quiet space with minimal distractions"
        }
    
    def _initialize_programs_library(self) -> Dict[str, MeditationProgram]:
        """Initialize library of meditation programs"""
        
        return {
            "stress_reduction": MeditationProgram(
                program_id="stress_reduction_8week",
                name="8-Week Stress Reduction Program",
                description="Comprehensive program for reducing stress and anxiety",
                duration_weeks=8,
                sessions_per_week=5,
                meditation_types=[MeditationType.BREATHING, MeditationType.BODY_SCAN, MeditationType.MINDFULNESS],
                target_outcomes=["reduced_stress", "improved_sleep", "better_emotional_regulation"],
                difficulty_progression=[MeditationLevel.BEGINNER, MeditationLevel.INTERMEDIATE]
            ),
            "focus_enhancement": MeditationProgram(
                program_id="focus_enhancement_6week",
                name="6-Week Focus Enhancement Program",
                description="Develop sustained attention and concentration",
                duration_weeks=6,
                sessions_per_week=4,
                meditation_types=[MeditationType.CONCENTRATION, MeditationType.MINDFULNESS],
                target_outcomes=["improved_focus", "enhanced_concentration", "better_productivity"],
                difficulty_progression=[MeditationLevel.INTERMEDIATE, MeditationLevel.ADVANCED]
            )
        }
    
    def _initialize_guidance_templates(self) -> Dict[MeditationType, List[str]]:
        """Initialize guidance templates for different meditation types"""
        
        return {
            MeditationType.MINDFULNESS: [
                "Begin by noticing your breath without trying to change it",
                "When thoughts arise, acknowledge them gently and return to the breath",
                "Observe sensations, thoughts, and emotions with curious awareness",
                "Rest in the present moment, just as it is"
            ],
            MeditationType.BREATHING: [
                "Focus your attention on the sensation of breathing",
                "Notice the breath at the nostrils, chest, or belly",
                "Count breaths from 1 to 10, then start over",
                "When the mind wanders, gently return to counting"
            ],
            MeditationType.BODY_SCAN: [
                "Start by noticing sensations in your feet",
                "Slowly move your attention up through each part of your body",
                "Notice areas of tension or relaxation without judgment",
                "Send breath and awareness to each body region"
            ],
            MeditationType.LOVING_KINDNESS: [
                "Begin by sending loving wishes to yourself",
                "Extend loving-kindness to someone you care about",
                "Include someone neutral, then someone difficult",
                "Finally, send loving-kindness to all beings everywhere"
            ]
        }
    
    def _get_guidance_script(
        self,
        meditation_type: MeditationType,
        duration_minutes: int,
        guidance_level: MeditationLevel
    ) -> Dict[str, Any]:
        """Get structured guidance script for session"""
        
        # This would contain detailed, time-based guidance
        # For brevity, returning a simplified structure
        
        return {
            "introduction": f"Welcome to your {duration_minutes}-minute {meditation_type.value} meditation",
            "settling_in": "Take a moment to settle into your posture and close your eyes",
            "main_practice": self.guidance_templates.get(meditation_type, []),
            "conclusion": "Gently bring your attention back to the room and open your eyes when ready",
            "timing_cues": self._generate_timing_cues(duration_minutes)
        }
    
    def _generate_timing_cues(self, duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate timing cues for meditation session"""
        
        cues = []
        
        # Beginning (first 10% of session)
        cues.append({
            "time_minutes": 1,
            "cue": "Allow yourself to settle into the practice"
        })
        
        # Middle reminders
        if duration_minutes >= 15:
            cues.append({
                "time_minutes": duration_minutes // 2,
                "cue": "You're halfway through - continue with gentle awareness"
            })
        
        # Near end (last 10% of session)
        end_cue_time = max(1, duration_minutes - 2)
        cues.append({
            "time_minutes": end_cue_time,
            "cue": "Begin to prepare for the end of your practice"
        })
        
        return cues
    
    async def _store_session_memory(
        self,
        user_id: str,
        session: MeditationSession,
        insights: List[str]
    ):
        """Store meditation session in memory graph"""
        
        try:
            memory_content = {
                "session_type": "meditation",
                "meditation_type": session.meditation_type.value,
                "duration_minutes": session.duration_minutes,
                "quality_score": session.quality_score,
                "insights": insights[:3],  # Top 3 insights
                "session_date": session.start_time.isoformat()
            }
            
            await self.memory_graph.store_memory(
                user_id=user_id,
                memory_type="meditation_session",
                content=memory_content,
                importance=0.7,
                tags=["meditation", "mindfulness", session.meditation_type.value]
            )
            
        except Exception as e:
            logger.warning("Failed to store meditation session in memory", error=str(e))
    
    async def get_meditation_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user's meditation statistics"""
        
        history = self.session_history.get(user_id, [])
        
        if not history:
            return {
                "total_sessions": 0,
                "total_minutes": 0,
                "average_quality": 0.0,
                "favorite_type": None,
                "streak_days": 0
            }
        
        total_sessions = len(history)
        total_minutes = sum(s.duration_minutes for s in history)
        quality_scores = [s.quality_score for s in history if s.quality_score]
        average_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Find favorite meditation type
        type_counts = {}
        for session in history:
            type_counts[session.meditation_type.value] = type_counts.get(session.meditation_type.value, 0) + 1
        
        favorite_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        
        # Calculate streak (simplified)
        streak_days = self._calculate_meditation_streak(history)
        
        return {
            "total_sessions": total_sessions,
            "total_minutes": total_minutes,
            "average_quality": round(average_quality, 1),
            "favorite_type": favorite_type,
            "streak_days": streak_days,
            "sessions_this_week": len([s for s in history if (datetime.utcnow() - s.start_time).days <= 7]),
            "improvement_trend": self._calculate_improvement_trend(history)
        }
    
    def _calculate_meditation_streak(self, history: List[MeditationSession]) -> int:
        """Calculate current meditation streak in days"""
        
        if not history:
            return 0
        
        # Sort by date
        sorted_sessions = sorted(history, key=lambda x: x.start_time, reverse=True)
        
        streak = 0
        current_date = datetime.utcnow().date()
        
        # Check if meditated today or yesterday
        if (current_date - sorted_sessions[0].start_time.date()).days <= 1:
            streak = 1
            last_date = sorted_sessions[0].start_time.date()
            
            # Count consecutive days
            for session in sorted_sessions[1:]:
                session_date = session.start_time.date()
                if (last_date - session_date).days == 1:
                    streak += 1
                    last_date = session_date
                elif (last_date - session_date).days == 0:
                    continue  # Same day, multiple sessions
                else:
                    break  # Streak broken
        
        return streak
    
    def _calculate_improvement_trend(self, history: List[MeditationSession]) -> str:
        """Calculate improvement trend over time"""
        
        if len(history) < 5:
            return "insufficient_data"
        
        # Get quality scores from recent sessions
        recent_sessions = sorted(history, key=lambda x: x.start_time)[-10:]
        quality_scores = [s.quality_score for s in recent_sessions if s.quality_score]
        
        if len(quality_scores) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = quality_scores[:len(quality_scores)//2]
        second_half = quality_scores[len(quality_scores)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        if second_avg > first_avg + 0.5:
            return "improving"
        elif second_avg < first_avg - 0.5:
            return "declining"
        else:
            return "stable"
