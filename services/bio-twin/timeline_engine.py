"""
Timeline Engine for tracking user evolution over time
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import redis
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

from shared.models import (
    TimelineEvent, UserEvolutionTimeline, ModelMetadata, 
    ExplanationData, TwinInsight
)
from shared.constants import (
    TIMELINE_EVENT_TYPES, CACHE_KEYS, MODEL_VERSIONS, SERVICE_NAMES
)
from shared.utils import generate_id, utc_now


class TimelineEngine:
    """Manages user evolution timeline tracking"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.logger = structlog.get_logger("timeline_engine")
        self.redis_client = redis_client or redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
    
    async def add_event(
        self, 
        user_id: str, 
        event_type: str, 
        title: str, 
        description: str, 
        data: Dict[str, Any],
        source_service: str = "bio-twin",
        impact_score: Optional[float] = None,
        tags: List[str] = None
    ) -> TimelineEvent:
        """Add a new event to user's timeline"""
        
        event = TimelineEvent(
            id=generate_id(),
            user_id=user_id,
            event_type=event_type,
            timestamp=utc_now(),
            title=title,
            description=description,
            data=data,
            impact_score=impact_score,
            tags=tags or [],
            source_service=source_service
        )
        
        # Store event in Redis
        timeline_key = CACHE_KEYS["USER_TIMELINE"].format(user_id=user_id)
        event_data = event.model_dump()
        
        # Add to timeline list (most recent first)
        self.redis_client.lpush(timeline_key, json.dumps(event_data))
        
        # Keep only last 1000 events
        self.redis_client.ltrim(timeline_key, 0, 999)
        
        # Set expiration (1 year)
        self.redis_client.expire(timeline_key, 31536000)
        
        self.logger.info(
            "Timeline event added",
            user_id=user_id,
            event_type=event_type,
            event_id=event.id
        )
        
        return event
    
    async def get_timeline(
        self, 
        user_id: str, 
        limit: int = 100,
        event_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UserEvolutionTimeline:
        """Get user's evolution timeline"""
        
        timeline_key = CACHE_KEYS["USER_TIMELINE"].format(user_id=user_id)
        
        # Get events from Redis
        raw_events = self.redis_client.lrange(timeline_key, 0, limit - 1)
        
        events = []
        for raw_event in raw_events:
            try:
                event_data = json.loads(raw_event)
                # Parse datetime strings back to datetime objects
                event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                if event_data.get('expires_at'):
                    event_data['expires_at'] = datetime.fromisoformat(event_data['expires_at'])
                
                event = TimelineEvent(**event_data)
                
                # Apply filters
                if event_types and event.event_type not in event_types:
                    continue
                if start_date and event.timestamp < start_date:
                    continue
                if end_date and event.timestamp > end_date:
                    continue
                    
                events.append(event)
                
            except Exception as e:
                self.logger.error(
                    "Failed to parse timeline event",
                    error=str(e),
                    raw_event=raw_event
                )
                continue
        
        # Calculate summary stats
        summary_stats = self._calculate_summary_stats(events)
        
        # Determine date range
        date_range = {}
        if events:
            date_range = {
                "start_date": min(event.timestamp for event in events),
                "end_date": max(event.timestamp for event in events)
            }
        
        timeline = UserEvolutionTimeline(
            user_id=user_id,
            timeline_id=generate_id(),
            events=events,
            summary_stats=summary_stats,
            generated_at=utc_now(),
            last_updated=events[0].timestamp if events else utc_now(),
            total_events=len(events),
            date_range=date_range
        )
        
        return timeline
    
    async def add_insight_event(
        self, 
        user_id: str, 
        insight: TwinInsight
    ) -> TimelineEvent:
        """Add insight generation event to timeline"""
        
        return await self.add_event(
            user_id=user_id,
            event_type=TIMELINE_EVENT_TYPES["INSIGHT_GENERATED"],
            title=f"New {insight.insight_type.title()} Insight",
            description=insight.content[:200] + "..." if len(insight.content) > 200 else insight.content,
            data={
                "insight_id": insight.id,
                "insight_type": insight.insight_type,
                "priority": insight.priority,
                "confidence_score": insight.explanation.confidence_score,
                "model_name": insight.explanation.model_metadata.model_name,
                "model_version": insight.explanation.model_metadata.model_version,
                "biological_relevance": insight.explanation.biological_relevance
            },
            impact_score=insight.explanation.confidence_score,
            tags=[insight.insight_type, "ai_generated"]
        )
    
    async def add_analysis_event(
        self,
        user_id: str,
        analysis_type: str,
        analysis_id: str,
        confidence_score: float,
        model_metadata: ModelMetadata
    ) -> TimelineEvent:
        """Add analysis completion event to timeline"""
        
        return await self.add_event(
            user_id=user_id,
            event_type=TIMELINE_EVENT_TYPES["ANALYSIS_COMPLETED"],
            title=f"{analysis_type.replace('_', ' ').title()} Analysis Complete",
            description=f"Comprehensive {analysis_type} analysis completed with {confidence_score:.1%} confidence",
            data={
                "analysis_id": analysis_id,
                "analysis_type": analysis_type,
                "confidence_score": confidence_score,
                "model_name": model_metadata.model_name,
                "model_version": model_metadata.model_version,
                "processing_time": model_metadata.processing_time
            },
            impact_score=confidence_score,
            tags=[analysis_type, "analysis", "completed"]
        )
    
    async def add_feedback_event(
        self,
        user_id: str,
        feedback_type: str,
        quality_score: float,
        related_insight_id: Optional[str] = None
    ) -> TimelineEvent:
        """Add feedback submission event to timeline"""
        
        return await self.add_event(
            user_id=user_id,
            event_type=TIMELINE_EVENT_TYPES["FEEDBACK_RECEIVED"],
            title=f"Feedback Received ({feedback_type})",
            description=f"User provided {feedback_type} feedback with quality score {quality_score:.2f}",
            data={
                "feedback_type": feedback_type,
                "quality_score": quality_score,
                "related_insight_id": related_insight_id
            },
            impact_score=quality_score,
            tags=[feedback_type, "feedback", "user_input"]
        )
    
    async def add_metric_update_event(
        self,
        user_id: str,
        metric_name: str,
        old_value: Optional[float],
        new_value: float,
        improvement: Optional[float] = None
    ) -> TimelineEvent:
        """Add metric update event to timeline"""
        
        # Calculate improvement if old value exists
        if old_value is not None and improvement is None:
            improvement = ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
        
        title = f"{metric_name.replace('_', ' ').title()} Updated"
        if improvement is not None:
            if improvement > 0:
                title += f" (+{improvement:.1f}%)"
            elif improvement < 0:
                title += f" ({improvement:.1f}%)"
        
        description = f"{metric_name} updated"
        if old_value is not None:
            description += f" from {old_value} to {new_value}"
        else:
            description += f" to {new_value}"
        
        return await self.add_event(
            user_id=user_id,
            event_type=TIMELINE_EVENT_TYPES["METRIC_UPDATED"],
            title=title,
            description=description,
            data={
                "metric_name": metric_name,
                "old_value": old_value,
                "new_value": new_value,
                "improvement_percentage": improvement
            },
            impact_score=min(abs(improvement) / 100, 1.0) if improvement is not None else 0.5,
            tags=["metrics", "quantified_self", metric_name]
        )
    
    def _calculate_summary_stats(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Calculate summary statistics for timeline"""
        
        if not events:
            return {}
        
        # Count by event type
        event_type_counts = {}
        for event in events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        # Count by source service
        service_counts = {}
        for event in events:
            service_counts[event.source_service] = service_counts.get(event.source_service, 0) + 1
        
        # Calculate impact metrics
        impact_scores = [e.impact_score for e in events if e.impact_score is not None]
        avg_impact = sum(impact_scores) / len(impact_scores) if impact_scores else 0
        
        # Recent activity (last 7 days)
        recent_cutoff = utc_now() - timedelta(days=7)
        recent_events = [e for e in events if e.timestamp >= recent_cutoff]
        
        return {
            "total_events": len(events),
            "event_type_distribution": event_type_counts,
            "service_distribution": service_counts,
            "average_impact_score": avg_impact,
            "recent_activity": {
                "last_7_days": len(recent_events),
                "daily_average": len(recent_events) / 7
            },
            "timeline_span_days": (events[0].timestamp - events[-1].timestamp).days if len(events) > 1 else 0
        }
    
    async def export_timeline_json(self, user_id: str) -> Dict[str, Any]:
        """Export timeline in structured JSON format for UI"""
        
        timeline = await self.get_timeline(user_id)
        
        # Structure for timeline visualization
        timeline_json = {
            "user_id": user_id,
            "timeline_metadata": {
                "generated_at": timeline.generated_at.isoformat(),
                "total_events": timeline.total_events,
                "date_range": {
                    "start": timeline.date_range.get("start_date", "").isoformat() if timeline.date_range.get("start_date") else None,
                    "end": timeline.date_range.get("end_date", "").isoformat() if timeline.date_range.get("end_date") else None
                }
            },
            "summary_statistics": timeline.summary_stats,
            "timeline_events": [
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "title": event.title,
                    "description": event.description,
                    "impact_score": event.impact_score,
                    "tags": event.tags,
                    "source_service": event.source_service,
                    "data": event.data
                }
                for event in timeline.events
            ]
        }
        
        return timeline_json
    
    async def cleanup_old_events(self, user_id: str, retention_days: int = 365):
        """Clean up old timeline events beyond retention period"""
        
        cutoff_date = utc_now() - timedelta(days=retention_days)
        timeline_key = CACHE_KEYS["USER_TIMELINE"].format(user_id=user_id)
        
        # Get all events
        raw_events = self.redis_client.lrange(timeline_key, 0, -1)
        
        # Filter events within retention period
        retained_events = []
        for raw_event in raw_events:
            try:
                event_data = json.loads(raw_event)
                event_timestamp = datetime.fromisoformat(event_data['timestamp'])
                
                if event_timestamp >= cutoff_date:
                    retained_events.append(raw_event)
                    
            except Exception:
                continue
        
        # Replace timeline with filtered events
        if len(retained_events) < len(raw_events):
            pipe = self.redis_client.pipeline()
            pipe.delete(timeline_key)
            if retained_events:
                pipe.lpush(timeline_key, *retained_events)
                pipe.expire(timeline_key, 31536000)
            pipe.execute()
            
            self.logger.info(
                "Timeline cleanup completed",
                user_id=user_id,
                original_count=len(raw_events),
                retained_count=len(retained_events),
                removed_count=len(raw_events) - len(retained_events)
            )
