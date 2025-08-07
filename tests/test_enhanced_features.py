"""
Basic tests for the enhanced EvoHuman.AI features
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import asyncio
from datetime import datetime, timedelta

from shared.models import (
    ModelMetadata, ExplanationData, TwinInsight, 
    FeedbackQualityScore, EnhancedFeedback, TimelineEvent,
    UserEvolutionTimeline, NotificationMessage, BatchFeedbackRequest
)
from shared.constants import MODEL_VERSIONS, RISK_LEVELS, FEEDBACK_QUALITY
from shared.utils import generate_id, utc_now

from services.bio_twin.timeline_engine import TimelineEngine
from services.bio_twin.explainability_engine import ExplainabilityEngine
from services.symbiotic_service.feedback_scorer import FeedbackQualityScorer


class TestExplainabilityEngine:
    """Test the explainability engine"""
    
    @pytest.fixture
    def explainability_engine(self):
        return ExplainabilityEngine()
    
    @pytest.fixture
    def model_metadata(self):
        return ModelMetadata(
            model_name="test-model",
            model_version="1.0.0",
            timestamp=utc_now(),
            processing_time=0.5
        )
    
    def test_create_explanation(self, explainability_engine, model_metadata):
        """Test basic explanation creation"""
        explanation = explainability_engine.create_explanation(
            insight_content="Test mutation A123V shows stabilizing effect",
            confidence_score=0.85,
            model_metadata=model_metadata,
            contributing_factors=["protein_structure", "evolutionary_data"],
            biological_context="genetic"
        )
        
        assert isinstance(explanation, ExplanationData)
        assert explanation.confidence_score == 0.85
        assert explanation.risk_level in ["low", "medium", "high"]
        assert "genetic" in explanation.biological_relevance.lower()
    
    def test_explain_mutation_insight(self, explainability_engine, model_metadata):
        """Test mutation-specific explanation"""
        protein_context = {
            "sequence": "MKLLVTACSFCFLAGHVGLLTTTGVTMFLTLQGRQHPPRCVP",
            "active_site_proximity": True,
            "conservation_score": 0.92
        }
        
        explanation = explainability_engine.explain_mutation_insight(
            mutation="A123V",
            stability_change=1.2,
            confidence_score=0.78,
            protein_context=protein_context,
            model_metadata=model_metadata
        )
        
        assert "stabilizing" in explanation.reason.lower()
        assert explanation.confidence_score == 0.78
        assert len(explanation.contributing_factors) > 0


class TestTimelineEngine:
    """Test the timeline engine"""
    
    @pytest.fixture
    async def timeline_engine(self):
        # Mock Redis client for testing
        class MockRedis:
            def __init__(self):
                self.data = {}
            
            def lpush(self, key, value):
                if key not in self.data:
                    self.data[key] = []
                self.data[key].insert(0, value)
            
            def ltrim(self, key, start, end):
                if key in self.data:
                    self.data[key] = self.data[key][start:end+1]
            
            def expire(self, key, seconds):
                pass  # Mock implementation
            
            def lrange(self, key, start, end):
                if key not in self.data:
                    return []
                if end == -1:
                    return self.data[key][start:]
                return self.data[key][start:end+1]
        
        engine = TimelineEngine()
        engine.redis_client = MockRedis()
        return engine
    
    @pytest.mark.asyncio
    async def test_add_event(self, timeline_engine):
        """Test adding timeline event"""
        event = await timeline_engine.add_event(
            user_id="test-user-123",
            event_type="insight_generated",
            title="New Genetic Insight",
            description="Found beneficial mutation A123V",
            data={"mutation": "A123V", "confidence": 0.85},
            impact_score=0.8,
            tags=["genetics", "mutation"]
        )
        
        assert isinstance(event, TimelineEvent)
        assert event.user_id == "test-user-123"
        assert event.event_type == "insight_generated"
        assert event.impact_score == 0.8
    
    @pytest.mark.asyncio
    async def test_get_timeline(self, timeline_engine):
        """Test getting user timeline"""
        # Add a few events first
        await timeline_engine.add_event(
            user_id="test-user-123",
            event_type="insight_generated",
            title="Insight 1",
            description="First insight",
            data={"test": "data1"}
        )
        
        await timeline_engine.add_event(
            user_id="test-user-123",
            event_type="metric_updated",
            title="Metric Update",
            description="Updated biomarker",
            data={"metric": "glucose", "value": 95}
        )
        
        timeline = await timeline_engine.get_timeline("test-user-123", limit=10)
        
        assert isinstance(timeline, UserEvolutionTimeline)
        assert timeline.user_id == "test-user-123"
        assert len(timeline.events) == 2
        assert timeline.total_events == 2


class TestFeedbackQualityScorer:
    """Test the feedback quality scoring system"""
    
    @pytest.fixture
    def feedback_scorer(self):
        # Mock Redis client for testing
        class MockRedis:
            def __init__(self):
                self.data = {}
            
            def get(self, key):
                return self.data.get(key)
            
            def setex(self, key, ttl, value):
                self.data[key] = value
        
        scorer = FeedbackQualityScorer()
        scorer.redis_client = MockRedis()
        return scorer
    
    def test_score_text_feedback(self, feedback_scorer):
        """Test scoring text feedback"""
        feedback_content = {
            "text": "This intervention helped me feel much better and more energetic. I noticed improvements after 3 days."
        }
        
        quality_score = feedback_scorer.score_feedback(
            user_id="test-user-123",
            feedback_content=feedback_content,
            feedback_type="text"
        )
        
        assert isinstance(quality_score, FeedbackQualityScore)
        assert 0.0 <= quality_score.score <= 1.0
        assert 0.0 <= quality_score.consistency_score <= 1.0
        assert 0.0 <= quality_score.relevance_score <= 1.0
        assert -1.0 <= quality_score.sentiment_score <= 1.0
    
    def test_score_rating_feedback(self, feedback_scorer):
        """Test scoring rating feedback"""
        feedback_content = {
            "rating": 4,
            "text": "Good results"
        }
        
        quality_score = feedback_scorer.score_feedback(
            user_id="test-user-123",
            feedback_content=feedback_content,
            feedback_type="rating"
        )
        
        assert quality_score.score > 0.5  # Should be decent quality
        assert quality_score.sentiment_score > 0  # Should be positive
    
    def test_analyze_sentiment(self, feedback_scorer):
        """Test sentiment analysis"""
        positive_feedback = {"text": "This is amazing and wonderful! I love it!"}
        negative_feedback = {"text": "This is terrible and awful. I hate it."}
        neutral_feedback = {"text": "This is okay. Nothing special."}
        
        positive_sentiment = feedback_scorer._analyze_sentiment(positive_feedback)
        negative_sentiment = feedback_scorer._analyze_sentiment(negative_feedback)
        neutral_sentiment = feedback_scorer._analyze_sentiment(neutral_feedback)
        
        assert positive_sentiment > 0
        assert negative_sentiment < 0
        assert abs(neutral_sentiment) < abs(positive_sentiment)


class TestSharedModels:
    """Test the shared data models"""
    
    def test_model_metadata(self):
        """Test ModelMetadata model"""
        metadata = ModelMetadata(
            model_name="esm3",
            model_version="1.2.0",
            timestamp=utc_now(),
            processing_time=2.5
        )
        
        assert metadata.model_name == "esm3"
        assert metadata.model_version == "1.2.0"
        assert metadata.processing_time == 2.5
    
    def test_explanation_data(self):
        """Test ExplanationData model"""
        model_metadata = ModelMetadata(
            model_name="test-model",
            model_version="1.0.0",
            timestamp=utc_now()
        )
        
        explanation = ExplanationData(
            reason="Test explanation",
            model_metadata=model_metadata,
            confidence_score=0.85,
            risk_level="low",
            biological_relevance="High relevance",
            contributing_factors=["factor1", "factor2"]
        )
        
        assert explanation.confidence_score == 0.85
        assert explanation.risk_level == "low"
        assert len(explanation.contributing_factors) == 2
    
    def test_twin_insight(self):
        """Test TwinInsight model"""
        model_metadata = ModelMetadata(
            model_name="bio-twin",
            model_version="2.1.0",
            timestamp=utc_now()
        )
        
        explanation = ExplanationData(
            reason="Genetic analysis reveals beneficial mutation",
            model_metadata=model_metadata,
            confidence_score=0.92,
            risk_level="low",
            biological_relevance="Strong genetic evidence",
            contributing_factors=["sequence_analysis", "conservation_data"]
        )
        
        insight = TwinInsight(
            id="insight-123",
            user_id="user-123",
            insight_type="mutation",
            content="A123V mutation shows 15% improvement in protein stability",
            priority=8,
            explanation=explanation,
            created_at=utc_now()
        )
        
        assert insight.insight_type == "mutation"
        assert insight.priority == 8
        assert insight.explanation.confidence_score == 0.92
    
    def test_batch_feedback_request(self):
        """Test BatchFeedbackRequest model"""
        batch_request = BatchFeedbackRequest(
            user_id="user-123",
            feedback_entries=[
                {
                    "feedback_type": "rating",
                    "content": {"rating": 5, "text": "Excellent!"}
                },
                {
                    "feedback_type": "text",
                    "content": {"text": "Very helpful intervention"}
                }
            ],
            priority=7
        )
        
        assert batch_request.user_id == "user-123"
        assert len(batch_request.feedback_entries) == 2
        assert batch_request.priority == 7
    
    def test_notification_message(self):
        """Test NotificationMessage model"""
        notification = NotificationMessage(
            id="notif-123",
            user_id="user-123",
            message_type="insight",
            title="New Genetic Insight Available",
            content="We've discovered a beneficial mutation in your profile",
            priority="high",
            created_at=utc_now(),
            expires_at=utc_now() + timedelta(days=7),
            actions=[
                {"type": "view", "label": "View Details"},
                {"type": "feedback", "label": "Provide Feedback"}
            ]
        )
        
        assert notification.message_type == "insight"
        assert notification.priority == "high"
        assert len(notification.actions) == 2
        assert not notification.read  # Default should be False


if __name__ == "__main__":
    # Run basic tests
    print("Running enhanced features tests...")
    
    # Test ExplanabilityEngine
    explainability_engine = ExplainabilityEngine()
    model_metadata = ModelMetadata(
        model_name="test-model",
        model_version="1.0.0",
        timestamp=utc_now()
    )
    
    explanation = explainability_engine.create_explanation(
        insight_content="Test insight",
        confidence_score=0.85,
        model_metadata=model_metadata,
        contributing_factors=["test_factor"],
        biological_context="genetic"
    )
    
    print(f"✅ ExplanabilityEngine: Created explanation with confidence {explanation.confidence_score}")
    
    # Test models
    insight = TwinInsight(
        id=generate_id(),
        user_id="test-user",
        insight_type="mutation",
        content="Test insight content",
        priority=5,
        explanation=explanation,
        created_at=utc_now()
    )
    
    print(f"✅ TwinInsight: Created insight with ID {insight.id}")
    
    # Test batch feedback
    batch_request = BatchFeedbackRequest(
        user_id="test-user",
        feedback_entries=[
            {"content": {"text": "Great!"}, "feedback_type": "text"}
        ]
    )
    
    print(f"✅ BatchFeedbackRequest: Created with {len(batch_request.feedback_entries)} entries")
    
    print("All basic tests passed! 🎉")
    print("\nTo run full pytest suite:")
    print("pytest tests/test_enhanced_features.py -v")
