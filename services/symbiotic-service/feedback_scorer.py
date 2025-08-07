"""
Feedback Quality Scoring System for SymbioticAIS
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import redis
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog
import re

from shared.models import FeedbackQualityScore, EnhancedFeedback
from shared.constants import FEEDBACK_QUALITY, CACHE_KEYS
from shared.utils import generate_id, utc_now


class FeedbackQualityScorer:
    """Assess and score quality of user feedback for learning loops"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.logger = structlog.get_logger("feedback_scorer")
        self.redis_client = redis_client or redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        
        # Sentiment keywords for basic sentiment analysis
        self.positive_keywords = {
            'excellent', 'amazing', 'great', 'good', 'helpful', 'effective', 
            'useful', 'beneficial', 'positive', 'improved', 'better', 'love',
            'fantastic', 'wonderful', 'perfect', 'satisfied', 'happy'
        }
        
        self.negative_keywords = {
            'terrible', 'awful', 'bad', 'poor', 'useless', 'ineffective',
            'unhelpful', 'worse', 'negative', 'declined', 'hate', 'horrible',
            'disappointing', 'frustrated', 'unsatisfied', 'angry'
        }
        
        self.uncertainty_keywords = {
            'maybe', 'perhaps', 'might', 'could', 'possibly', 'unsure',
            'uncertain', 'confused', 'not sure', 'unclear', 'ambiguous'
        }
    
    def score_feedback(
        self, 
        user_id: str, 
        feedback_content: Dict[str, Any],
        feedback_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FeedbackQualityScore:
        """Generate comprehensive quality score for feedback"""
        
        # Get user's feedback history for consistency analysis
        feedback_history = self._get_feedback_history(user_id)
        
        # Calculate individual quality components
        consistency_score = self._calculate_consistency_score(
            feedback_content, feedback_history, feedback_type
        )
        
        relevance_score = self._calculate_relevance_score(
            feedback_content, feedback_type, context
        )
        
        sentiment_score = self._analyze_sentiment(feedback_content)
        
        # Calculate reliability factors
        reliability_factors = self._assess_reliability_factors(
            feedback_content, feedback_history, feedback_type
        )
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            consistency_score, relevance_score, abs(sentiment_score), reliability_factors
        )
        
        quality_score = FeedbackQualityScore(
            score=overall_score,
            consistency_score=consistency_score,
            relevance_score=relevance_score,
            sentiment_score=sentiment_score,
            reliability_factors=reliability_factors,
            timestamp=utc_now()
        )
        
        # Update feedback history
        self._update_feedback_history(user_id, feedback_content, quality_score)
        
        self.logger.info(
            "Feedback quality scored",
            user_id=user_id,
            feedback_type=feedback_type,
            overall_score=overall_score,
            consistency_score=consistency_score,
            relevance_score=relevance_score
        )
        
        return quality_score
    
    def _calculate_consistency_score(
        self, 
        feedback_content: Dict[str, Any], 
        history: List[Dict[str, Any]], 
        feedback_type: str
    ) -> float:
        """Calculate consistency with user's historical feedback patterns"""
        
        if len(history) < FEEDBACK_QUALITY["MIN_CONSISTENCY_SAMPLES"]:
            return 0.5  # Neutral score for insufficient history
        
        # Filter history for same feedback type
        relevant_history = [
            item for item in history 
            if item.get('feedback_type') == feedback_type
        ]
        
        if not relevant_history:
            return 0.5  # Neutral score for no relevant history
        
        consistency_factors = []
        
        # Rating consistency (for rating-type feedback)
        if feedback_type == 'rating' and 'rating' in feedback_content:
            current_rating = feedback_content['rating']
            historical_ratings = [
                item['content'].get('rating', 0) 
                for item in relevant_history 
                if 'rating' in item.get('content', {})
            ]
            
            if historical_ratings:
                # Calculate variance in ratings
                rating_variance = np.var(historical_ratings + [current_rating])
                # Lower variance = higher consistency
                rating_consistency = max(0.0, 1.0 - (rating_variance / 25.0))  # Normalize to 0-1
                consistency_factors.append(rating_consistency)
        
        # Response length consistency (for text feedback)
        if 'text' in feedback_content:
            current_length = len(feedback_content['text'].split())
            historical_lengths = [
                len(item['content'].get('text', '').split()) 
                for item in relevant_history 
                if 'text' in item.get('content', {})
            ]
            
            if historical_lengths:
                avg_length = np.mean(historical_lengths)
                if avg_length > 0:
                    length_ratio = min(current_length, avg_length) / max(current_length, avg_length)
                    consistency_factors.append(length_ratio)
        
        # Response timing consistency
        current_time = utc_now()
        response_times = []
        
        for item in relevant_history[-5:]:  # Last 5 responses
            if 'timestamp' in item:
                try:
                    hist_time = datetime.fromisoformat(item['timestamp'])
                    # Look at hour of day for timing patterns
                    response_times.append(hist_time.hour)
                except:
                    continue
        
        if response_times:
            current_hour = current_time.hour
            # Check if current response time is within typical range
            time_variance = np.var(response_times + [current_hour])
            time_consistency = max(0.0, 1.0 - (time_variance / 144.0))  # Normalize
            consistency_factors.append(time_consistency)
        
        # Overall consistency score
        if consistency_factors:
            return np.mean(consistency_factors)
        else:
            return 0.5
    
    def _calculate_relevance_score(
        self, 
        feedback_content: Dict[str, Any], 
        feedback_type: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate relevance of feedback to the context"""
        
        relevance_factors = []
        
        # Content completeness
        completeness_score = self._assess_completeness(feedback_content, feedback_type)
        relevance_factors.append(completeness_score)
        
        # Context alignment (if context provided)
        if context:
            context_alignment = self._assess_context_alignment(feedback_content, context)
            relevance_factors.append(context_alignment)
        
        # Specificity of feedback
        specificity_score = self._assess_specificity(feedback_content, feedback_type)
        relevance_factors.append(specificity_score)
        
        return np.mean(relevance_factors) if relevance_factors else 0.5
    
    def _assess_completeness(self, feedback_content: Dict[str, Any], feedback_type: str) -> float:
        """Assess completeness of feedback based on type"""
        
        if feedback_type == 'rating':
            # Rating should have numerical score
            if 'rating' in feedback_content:
                score = 1.0
                # Bonus for additional context
                if 'text' in feedback_content and len(feedback_content['text'].strip()) > 10:
                    score = min(1.0, score + 0.2)
                return score
            return 0.2
        
        elif feedback_type == 'text':
            # Text feedback should have meaningful content
            if 'text' in feedback_content:
                text_length = len(feedback_content['text'].strip())
                if text_length > 100:
                    return 1.0
                elif text_length > 50:
                    return 0.8
                elif text_length > 20:
                    return 0.6
                elif text_length > 5:
                    return 0.4
                else:
                    return 0.2
            return 0.1
        
        elif feedback_type == 'behavioral':
            # Behavioral feedback should have metrics
            expected_fields = ['activity_level', 'compliance', 'effectiveness']
            present_fields = sum(1 for field in expected_fields if field in feedback_content)
            return present_fields / len(expected_fields)
        
        elif feedback_type == 'physiological':
            # Physiological feedback should have measurements
            if 'measurements' in feedback_content and feedback_content['measurements']:
                return 1.0
            return 0.3
        
        return 0.5  # Default for unknown types
    
    def _assess_context_alignment(self, feedback_content: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Assess how well feedback aligns with the given context"""
        
        alignment_score = 0.5  # Base score
        
        # Check if feedback references the intervention/insight being evaluated
        if 'intervention_id' in context and 'text' in feedback_content:
            text = feedback_content['text'].lower()
            # Look for references to the specific intervention
            if any(word in text for word in ['intervention', 'recommendation', 'suggestion', 'advice']):
                alignment_score += 0.2
        
        # Check temporal alignment
        if 'expected_timeframe' in context:
            # Feedback should be within reasonable timeframe
            # This would require timestamp comparison logic
            alignment_score += 0.1
        
        # Check if feedback addresses the right domain
        if 'domain' in context:
            domain = context['domain'].lower()
            if 'text' in feedback_content:
                text = feedback_content['text'].lower()
                if domain in text or any(related_word in text for related_word in self._get_domain_keywords(domain)):
                    alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    def _assess_specificity(self, feedback_content: Dict[str, Any], feedback_type: str) -> float:
        """Assess specificity and detail level of feedback"""
        
        if feedback_type == 'text' and 'text' in feedback_content:
            text = feedback_content['text']
            
            # Count specific details
            specificity_indicators = 0
            
            # Numbers or measurements
            if re.search(r'\d+', text):
                specificity_indicators += 1
            
            # Time references
            if re.search(r'\b(day|hour|week|month|minute|time|after|before|during)\b', text.lower()):
                specificity_indicators += 1
            
            # Degree qualifiers
            if re.search(r'\b(very|quite|somewhat|slightly|extremely|moderately)\b', text.lower()):
                specificity_indicators += 1
            
            # Specific symptoms or effects
            if re.search(r'\b(feel|felt|noticed|experienced|improved|worse|better|change)\b', text.lower()):
                specificity_indicators += 1
            
            # Comparison words
            if re.search(r'\b(more|less|compared|than|versus|before|after)\b', text.lower()):
                specificity_indicators += 1
            
            # Convert count to score (0-1)
            return min(1.0, specificity_indicators / 5.0)
        
        elif feedback_type == 'rating':
            # Ratings are inherently specific if provided
            return 0.8 if 'rating' in feedback_content else 0.2
        
        return 0.5
    
    def _analyze_sentiment(self, feedback_content: Dict[str, Any]) -> float:
        """Analyze sentiment of feedback (-1 to 1)"""
        
        text_content = ""
        
        # Extract text for sentiment analysis
        if 'text' in feedback_content:
            text_content = feedback_content['text'].lower()
        
        # Add rating context
        if 'rating' in feedback_content:
            rating = feedback_content['rating']
            if isinstance(rating, (int, float)):
                if rating >= 4:
                    text_content += " positive rating"
                elif rating <= 2:
                    text_content += " negative rating"
                else:
                    text_content += " neutral rating"
        
        if not text_content:
            return 0.0  # Neutral sentiment
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in self.positive_keywords if word in text_content)
        negative_count = sum(1 for word in self.negative_keywords if word in text_content)
        uncertainty_count = sum(1 for word in self.uncertainty_keywords if word in text_content)
        
        # Calculate base sentiment
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment = 0.0
        else:
            sentiment = (positive_count - negative_count) / total_sentiment_words
        
        # Adjust for uncertainty
        if uncertainty_count > 0:
            uncertainty_penalty = min(0.3, uncertainty_count * 0.1)
            sentiment = sentiment * (1 - uncertainty_penalty)
        
        return max(-1.0, min(1.0, sentiment))
    
    def _assess_reliability_factors(
        self, 
        feedback_content: Dict[str, Any], 
        history: List[Dict[str, Any]], 
        feedback_type: str
    ) -> List[str]:
        """Identify factors that contribute to or detract from reliability"""
        
        factors = []
        
        # Historical engagement
        if len(history) >= 10:
            factors.append("consistent_engagement")
        elif len(history) >= 5:
            factors.append("moderate_engagement")
        else:
            factors.append("limited_history")
        
        # Response detail level
        if feedback_type == 'text' and 'text' in feedback_content:
            text_length = len(feedback_content['text'].strip())
            if text_length > 100:
                factors.append("detailed_response")
            elif text_length < 20:
                factors.append("brief_response")
        
        # Response timing
        if history:
            # Check if response is unusually fast (might indicate rushed feedback)
            # This would require timestamp comparison with intervention time
            pass
        
        # Sentiment consistency
        if len(history) >= 3:
            recent_sentiments = []
            for item in history[-3:]:
                if 'quality_score' in item and 'sentiment_score' in item['quality_score']:
                    recent_sentiments.append(item['quality_score']['sentiment_score'])
            
            if recent_sentiments:
                current_sentiment = self._analyze_sentiment(feedback_content)
                sentiment_variance = np.var(recent_sentiments + [current_sentiment])
                
                if sentiment_variance < 0.1:
                    factors.append("consistent_sentiment")
                elif sentiment_variance > 0.5:
                    factors.append("variable_sentiment")
        
        # Content specificity
        specificity_score = self._assess_specificity(feedback_content, feedback_type)
        if specificity_score > 0.7:
            factors.append("high_specificity")
        elif specificity_score < 0.3:
            factors.append("low_specificity")
        
        return factors
    
    def _calculate_overall_score(
        self, 
        consistency: float, 
        relevance: float, 
        sentiment_strength: float, 
        reliability_factors: List[str]
    ) -> float:
        """Calculate overall quality score from components"""
        
        # Base score from primary components
        base_score = (consistency * 0.4 + relevance * 0.4 + sentiment_strength * 0.2)
        
        # Reliability adjustments
        reliability_bonus = 0.0
        reliability_penalty = 0.0
        
        positive_factors = {'consistent_engagement', 'detailed_response', 'consistent_sentiment', 'high_specificity'}
        negative_factors = {'limited_history', 'brief_response', 'variable_sentiment', 'low_specificity'}
        
        for factor in reliability_factors:
            if factor in positive_factors:
                reliability_bonus += 0.05
            elif factor in negative_factors:
                reliability_penalty += 0.05
        
        # Apply adjustments
        final_score = base_score + reliability_bonus - reliability_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _get_feedback_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's feedback history from Redis"""
        
        history_key = CACHE_KEYS["USER_FEEDBACK_HISTORY"].format(user_id=user_id)
        
        try:
            history_data = self.redis_client.get(history_key)
            if history_data:
                return json.loads(history_data)
        except Exception as e:
            self.logger.error("Failed to get feedback history", user_id=user_id, error=str(e))
        
        return []
    
    def _update_feedback_history(
        self, 
        user_id: str, 
        feedback_content: Dict[str, Any], 
        quality_score: FeedbackQualityScore
    ):
        """Update user's feedback history"""
        
        history_key = CACHE_KEYS["USER_FEEDBACK_HISTORY"].format(user_id=user_id)
        
        # Get current history
        history = self._get_feedback_history(user_id)
        
        # Add new feedback entry
        new_entry = {
            "timestamp": utc_now().isoformat(),
            "content": feedback_content,
            "quality_score": quality_score.model_dump()
        }
        
        history.append(new_entry)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        try:
            self.redis_client.setex(
                history_key,
                31536000,  # 1 year expiration
                json.dumps(history)
            )
        except Exception as e:
            self.logger.error("Failed to update feedback history", user_id=user_id, error=str(e))
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords related to a specific domain"""
        
        domain_keywords = {
            'fitness': ['exercise', 'workout', 'training', 'physical', 'strength', 'cardio'],
            'nutrition': ['food', 'diet', 'eating', 'nutrition', 'meal', 'calories'],
            'sleep': ['sleep', 'rest', 'tired', 'energy', 'bed', 'night'],
            'stress': ['stress', 'anxiety', 'tension', 'pressure', 'calm', 'relax'],
            'cognitive': ['focus', 'memory', 'concentration', 'thinking', 'mental', 'brain']
        }
        
        return domain_keywords.get(domain, [])
