"""
EvoHuman.AI Feedback Adaptation System
Handles dynamic adaptation based on user feedback
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import redis
import json
import faiss
import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("feedback_adaptation")

class FeedbackSignal(BaseModel):
    """Structured feedback from user"""
    user_id: str
    intervention_id: str
    feedback_type: str  # positive, negative, neutral
    feedback_text: str
    effectiveness_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, any] = Field(default_factory=dict)

class AdaptationResult(BaseModel):
    """Result of adaptation process"""
    adaptation_type: str
    confidence_delta: float
    modified_weights: Dict[str, float]
    metadata: Dict[str, any] = Field(default_factory=dict)

class FeedbackAdapter:
    def __init__(self, redis_client: redis.Redis):
        """Initialize feedback adaptation system"""
        self.redis = redis_client
        self.vector_dim = 128  # Dimension of intervention embeddings
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self._load_index()

    def _get_redis_key(self, key_type: str, user_id: Optional[str] = None) -> str:
        """Generate Redis key"""
        base = f"feedback:{key_type}"
        return f"{base}:{user_id}" if user_id else base

    def _load_index(self):
        """Load FAISS index from Redis"""
        index_data = self.redis.get(self._get_redis_key("faiss_index"))
        if index_data:
            self.index = faiss.deserialize_index(index_data)

    def _save_index(self):
        """Save FAISS index to Redis"""
        index_data = faiss.serialize_index(self.index)
        self.redis.set(self._get_redis_key("faiss_index"), index_data)

    def _get_intervention_embedding(
        self,
        intervention: str,
        context: Dict[str, any]
    ) -> np.ndarray:
        """Generate embedding for intervention"""
        # TODO: Replace with actual embedding logic
        # Currently using random embedding for demonstration
        embedding = np.random.rand(self.vector_dim).astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _update_user_weights(
        self,
        user_id: str,
        intervention_type: str,
        effectiveness: float
    ):
        """Update intervention weights for user"""
        weights_key = self._get_redis_key("weights", user_id)
        weights = self.redis.get(weights_key)
        
        if weights:
            weights = json.loads(weights)
        else:
            weights = {}

        # Update weight for intervention type
        current_weight = weights.get(intervention_type, 1.0)
        new_weight = current_weight * (0.8 + 0.4 * effectiveness)
        weights[intervention_type] = max(0.1, min(2.0, new_weight))

        self.redis.set(weights_key, json.dumps(weights))
        return weights

    def process_feedback(
        self,
        feedback: FeedbackSignal
    ) -> AdaptationResult:
        """Process user feedback and adapt system"""
        try:
            # Generate embedding for the intervention
            embedding = self._get_intervention_embedding(
                feedback.intervention_id,
                feedback.context
            )

            # Update FAISS index
            self.index.add(embedding.reshape(1, -1))
            self._save_index()

            # Calculate adaptation based on feedback
            effectiveness = feedback.effectiveness_score
            confidence_delta = (effectiveness - 0.5) * 0.2

            # Update intervention weights
            modified_weights = self._update_user_weights(
                feedback.user_id,
                feedback.context.get("intervention_type", "general"),
                effectiveness
            )

            # Store feedback in Redis
            feedback_key = self._get_redis_key(
                "history",
                feedback.user_id
            )
            self.redis.lpush(
                feedback_key,
                json.dumps(feedback.dict())
            )
            self.redis.ltrim(feedback_key, 0, 999)  # Keep last 1000 feedbacks

            return AdaptationResult(
                adaptation_type="weight_adjustment",
                confidence_delta=confidence_delta,
                modified_weights=modified_weights,
                metadata={
                    "feedback_processed": True,
                    "index_updated": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(
                "Feedback processing failed",
                error=str(e),
                user_id=feedback.user_id
            )
            raise

    def get_similar_interventions(
        self,
        intervention: str,
        context: Dict[str, any],
        k: int = 5
    ) -> List[Tuple[float, Dict[str, any]]]:
        """Find similar successful interventions"""
        embedding = self._get_intervention_embedding(intervention, context)
        
        # Search FAISS index
        D, I = self.index.search(embedding.reshape(1, -1), k)
        
        # Convert to list of (distance, metadata) tuples
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1:  # Valid index
                meta_key = self._get_redis_key(f"meta:{idx}")
                meta = self.redis.get(meta_key)
                if meta:
                    results.append((
                        float(dist),
                        json.loads(meta)
                    ))

        return results

    def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, float]:
        """Get user's intervention preferences"""
        weights_key = self._get_redis_key("weights", user_id)
        weights = self.redis.get(weights_key)
        return json.loads(weights) if weights else {}