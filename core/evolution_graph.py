"""
EvoHuman.AI Evolution Graph System
Handles user evolution timeline tracking and adaptation
"""
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
import networkx as nx
from pydantic import BaseModel, Field
import json
import redis
from dataclasses import dataclass, asdict
import structlog

# Configure logging
logger = structlog.get_logger("evolution_graph")

class BiometricState(BaseModel):
    """Represents user's biometric measurements"""
    energy_level: float = Field(0.0, ge=0.0, le=1.0)
    focus_score: float = Field(0.0, ge=0.0, le=1.0)
    stress_level: float = Field(0.0, ge=0.0, le=1.0)
    sleep_quality: float = Field(0.0, ge=0.0, le=1.0)
    inflammation_markers: float = Field(0.0, ge=0.0, le=1.0)
    metabolic_score: float = Field(0.0, ge=0.0, le=1.0)

class BioCognitiveState(BaseModel):
    """Complete biological and cognitive state"""
    timestamp: datetime
    biometrics: BiometricState
    cognitive_state: str
    longevity_score: float = Field(0.0, ge=0.0, le=100.0)
    active_interventions: List[str] = Field(default_factory=list)

class EvolutionNode(BaseModel):
    """Represents a point in user's evolution timeline"""
    node_id: str
    timestamp: datetime
    bio_score: float
    cognitive_state: str
    intervention: str
    result: Optional[str]
    confidence: float
    feedback: Optional[str]
    biometrics: BiometricState

class EvolutionEdge(BaseModel):
    """Represents a transition in user's evolution"""
    from_node: str
    to_node: str
    edge_type: str  # "time" or "feedback"
    weight: float
    metadata: Dict[str, any] = Field(default_factory=dict)

class UserEvolutionGraph:
    def __init__(self, user_id: str, redis_client: redis.Redis):
        self.user_id = user_id
        self.redis = redis_client
        self.graph = nx.DiGraph()
        self._load_graph()

    def _get_redis_key(self, key_type: str) -> str:
        """Generate Redis key for different data types"""
        return f"evolution:{self.user_id}:{key_type}"

    def _load_graph(self):
        """Load graph from Redis"""
        nodes = self.redis.get(self._get_redis_key("nodes"))
        edges = self.redis.get(self._get_redis_key("edges"))

        if nodes:
            nodes = json.loads(nodes)
            for node_data in nodes:
                node = EvolutionNode(**node_data)
                self.graph.add_node(node.node_id, **node.dict())

        if edges:
            edges = json.loads(edges)
            for edge_data in edges:
                edge = EvolutionEdge(**edge_data)
                self.graph.add_edge(
                    edge.from_node,
                    edge.to_node,
                    **edge.dict()
                )

    def _save_graph(self):
        """Save graph to Redis"""
        nodes = [
            EvolutionNode(**node_data).dict()
            for _, node_data in self.graph.nodes(data=True)
        ]
        
        edges = [
            EvolutionEdge(
                from_node=u,
                to_node=v,
                **edge_data
            ).dict()
            for u, v, edge_data in self.graph.edges(data=True)
        ]

        self.redis.set(
            self._get_redis_key("nodes"),
            json.dumps(nodes)
        )
        self.redis.set(
            self._get_redis_key("edges"),
            json.dumps(edges)
        )

    def add_evolution_point(
        self,
        state: BioCognitiveState,
        intervention: str,
        confidence: float
    ) -> str:
        """Add new evolution point to graph"""
        node_id = f"{self.user_id}_{int(datetime.utcnow().timestamp())}"
        
        node = EvolutionNode(
            node_id=node_id,
            timestamp=state.timestamp,
            bio_score=state.longevity_score,
            cognitive_state=state.cognitive_state,
            intervention=intervention,
            result=None,
            confidence=confidence,
            feedback=None,
            biometrics=state.biometrics
        )

        self.graph.add_node(node_id, **node.dict())

        # Add edge from most recent node if exists
        latest_node = self._get_latest_node()
        if latest_node:
            self.graph.add_edge(
                latest_node,
                node_id,
                edge_type="time",
                weight=1.0,
                metadata={}
            )

        self._save_graph()
        return node_id

    def add_feedback(
        self,
        node_id: str,
        feedback: str,
        result: str,
        adaptation_metadata: Dict[str, any]
    ):
        """Add user feedback to evolution point"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found")

        # Update node with feedback
        node_data = self.graph.nodes[node_id]
        node_data["feedback"] = feedback
        node_data["result"] = result

        # Add feedback edge if next node exists
        next_node = self._get_next_node(node_id)
        if next_node:
            self.graph.add_edge(
                node_id,
                next_node,
                edge_type="feedback",
                weight=0.8,
                metadata=adaptation_metadata
            )

        self._save_graph()

    def _get_latest_node(self) -> Optional[str]:
        """Get most recent node ID"""
        if not self.graph.nodes:
            return None

        return max(
            self.graph.nodes,
            key=lambda n: self.graph.nodes[n]["timestamp"]
        )

    def _get_next_node(self, node_id: str) -> Optional[str]:
        """Get next node in timeline"""
        successors = list(self.graph.successors(node_id))
        if not successors:
            return None
        return successors[0]

    def get_evolution_metrics(self) -> Dict[str, any]:
        """Calculate evolution metrics"""
        if not self.graph.nodes:
            return {
                "resilience_growth": 0.0,
                "adaptation_loops": 0,
                "confidence_trend": 0.0
            }

        nodes = list(self.graph.nodes(data=True))
        
        # Calculate metrics
        start_score = nodes[0][1]["bio_score"]
        end_score = nodes[-1][1]["bio_score"]
        resilience_growth = (end_score - start_score) / start_score

        adaptation_loops = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d["edge_type"] == "feedback"
        )

        confidence_scores = [
            n[1]["confidence"] for n in nodes
        ]
        confidence_trend = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )

        return {
            "resilience_growth": round(resilience_growth, 2),
            "adaptation_loops": adaptation_loops,
            "confidence_trend": round(confidence_trend, 2)
        }

    def export_graph(self) -> Dict[str, any]:
        """Export graph as JSON structure"""
        return {
            "user_id": self.user_id,
            "nodes": [
                {
                    "node_id": n,
                    **self.graph.nodes[n]
                }
                for n in self.graph.nodes()
            ],
            "edges": [
                {
                    "from_node": u,
                    "to_node": v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "insights": self.get_evolution_metrics()
        }

    def get_current_state(self) -> Optional[BioCognitiveState]:
        """Get user's current state"""
        latest = self._get_latest_node()
        if not latest:
            return None

        node_data = self.graph.nodes[latest]
        return BioCognitiveState(
            timestamp=node_data["timestamp"],
            biometrics=node_data["biometrics"],
            cognitive_state=node_data["cognitive_state"],
            longevity_score=node_data["bio_score"],
            active_interventions=[node_data["intervention"]]
        )