"""
AiCE Memory Graph Engine
Advanced graph-based memory storage and retrieval system
"""
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger(__name__)


class MemoryGraphEngine:
    """Graph-based memory storage and retrieval system"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.embeddings_model = None
        self.memory_store = {}  # {memory_id: memory_data}
        self.user_graphs = {}   # {user_id: user-specific subgraph}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory graph engine"""
        try:
            # Load sentence transformer model for semantic similarity
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.initialized = True
            logger.info("Memory graph engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load embeddings model: {e}")
            logger.info("Running in mock mode")
    
    async def store_memory(
        self,
        user_id: str,
        memory_type: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Store a new memory in the graph"""
        
        memory_id = f"mem_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create memory node
        memory_data = {
            "memory_id": memory_id,
            "user_id": user_id,
            "memory_type": memory_type,
            "content": content,
            "importance": importance,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "access_count": 0,
            "last_accessed": datetime.utcnow().isoformat()
        }
        
        # Generate embedding if model is available
        if self.embeddings_model and isinstance(content, dict):
            text_content = self._extract_text_content(content)
            if text_content:
                try:
                    embedding = self.embeddings_model.encode(text_content)
                    memory_data["embedding"] = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
        
        # Add to graph
        self.graph.add_node(memory_id, **memory_data)
        self.memory_store[memory_id] = memory_data
        
        # Initialize user subgraph if needed
        if user_id not in self.user_graphs:
            self.user_graphs[user_id] = set()
        self.user_graphs[user_id].add(memory_id)
        
        # Connect to similar memories
        await self._create_semantic_connections(memory_id, user_id)
        
        logger.info(f"Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    async def recall_memories(
        self,
        user_id: str,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10,
        importance_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Recall memories from the graph"""
        
        if user_id not in self.user_graphs:
            return []
        
        user_memories = self.user_graphs[user_id]
        
        # Filter by type if specified
        if memory_type:
            user_memories = [
                mid for mid in user_memories 
                if self.memory_store.get(mid, {}).get("memory_type") == memory_type
            ]
        
        # Filter by importance
        user_memories = [
            mid for mid in user_memories
            if self.memory_store.get(mid, {}).get("importance", 0) >= importance_threshold
        ]
        
        recalled_memories = []
        
        if query and self.embeddings_model:
            # Semantic search
            try:
                query_embedding = self.embeddings_model.encode(query)
                similarities = []
                
                for memory_id in user_memories:
                    memory_data = self.memory_store.get(memory_id)
                    if memory_data and "embedding" in memory_data:
                        memory_embedding = np.array(memory_data["embedding"])
                        similarity = np.dot(query_embedding, memory_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                        )
                        similarities.append((memory_id, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for memory_id, similarity in similarities[:limit]:
                    memory_data = self.memory_store[memory_id].copy()
                    memory_data["similarity_score"] = float(similarity)
                    # Update access tracking
                    memory_data["access_count"] += 1
                    memory_data["last_accessed"] = datetime.utcnow().isoformat()
                    recalled_memories.append(memory_data)
                    
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
                # Fall back to simple recall
                recalled_memories = self._simple_recall(user_memories, limit)
        else:
            # Simple recall by importance and recency
            recalled_memories = self._simple_recall(user_memories, limit)
        
        return recalled_memories
    
    async def create_connections(
        self,
        user_id: str,
        memory_ids: List[str],
        connection_type: str = "related",
        strength: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Create connections between memories"""
        
        connections = []
        
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                mem1_id = memory_ids[i]
                mem2_id = memory_ids[j]
                
                # Verify memories belong to user
                if (mem1_id in self.user_graphs.get(user_id, set()) and 
                    mem2_id in self.user_graphs.get(user_id, set())):
                    
                    # Add edge to graph
                    self.graph.add_edge(
                        mem1_id, 
                        mem2_id,
                        connection_type=connection_type,
                        strength=strength,
                        created_at=datetime.utcnow().isoformat()
                    )
                    
                    connections.append({
                        "from_memory": mem1_id,
                        "to_memory": mem2_id,
                        "connection_type": connection_type,
                        "strength": strength
                    })
        
        logger.info(f"Created {len(connections)} connections for user {user_id}")
        return connections
    
    async def get_memory_network(self, user_id: str, memory_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get the network of connected memories around a specific memory"""
        
        if user_id not in self.user_graphs or memory_id not in self.user_graphs[user_id]:
            return {"nodes": [], "edges": []}
        
        # Get subgraph around memory
        try:
            subgraph_nodes = set([memory_id])
            
            # Expand network by depth
            current_nodes = {memory_id}
            for _ in range(depth):
                next_nodes = set()
                for node in current_nodes:
                    # Add connected nodes
                    neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                    # Filter to user's memories only
                    neighbors = neighbors.intersection(self.user_graphs[user_id])
                    next_nodes.update(neighbors)
                
                subgraph_nodes.update(next_nodes)
                current_nodes = next_nodes
                
                if not current_nodes:  # No more connections
                    break
            
            # Build network representation
            nodes = []
            for node_id in subgraph_nodes:
                if node_id in self.memory_store:
                    node_data = self.memory_store[node_id].copy()
                    # Simplify content for network view
                    if "embedding" in node_data:
                        del node_data["embedding"]
                    nodes.append(node_data)
            
            edges = []
            for node_id in subgraph_nodes:
                for successor in self.graph.successors(node_id):
                    if successor in subgraph_nodes:
                        edge_data = self.graph[node_id][successor]
                        for key, attrs in edge_data.items():
                            edges.append({
                                "from": node_id,
                                "to": successor,
                                **attrs
                            })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "center_memory": memory_id,
                "depth": depth
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory network: {e}")
            return {"nodes": [], "edges": []}
    
    async def find_memory_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Find patterns in user's memory graph"""
        
        if user_id not in self.user_graphs:
            return []
        
        patterns = []
        user_memories = self.user_graphs[user_id]
        
        try:
            # Pattern 1: Highly connected memories (hubs)
            degree_centrality = nx.degree_centrality(self.graph)
            high_centrality = [
                (node, centrality) for node, centrality in degree_centrality.items()
                if node in user_memories and centrality > 0.1
            ]
            
            if high_centrality:
                patterns.append({
                    "pattern_type": "hub_memories",
                    "description": "Memories that are highly connected to other memories",
                    "memories": [
                        {
                            "memory_id": node,
                            "centrality_score": centrality,
                            "content_summary": self._get_memory_summary(node)
                        }
                        for node, centrality in sorted(high_centrality, key=lambda x: x[1], reverse=True)[:5]
                    ]
                })
            
            # Pattern 2: Memory clusters
            user_subgraph = self.graph.subgraph(user_memories)
            communities = nx.community.greedy_modularity_communities(user_subgraph.to_undirected())
            
            if len(communities) > 1:
                patterns.append({
                    "pattern_type": "memory_clusters",
                    "description": "Groups of related memories that form clusters",
                    "clusters": [
                        {
                            "cluster_id": i,
                            "size": len(community),
                            "memories": [
                                {
                                    "memory_id": node,
                                    "content_summary": self._get_memory_summary(node)
                                }
                                for node in list(community)[:3]  # Show first 3 memories
                            ]
                        }
                        for i, community in enumerate(communities[:5])  # Show first 5 clusters
                    ]
                })
            
            # Pattern 3: Temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(user_memories)
            if temporal_patterns:
                patterns.append(temporal_patterns)
            
        except Exception as e:
            logger.warning(f"Failed to find patterns: {e}")
        
        return patterns
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.embeddings_model:
            # Cleanup model resources if needed
            pass
        logger.info("Memory graph engine cleaned up")
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract text content from memory content dict"""
        text_parts = []
        
        for key, value in content.items():
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                text_parts.extend(value)
        
        return " ".join(text_parts)
    
    async def _create_semantic_connections(self, memory_id: str, user_id: str):
        """Create connections to semantically similar memories"""
        
        if not self.embeddings_model or user_id not in self.user_graphs:
            return
        
        current_memory = self.memory_store.get(memory_id)
        if not current_memory or "embedding" not in current_memory:
            return
        
        current_embedding = np.array(current_memory["embedding"])
        
        # Find similar memories
        similarities = []
        for other_memory_id in self.user_graphs[user_id]:
            if other_memory_id == memory_id:
                continue
            
            other_memory = self.memory_store.get(other_memory_id)
            if not other_memory or "embedding" not in other_memory:
                continue
            
            other_embedding = np.array(other_memory["embedding"])
            similarity = np.dot(current_embedding, other_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
            )
            
            if similarity > 0.7:  # High similarity threshold
                similarities.append((other_memory_id, similarity))
        
        # Create connections to most similar memories
        similarities.sort(key=lambda x: x[1], reverse=True)
        for other_memory_id, similarity in similarities[:3]:  # Top 3 similar memories
            self.graph.add_edge(
                memory_id,
                other_memory_id,
                connection_type="semantic_similarity",
                strength=float(similarity),
                created_at=datetime.utcnow().isoformat()
            )
    
    def _simple_recall(self, memory_ids: List[str], limit: int) -> List[Dict[str, Any]]:
        """Simple recall based on importance and recency"""
        
        memories = []
        for memory_id in memory_ids:
            memory_data = self.memory_store.get(memory_id)
            if memory_data:
                # Calculate composite score
                importance = memory_data.get("importance", 0.5)
                created_at = datetime.fromisoformat(memory_data["created_at"])
                recency = 1.0 / max(1, (datetime.utcnow() - created_at).days)
                
                memory_data = memory_data.copy()
                memory_data["composite_score"] = importance * 0.7 + recency * 0.3
                memory_data["access_count"] += 1
                memory_data["last_accessed"] = datetime.utcnow().isoformat()
                memories.append(memory_data)
        
        # Sort by composite score
        memories.sort(key=lambda x: x["composite_score"], reverse=True)
        return memories[:limit]
    
    def _analyze_temporal_patterns(self, memory_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns in memories"""
        
        try:
            memory_times = []
            for memory_id in memory_ids:
                memory_data = self.memory_store.get(memory_id)
                if memory_data:
                    created_at = datetime.fromisoformat(memory_data["created_at"])
                    memory_times.append((created_at, memory_data.get("memory_type", "unknown")))
            
            if len(memory_times) < 5:  # Need enough data points
                return None
            
            # Find most active time periods
            memory_times.sort()
            
            # Group by day
            daily_counts = {}
            type_counts = {}
            
            for time_stamp, memory_type in memory_times:
                day = time_stamp.date()
                daily_counts[day] = daily_counts.get(day, 0) + 1
                type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            
            # Find peak activity day
            peak_day = max(daily_counts.items(), key=lambda x: x[1])
            most_common_type = max(type_counts.items(), key=lambda x: x[1])
            
            return {
                "pattern_type": "temporal_patterns",
                "description": "Patterns in when memories are created",
                "insights": [
                    f"Peak memory creation day: {peak_day[0]} ({peak_day[1]} memories)",
                    f"Most common memory type: {most_common_type[0]} ({most_common_type[1]} memories)",
                    f"Total memories analyzed: {len(memory_times)}"
                ],
                "peak_day": str(peak_day[0]),
                "dominant_type": most_common_type[0]
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze temporal patterns: {e}")
            return None
    
    def _get_memory_summary(self, memory_id: str) -> str:
        """Get a brief summary of memory content"""
        
        memory = self.memory_store.get(memory_id)
        if not memory:
            return "Unknown memory"
        
        content = memory.get("content", {})
        if isinstance(content, dict):
            # Try to find a summary or title field
            for key in ["summary", "title", "description", "text", "content"]:
                if key in content and isinstance(content[key], str):
                    return content[key][:100] + ("..." if len(content[key]) > 100 else "")
            
            # Fall back to first string value
            for value in content.values():
                if isinstance(value, str) and len(value) > 10:
                    return value[:100] + ("..." if len(value) > 100 else "")
        
        return f"{memory.get('memory_type', 'Memory')} from {memory.get('created_at', 'unknown time')[:10]}"
