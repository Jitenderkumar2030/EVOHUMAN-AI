"""
SymbioticAIS Multi-Agent System
Advanced human-AI symbiotic evolution through multi-agent reinforcement learning
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
import structlog
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import random


logger = structlog.get_logger("multi-agent-system")


class AgentType(Enum):
    HUMAN_PROXY = "human_proxy"
    EVOLUTION_OPTIMIZER = "evolution_optimizer"
    FEEDBACK_PROCESSOR = "feedback_processor"
    GOAL_COORDINATOR = "goal_coordinator"
    ADAPTATION_SPECIALIST = "adaptation_specialist"


class AgentState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    LEARNING = "learning"
    COLLABORATING = "collaborating"
    ADAPTING = "adapting"


@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    target_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0


@dataclass
class AgentObservation:
    """Observation received by an agent"""
    observer_id: str
    observation_type: str
    data: Dict[str, Any]
    source_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float = 1.0


@dataclass
class AgentReward:
    """Reward signal for reinforcement learning"""
    agent_id: str
    reward_value: float
    reward_type: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SymbioticAgent:
    """Individual agent in the multi-agent system"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState.IDLE
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Q-learning parameters
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.action_history: List[AgentAction] = []
        self.observation_history: List[AgentObservation] = []
        self.reward_history: List[AgentReward] = []
        
        # Agent-specific parameters
        self.specialization_weights = self._initialize_specialization()
        self.collaboration_preferences = {}
        self.adaptation_memory = {}
        
        # Performance metrics
        self.success_rate = 0.5
        self.collaboration_score = 0.5
        self.learning_progress = 0.0
        
        logger.info("Agent initialized", 
                   agent_id=agent_id, 
                   agent_type=agent_type.value)
    
    def _initialize_specialization(self) -> Dict[str, float]:
        """Initialize agent specialization weights"""
        
        base_weights = {
            "human_interaction": 0.5,
            "data_analysis": 0.5,
            "goal_optimization": 0.5,
            "feedback_processing": 0.5,
            "adaptation": 0.5
        }
        
        # Specialize based on agent type
        if self.agent_type == AgentType.HUMAN_PROXY:
            base_weights["human_interaction"] = 0.9
            base_weights["feedback_processing"] = 0.8
        elif self.agent_type == AgentType.EVOLUTION_OPTIMIZER:
            base_weights["goal_optimization"] = 0.9
            base_weights["data_analysis"] = 0.8
        elif self.agent_type == AgentType.FEEDBACK_PROCESSOR:
            base_weights["feedback_processing"] = 0.9
            base_weights["adaptation"] = 0.7
        elif self.agent_type == AgentType.GOAL_COORDINATOR:
            base_weights["goal_optimization"] = 0.8
            base_weights["human_interaction"] = 0.7
        elif self.agent_type == AgentType.ADAPTATION_SPECIALIST:
            base_weights["adaptation"] = 0.9
            base_weights["data_analysis"] = 0.8
        
        return base_weights
    
    async def perceive(self, observation: AgentObservation):
        """Process incoming observation"""
        
        self.observation_history.append(observation)
        
        # Update internal state based on observation
        await self._process_observation(observation)
        
        # Trigger learning if appropriate
        if len(self.observation_history) % 10 == 0:
            await self._update_learning()
    
    async def act(self, environment_state: Dict[str, Any]) -> Optional[AgentAction]:
        """Choose and execute action based on current state"""
        
        # Get possible actions
        possible_actions = self._get_possible_actions(environment_state)
        
        if not possible_actions:
            return None
        
        # Choose action using epsilon-greedy strategy
        if random.random() < self.exploration_rate:
            # Explore: random action
            chosen_action = random.choice(possible_actions)
        else:
            # Exploit: best known action
            chosen_action = self._choose_best_action(possible_actions, environment_state)
        
        # Create action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=chosen_action["type"],
            parameters=chosen_action["parameters"],
            target_agent=chosen_action.get("target"),
            confidence=chosen_action.get("confidence", 0.8)
        )
        
        self.action_history.append(action)
        self.state = AgentState.ACTIVE
        
        logger.debug("Agent action", 
                    agent_id=self.agent_id,
                    action_type=action.action_type)
        
        return action
    
    async def receive_reward(self, reward: AgentReward):
        """Process reward signal for learning"""
        
        self.reward_history.append(reward)
        
        # Update Q-values based on reward
        await self._update_q_values(reward)
        
        # Update performance metrics
        self._update_performance_metrics(reward)
    
    def _get_possible_actions(self, environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of possible actions given current environment"""
        
        actions = []
        
        if self.agent_type == AgentType.HUMAN_PROXY:
            actions.extend([
                {"type": "request_human_feedback", "parameters": {"topic": "current_goal"}},
                {"type": "interpret_human_input", "parameters": {"input_type": "preference"}},
                {"type": "suggest_human_action", "parameters": {"action_category": "health"}}
            ])
        
        elif self.agent_type == AgentType.EVOLUTION_OPTIMIZER:
            actions.extend([
                {"type": "optimize_evolution_path", "parameters": {"target": "longevity"}},
                {"type": "analyze_progress", "parameters": {"metric": "overall_health"}},
                {"type": "suggest_intervention", "parameters": {"intervention_type": "lifestyle"}}
            ])
        
        elif self.agent_type == AgentType.FEEDBACK_PROCESSOR:
            actions.extend([
                {"type": "process_feedback", "parameters": {"feedback_type": "human"}},
                {"type": "analyze_feedback_patterns", "parameters": {"time_window": "week"}},
                {"type": "generate_feedback_summary", "parameters": {"target": "human"}}
            ])
        
        elif self.agent_type == AgentType.GOAL_COORDINATOR:
            actions.extend([
                {"type": "coordinate_goals", "parameters": {"priority": "high"}},
                {"type": "resolve_goal_conflicts", "parameters": {"strategy": "compromise"}},
                {"type": "update_goal_priorities", "parameters": {"based_on": "progress"}}
            ])
        
        elif self.agent_type == AgentType.ADAPTATION_SPECIALIST:
            actions.extend([
                {"type": "adapt_strategy", "parameters": {"adaptation_type": "learning_rate"}},
                {"type": "analyze_adaptation_needs", "parameters": {"scope": "system"}},
                {"type": "implement_adaptation", "parameters": {"target": "collaboration"}}
            ])
        
        # Add collaboration actions
        actions.extend([
            {"type": "collaborate", "parameters": {"collaboration_type": "information_sharing"}},
            {"type": "request_assistance", "parameters": {"assistance_type": "analysis"}},
            {"type": "share_knowledge", "parameters": {"knowledge_type": "experience"}}
        ])
        
        return actions
    
    def _choose_best_action(
        self, 
        possible_actions: List[Dict[str, Any]], 
        environment_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Choose best action based on Q-values"""
        
        state_key = self._encode_state(environment_state)
        
        if state_key not in self.q_table:
            # Initialize Q-values for new state
            self.q_table[state_key] = {}
            for action in possible_actions:
                action_key = action["type"]
                self.q_table[state_key][action_key] = 0.0
        
        # Find action with highest Q-value
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            action_key = action["type"]
            q_value = self.q_table[state_key].get(action_key, 0.0)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action or possible_actions[0]
    
    def _encode_state(self, environment_state: Dict[str, Any]) -> str:
        """Encode environment state as string key"""
        
        # Simplified state encoding
        key_components = []
        
        # Include relevant state components
        if "human_satisfaction" in environment_state:
            satisfaction = environment_state["human_satisfaction"]
            key_components.append(f"sat_{int(satisfaction * 10)}")
        
        if "goal_progress" in environment_state:
            progress = environment_state["goal_progress"]
            key_components.append(f"prog_{int(progress * 10)}")
        
        if "system_performance" in environment_state:
            performance = environment_state["system_performance"]
            key_components.append(f"perf_{int(performance * 10)}")
        
        return "_".join(key_components) if key_components else "default"
    
    async def _process_observation(self, observation: AgentObservation):
        """Process incoming observation and update internal state"""
        
        # Update collaboration preferences based on observation source
        if observation.source_agent:
            if observation.source_agent not in self.collaboration_preferences:
                self.collaboration_preferences[observation.source_agent] = 0.5
            
            # Adjust preference based on observation relevance
            current_pref = self.collaboration_preferences[observation.source_agent]
            adjustment = (observation.relevance_score - 0.5) * 0.1
            self.collaboration_preferences[observation.source_agent] = max(0.0, min(1.0, current_pref + adjustment))
        
        # Store relevant observations in adaptation memory
        if observation.relevance_score > 0.7:
            memory_key = f"{observation.observation_type}_{observation.timestamp.strftime('%Y%m%d')}"
            self.adaptation_memory[memory_key] = observation.data
    
    async def _update_learning(self):
        """Update learning based on recent experiences"""
        
        if len(self.reward_history) < 5:
            return
        
        # Calculate recent performance
        recent_rewards = self.reward_history[-5:]
        avg_reward = np.mean([r.reward_value for r in recent_rewards])
        
        # Adjust exploration rate based on performance
        if avg_reward > 0.7:
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)  # Reduce exploration
        elif avg_reward < 0.3:
            self.exploration_rate = min(0.3, self.exploration_rate * 1.05)  # Increase exploration
        
        # Update learning progress
        self.learning_progress = min(1.0, self.learning_progress + 0.01)
    
    async def _update_q_values(self, reward: AgentReward):
        """Update Q-values based on reward"""
        
        if not self.action_history:
            return
        
        # Get last action
        last_action = self.action_history[-1]
        
        # Simplified Q-learning update
        # In practice, would need proper state transitions
        state_key = "current_state"  # Simplified
        action_key = last_action.action_type
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state_key][action_key]
        learning_rate = self.learning_rate
        discount_factor = 0.9
        
        # Simplified update (no next state consideration)
        new_q = current_q + learning_rate * (reward.reward_value - current_q)
        self.q_table[state_key][action_key] = new_q
    
    def _update_performance_metrics(self, reward: AgentReward):
        """Update agent performance metrics"""
        
        # Update success rate
        success = 1.0 if reward.reward_value > 0.5 else 0.0
        self.success_rate = 0.9 * self.success_rate + 0.1 * success
        
        # Update collaboration score if reward is collaboration-related
        if "collaboration" in reward.reward_type:
            collab_success = 1.0 if reward.reward_value > 0.6 else 0.0
            self.collaboration_score = 0.9 * self.collaboration_score + 0.1 * collab_success
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "success_rate": self.success_rate,
            "collaboration_score": self.collaboration_score,
            "learning_progress": self.learning_progress,
            "exploration_rate": self.exploration_rate,
            "total_actions": len(self.action_history),
            "total_observations": len(self.observation_history),
            "total_rewards": len(self.reward_history),
            "specialization_weights": self.specialization_weights,
            "collaboration_preferences": self.collaboration_preferences
        }


class MultiAgentSystem:
    """Multi-agent system for symbiotic human-AI evolution"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.agents: Dict[str, SymbioticAgent] = {}
        self.environment_state: Dict[str, Any] = {}
        self.system_performance = 0.5
        self.collaboration_network: Dict[str, Set[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default agents
        self._initialize_default_agents()
        
        logger.info("Multi-agent system initialized", user_id=user_id)
    
    def _initialize_default_agents(self):
        """Initialize default set of agents"""
        
        agent_configs = [
            ("human_proxy_1", AgentType.HUMAN_PROXY),
            ("evolution_optimizer_1", AgentType.EVOLUTION_OPTIMIZER),
            ("feedback_processor_1", AgentType.FEEDBACK_PROCESSOR),
            ("goal_coordinator_1", AgentType.GOAL_COORDINATOR),
            ("adaptation_specialist_1", AgentType.ADAPTATION_SPECIALIST)
        ]
        
        for agent_id, agent_type in agent_configs:
            agent = SymbioticAgent(agent_id, agent_type)
            self.agents[agent_id] = agent
            self.collaboration_network[agent_id] = set()
        
        # Set up initial collaboration connections
        self._setup_collaboration_network()
    
    def _setup_collaboration_network(self):
        """Set up initial collaboration connections between agents"""
        
        # Human proxy collaborates with all agents
        human_proxy = "human_proxy_1"
        for agent_id in self.agents.keys():
            if agent_id != human_proxy:
                self.collaboration_network[human_proxy].add(agent_id)
                self.collaboration_network[agent_id].add(human_proxy)
        
        # Evolution optimizer collaborates with goal coordinator
        self.collaboration_network["evolution_optimizer_1"].add("goal_coordinator_1")
        self.collaboration_network["goal_coordinator_1"].add("evolution_optimizer_1")
        
        # Feedback processor collaborates with adaptation specialist
        self.collaboration_network["feedback_processor_1"].add("adaptation_specialist_1")
        self.collaboration_network["adaptation_specialist_1"].add("feedback_processor_1")
    
    async def step(self, human_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one step of the multi-agent system"""
        
        # Update environment state
        await self._update_environment_state(human_input)
        
        # Get actions from all agents
        agent_actions = {}
        for agent_id, agent in self.agents.items():
            action = await agent.act(self.environment_state)
            if action:
                agent_actions[agent_id] = action
        
        # Process agent interactions
        interaction_results = await self._process_agent_interactions(agent_actions)
        
        # Calculate rewards
        rewards = await self._calculate_rewards(agent_actions, interaction_results)
        
        # Distribute rewards to agents
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                await self.agents[agent_id].receive_reward(reward)
        
        # Update system performance
        self._update_system_performance(rewards)
        
        return {
            "step_results": {
                "actions_taken": len(agent_actions),
                "interactions_processed": len(interaction_results),
                "rewards_distributed": len(rewards),
                "system_performance": self.system_performance
            },
            "agent_actions": {aid: action.action_type for aid, action in agent_actions.items()},
            "system_state": self.environment_state.copy()
        }
    
    async def _update_environment_state(self, human_input: Optional[Dict[str, Any]]):
        """Update environment state based on human input and system state"""
        
        # Update human-related state
        if human_input:
            self.environment_state["human_satisfaction"] = human_input.get("satisfaction", 0.5)
            self.environment_state["human_goals"] = human_input.get("goals", [])
            self.environment_state["human_feedback"] = human_input.get("feedback", {})
        
        # Update system state
        self.environment_state["system_performance"] = self.system_performance
        self.environment_state["agent_count"] = len(self.agents)
        self.environment_state["collaboration_density"] = self._calculate_collaboration_density()
        
        # Calculate goal progress (simplified)
        if "human_goals" in self.environment_state:
            goal_progress = min(1.0, len(self.environment_state["human_goals"]) * 0.2)
            self.environment_state["goal_progress"] = goal_progress
    
    async def _process_agent_interactions(
        self, 
        agent_actions: Dict[str, AgentAction]
    ) -> List[Dict[str, Any]]:
        """Process interactions between agents"""
        
        interactions = []
        
        for agent_id, action in agent_actions.items():
            # Process collaboration actions
            if action.action_type == "collaborate":
                collaborators = self.collaboration_network.get(agent_id, set())
                for collaborator_id in collaborators:
                    if collaborator_id in self.agents:
                        # Create observation for collaborator
                        observation = AgentObservation(
                            observer_id=collaborator_id,
                            observation_type="collaboration_request",
                            data=action.parameters,
                            source_agent=agent_id,
                            relevance_score=0.8
                        )
                        
                        await self.agents[collaborator_id].perceive(observation)
                        
                        interactions.append({
                            "type": "collaboration",
                            "initiator": agent_id,
                            "target": collaborator_id,
                            "success": True
                        })
            
            # Process information sharing
            elif action.action_type == "share_knowledge":
                # Broadcast knowledge to connected agents
                connected_agents = self.collaboration_network.get(agent_id, set())
                for target_id in connected_agents:
                    if target_id in self.agents:
                        observation = AgentObservation(
                            observer_id=target_id,
                            observation_type="knowledge_sharing",
                            data=action.parameters,
                            source_agent=agent_id,
                            relevance_score=0.7
                        )
                        
                        await self.agents[target_id].perceive(observation)
                        
                        interactions.append({
                            "type": "knowledge_sharing",
                            "source": agent_id,
                            "target": target_id,
                            "success": True
                        })
        
        return interactions
    
    async def _calculate_rewards(
        self, 
        agent_actions: Dict[str, AgentAction],
        interaction_results: List[Dict[str, Any]]
    ) -> Dict[str, AgentReward]:
        """Calculate rewards for agents based on actions and outcomes"""
        
        rewards = {}
        
        # Base reward for taking action
        for agent_id, action in agent_actions.items():
            base_reward = 0.1  # Small reward for being active
            
            # Bonus for successful collaboration
            collaboration_bonus = 0.0
            for interaction in interaction_results:
                if (interaction.get("initiator") == agent_id or 
                    interaction.get("source") == agent_id) and interaction.get("success"):
                    collaboration_bonus += 0.2
            
            # Performance-based reward
            performance_reward = self.system_performance * 0.3
            
            total_reward = base_reward + collaboration_bonus + performance_reward
            
            rewards[agent_id] = AgentReward(
                agent_id=agent_id,
                reward_value=min(1.0, total_reward),
                reward_type="action_outcome",
                context={
                    "action_type": action.action_type,
                    "collaboration_bonus": collaboration_bonus,
                    "performance_reward": performance_reward
                }
            )
        
        return rewards
    
    def _calculate_collaboration_density(self) -> float:
        """Calculate density of collaboration network"""
        
        if len(self.agents) <= 1:
            return 0.0
        
        total_possible_connections = len(self.agents) * (len(self.agents) - 1)
        actual_connections = sum(len(connections) for connections in self.collaboration_network.values())
        
        return actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
    
    def _update_system_performance(self, rewards: Dict[str, AgentReward]):
        """Update overall system performance based on agent rewards"""
        
        if not rewards:
            return
        
        avg_reward = np.mean([reward.reward_value for reward in rewards.values()])
        
        # Update system performance with momentum
        self.system_performance = 0.9 * self.system_performance + 0.1 * avg_reward
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_agent_status()
        
        return {
            "user_id": self.user_id,
            "system_performance": self.system_performance,
            "environment_state": self.environment_state,
            "collaboration_density": self._calculate_collaboration_density(),
            "agent_count": len(self.agents),
            "agents": agent_statuses,
            "collaboration_network": {
                agent_id: list(connections) 
                for agent_id, connections in self.collaboration_network.items()
            }
        }
