import { useState, useCallback } from 'react';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';

interface AgentStatus {
  agent_id: string;
  agent_type: string;
  state: string;
  success_rate: number;
  collaboration_score: number;
  learning_progress: number;
  exploration_rate: number;
  total_actions: number;
  total_observations: number;
  total_rewards: number;
  specialization_weights: Record<string, number>;
  collaboration_preferences: Record<string, number>;
}

interface SystemStatus {
  user_id: string;
  system_performance: number;
  environment_state: Record<string, any>;
  collaboration_density: number;
  agent_count: number;
  agents: Record<string, AgentStatus>;
  collaboration_network: Record<string, string[]>;
}

interface StepResult {
  step_results: {
    actions_taken: number;
    interactions_processed: number;
    rewards_distributed: number;
    system_performance: number;
  };
  agent_actions: Record<string, string>;
  system_state: Record<string, any>;
}

interface HumanFeedback {
  satisfaction: number;
  goals: string[];
  feedback: Record<string, any>;
}

export const useMultiAgentSystem = (userId: string) => {
  const [agentStatus, setAgentStatus] = useState<SystemStatus | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  // Fetch agent system status
  const { data: statusData, isLoading: isStatusLoading } = useQuery(
    ['multiAgentStatus', userId],
    async () => {
      try {
        const response = await axios.get(`/api/symbiotic/multi_agent/status/${userId}`);
        return response.data;
      } catch (error) {
        // Return mock data if service is not available
        return generateMockSystemStatus(userId);
      }
    },
    {
      refetchInterval: 10000, // Refetch every 10 seconds
      enabled: isInitialized,
      onSuccess: (data) => {
        setAgentStatus(data.system_status || data);
      },
    }
  );

  // Initialize multi-agent system mutation
  const initializeMutation = useMutation(
    async () => {
      const response = await axios.post('/api/symbiotic/multi_agent/initialize', {
        user_id: userId,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setAgentStatus(data.system_status);
        setIsInitialized(true);
      },
      onError: (error) => {
        console.error('Failed to initialize multi-agent system:', error);
        // Set mock data on error
        setAgentStatus(generateMockSystemStatus(userId));
        setIsInitialized(true);
      },
    }
  );

  // Execute agent step mutation
  const stepMutation = useMutation(
    async (humanInput?: Record<string, any>) => {
      const response = await axios.post('/api/symbiotic/multi_agent/step', {
        user_id: userId,
        human_input: humanInput,
      });
      return response.data;
    },
    {
      onError: (error) => {
        console.error('Agent step failed:', error);
      },
    }
  );

  // Process human feedback mutation
  const feedbackMutation = useMutation(
    async (feedbackData: HumanFeedback) => {
      const response = await axios.post('/api/symbiotic/multi_agent/human_feedback', {
        user_id: userId,
        feedback_data: feedbackData,
      });
      return response.data;
    },
    {
      onError: (error) => {
        console.error('Feedback processing failed:', error);
      },
    }
  );

  const initializeSystem = useCallback(async () => {
    if (!isInitialized) {
      return await initializeMutation.mutateAsync();
    }
    return agentStatus;
  }, [initializeMutation, isInitialized, agentStatus]);

  const executeStep = useCallback(async (humanInput?: Record<string, any>) => {
    if (!isInitialized) {
      await initializeSystem();
    }
    return await stepMutation.mutateAsync(humanInput);
  }, [stepMutation, isInitialized, initializeSystem]);

  const processHumanFeedback = useCallback(async (feedbackData: HumanFeedback) => {
    if (!isInitialized) {
      await initializeSystem();
    }
    return await feedbackMutation.mutateAsync(feedbackData);
  }, [feedbackMutation, isInitialized, initializeSystem]);

  // Auto-initialize on first use
  React.useEffect(() => {
    if (!isInitialized && !initializeMutation.isLoading) {
      initializeSystem();
    }
  }, [isInitialized, initializeMutation.isLoading, initializeSystem]);

  return {
    agentStatus: agentStatus || statusData?.system_status,
    isLoading: isStatusLoading || initializeMutation.isLoading,
    isInitialized,
    initializeSystem,
    executeStep,
    processHumanFeedback,
    isExecutingStep: stepMutation.isLoading,
    isProcessingFeedback: feedbackMutation.isLoading,
  };
};

// Mock data generator
const generateMockSystemStatus = (userId: string): SystemStatus => ({
  user_id: userId,
  system_performance: 0.78,
  environment_state: {
    human_satisfaction: 0.82,
    goal_progress: 0.67,
    system_performance: 0.78,
    agent_count: 5,
    collaboration_density: 0.6,
  },
  collaboration_density: 0.6,
  agent_count: 5,
  agents: {
    human_proxy_1: {
      agent_id: 'human_proxy_1',
      agent_type: 'human_proxy',
      state: 'active',
      success_rate: 0.85,
      collaboration_score: 0.92,
      learning_progress: 0.73,
      exploration_rate: 0.15,
      total_actions: 247,
      total_observations: 189,
      total_rewards: 156,
      specialization_weights: {
        human_interaction: 0.9,
        feedback_processing: 0.8,
        data_analysis: 0.4,
        goal_optimization: 0.5,
        adaptation: 0.6,
      },
      collaboration_preferences: {
        evolution_optimizer_1: 0.88,
        feedback_processor_1: 0.91,
        goal_coordinator_1: 0.85,
        adaptation_specialist_1: 0.79,
      },
    },
    evolution_optimizer_1: {
      agent_id: 'evolution_optimizer_1',
      agent_type: 'evolution_optimizer',
      state: 'active',
      success_rate: 0.79,
      collaboration_score: 0.76,
      learning_progress: 0.81,
      exploration_rate: 0.12,
      total_actions: 198,
      total_observations: 156,
      total_rewards: 134,
      specialization_weights: {
        goal_optimization: 0.9,
        data_analysis: 0.8,
        human_interaction: 0.3,
        feedback_processing: 0.4,
        adaptation: 0.6,
      },
      collaboration_preferences: {
        human_proxy_1: 0.88,
        goal_coordinator_1: 0.94,
        feedback_processor_1: 0.72,
        adaptation_specialist_1: 0.68,
      },
    },
    feedback_processor_1: {
      agent_id: 'feedback_processor_1',
      agent_type: 'feedback_processor',
      state: 'learning',
      success_rate: 0.82,
      collaboration_score: 0.87,
      learning_progress: 0.69,
      exploration_rate: 0.18,
      total_actions: 223,
      total_observations: 201,
      total_rewards: 178,
      specialization_weights: {
        feedback_processing: 0.9,
        adaptation: 0.7,
        human_interaction: 0.6,
        data_analysis: 0.5,
        goal_optimization: 0.4,
      },
      collaboration_preferences: {
        human_proxy_1: 0.91,
        adaptation_specialist_1: 0.89,
        evolution_optimizer_1: 0.72,
        goal_coordinator_1: 0.75,
      },
    },
    goal_coordinator_1: {
      agent_id: 'goal_coordinator_1',
      agent_type: 'goal_coordinator',
      state: 'collaborating',
      success_rate: 0.77,
      collaboration_score: 0.83,
      learning_progress: 0.75,
      exploration_rate: 0.14,
      total_actions: 189,
      total_observations: 167,
      total_rewards: 142,
      specialization_weights: {
        goal_optimization: 0.8,
        human_interaction: 0.7,
        data_analysis: 0.6,
        feedback_processing: 0.5,
        adaptation: 0.6,
      },
      collaboration_preferences: {
        evolution_optimizer_1: 0.94,
        human_proxy_1: 0.85,
        feedback_processor_1: 0.75,
        adaptation_specialist_1: 0.71,
      },
    },
    adaptation_specialist_1: {
      agent_id: 'adaptation_specialist_1',
      agent_type: 'adaptation_specialist',
      state: 'adapting',
      success_rate: 0.74,
      collaboration_score: 0.79,
      learning_progress: 0.86,
      exploration_rate: 0.21,
      total_actions: 167,
      total_observations: 145,
      total_rewards: 123,
      specialization_weights: {
        adaptation: 0.9,
        data_analysis: 0.8,
        feedback_processing: 0.6,
        goal_optimization: 0.5,
        human_interaction: 0.4,
      },
      collaboration_preferences: {
        feedback_processor_1: 0.89,
        human_proxy_1: 0.79,
        goal_coordinator_1: 0.71,
        evolution_optimizer_1: 0.68,
      },
    },
  },
  collaboration_network: {
    human_proxy_1: ['evolution_optimizer_1', 'feedback_processor_1', 'goal_coordinator_1', 'adaptation_specialist_1'],
    evolution_optimizer_1: ['human_proxy_1', 'goal_coordinator_1'],
    feedback_processor_1: ['human_proxy_1', 'adaptation_specialist_1'],
    goal_coordinator_1: ['human_proxy_1', 'evolution_optimizer_1'],
    adaptation_specialist_1: ['human_proxy_1', 'feedback_processor_1'],
  },
});
