import { useState, useCallback } from 'react';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';

interface EvolutionGoal {
  id: string;
  title: string;
  description: string;
  category: 'health' | 'cognitive' | 'longevity' | 'performance';
  priority: 'low' | 'medium' | 'high';
  targetValue: number;
  currentValue: number;
  unit: string;
  deadline: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'paused';
  milestones: Array<{
    id: string;
    title: string;
    targetDate: string;
    completed: boolean;
  }>;
}

interface EvolutionPlan {
  id: string;
  userId: string;
  goals: EvolutionGoal[];
  overallProgress: number;
  estimatedCompletion: string;
  riskFactors: string[];
  successProbability: number;
}

interface Recommendation {
  id: string;
  type: 'intervention' | 'optimization' | 'warning' | 'insight';
  title: string;
  description: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high';
  category: string;
  actionItems: string[];
  estimatedImpact: number;
  timeframe: string;
}

export const useEvolutionPlanning = (userId: string) => {
  const [evolutionPlan, setEvolutionPlan] = useState<EvolutionPlan | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  // Fetch evolution plan
  const { data: planData, isLoading: isPlanLoading } = useQuery(
    ['evolutionPlan', userId],
    async () => {
      try {
        const response = await axios.get(`/api/symbiotic/evolution/plan/${userId}`);
        return response.data;
      } catch (error) {
        // Return mock data if service is not available
        return generateMockEvolutionPlan(userId);
      }
    },
    {
      refetchInterval: 60000, // Refetch every minute
      onSuccess: (data) => {
        setEvolutionPlan(data);
      },
    }
  );

  // Fetch recommendations
  const { data: recommendationsData, isLoading: isRecommendationsLoading } = useQuery(
    ['evolutionRecommendations', userId],
    async () => {
      try {
        const response = await axios.get(`/api/symbiotic/evolution/recommendations/${userId}`);
        return response.data;
      } catch (error) {
        // Return mock recommendations if service is not available
        return generateMockRecommendations();
      }
    },
    {
      refetchInterval: 300000, // Refetch every 5 minutes
      onSuccess: (data) => {
        setRecommendations(data);
      },
    }
  );

  // Create evolution plan mutation
  const createPlanMutation = useMutation(
    async (planData: Partial<EvolutionPlan>) => {
      const response = await axios.post('/api/symbiotic/evolution/plan', {
        userId,
        ...planData,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setEvolutionPlan(data);
      },
      onError: (error) => {
        console.error('Failed to create evolution plan:', error);
      },
    }
  );

  // Update goal mutation
  const updateGoalMutation = useMutation(
    async ({ goalId, updates }: { goalId: string; updates: Partial<EvolutionGoal> }) => {
      const response = await axios.patch(`/api/symbiotic/evolution/goal/${goalId}`, updates);
      return response.data;
    },
    {
      onSuccess: (updatedGoal) => {
        setEvolutionPlan(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            goals: prev.goals.map(goal => 
              goal.id === updatedGoal.id ? updatedGoal : goal
            ),
          };
        });
      },
      onError: (error) => {
        console.error('Failed to update goal:', error);
      },
    }
  );

  // Get recommendations mutation
  const getRecommendationsMutation = useMutation(
    async (context: { goals: string[]; currentMetrics: Record<string, number> }) => {
      const response = await axios.post('/api/symbiotic/evolution/recommendations', {
        userId,
        ...context,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setRecommendations(data);
      },
      onError: (error) => {
        console.error('Failed to get recommendations:', error);
        // Set mock recommendations on error
        setRecommendations(generateMockRecommendations());
      },
    }
  );

  const createEvolutionPlan = useCallback(async (planData: Partial<EvolutionPlan>) => {
    return await createPlanMutation.mutateAsync(planData);
  }, [createPlanMutation]);

  const updateGoal = useCallback(async (goalId: string, updates: Partial<EvolutionGoal>) => {
    return await updateGoalMutation.mutateAsync({ goalId, updates });
  }, [updateGoalMutation]);

  const getRecommendations = useCallback(async (context: { goals: string[]; currentMetrics: Record<string, number> }) => {
    return await getRecommendationsMutation.mutateAsync(context);
  }, [getRecommendationsMutation]);

  return {
    evolutionPlan: evolutionPlan || planData,
    recommendations: recommendations.length > 0 ? recommendations : recommendationsData || [],
    isLoading: isPlanLoading || isRecommendationsLoading,
    createEvolutionPlan,
    updateGoal,
    getRecommendations,
  };
};

// Mock data generators
const generateMockEvolutionPlan = (userId: string): EvolutionPlan => ({
  id: `plan_${userId}`,
  userId,
  goals: [
    {
      id: '1',
      title: 'Reduce Biological Age',
      description: 'Decrease biological age by 2 years through lifestyle optimization',
      category: 'longevity',
      priority: 'high',
      targetValue: 26,
      currentValue: 28,
      unit: 'years',
      deadline: '2024-12-31',
      status: 'in_progress',
      milestones: [
        { id: '1a', title: 'Optimize sleep schedule', targetDate: '2024-09-15', completed: true },
        { id: '1b', title: 'Implement fasting protocol', targetDate: '2024-10-01', completed: false },
        { id: '1c', title: 'Advanced biomarker testing', targetDate: '2024-11-01', completed: false },
      ]
    },
    {
      id: '2',
      title: 'Enhance Cognitive Performance',
      description: 'Improve working memory and processing speed by 15%',
      category: 'cognitive',
      priority: 'high',
      targetValue: 95,
      currentValue: 82,
      unit: 'percentile',
      deadline: '2024-11-30',
      status: 'in_progress',
      milestones: [
        { id: '2a', title: 'Daily meditation practice', targetDate: '2024-09-01', completed: true },
        { id: '2b', title: 'Cognitive training program', targetDate: '2024-10-15', completed: false },
        { id: '2c', title: 'Nootropic optimization', targetDate: '2024-11-15', completed: false },
      ]
    },
  ],
  overallProgress: 67,
  estimatedCompletion: '2024-12-15',
  riskFactors: [
    'Stress levels may impact sleep optimization',
    'Genetic predisposition to slower cognitive enhancement',
  ],
  successProbability: 0.78,
});

const generateMockRecommendations = (): Recommendation[] => [
  {
    id: 'rec_1',
    type: 'intervention',
    title: 'Implement Time-Restricted Eating',
    description: 'Based on your current metabolic profile, implementing a 16:8 intermittent fasting protocol could accelerate biological age reduction.',
    confidence: 0.87,
    priority: 'high',
    category: 'longevity',
    actionItems: [
      'Start with 12:12 eating window for first week',
      'Gradually extend to 16:8 over 2 weeks',
      'Monitor glucose levels during transition',
      'Adjust eating window based on sleep quality',
    ],
    estimatedImpact: 0.23,
    timeframe: '2-4 weeks',
  },
  {
    id: 'rec_2',
    type: 'optimization',
    title: 'Enhance Meditation Practice',
    description: 'AI analysis suggests increasing meditation duration and adding specific techniques for cognitive enhancement.',
    confidence: 0.76,
    priority: 'medium',
    category: 'cognitive',
    actionItems: [
      'Extend daily meditation to 25 minutes',
      'Add focused attention training',
      'Include loving-kindness meditation twice weekly',
      'Use binaural beats for deeper states',
    ],
    estimatedImpact: 0.18,
    timeframe: '3-6 weeks',
  },
  {
    id: 'rec_3',
    type: 'insight',
    title: 'Cellular Regeneration Optimization',
    description: 'Your cellular simulation data indicates optimal timing for regenerative interventions.',
    confidence: 0.92,
    priority: 'medium',
    category: 'health',
    actionItems: [
      'Schedule heat shock therapy sessions',
      'Increase NAD+ precursor supplementation',
      'Optimize protein intake timing',
      'Consider red light therapy',
    ],
    estimatedImpact: 0.31,
    timeframe: '4-8 weeks',
  },
  {
    id: 'rec_4',
    type: 'warning',
    title: 'Stress Management Priority',
    description: 'Elevated cortisol patterns detected. Immediate stress management intervention recommended.',
    confidence: 0.84,
    priority: 'high',
    category: 'health',
    actionItems: [
      'Implement daily stress monitoring',
      'Add breathwork sessions',
      'Consider adaptogenic supplements',
      'Schedule stress assessment with healthcare provider',
    ],
    estimatedImpact: 0.42,
    timeframe: '1-2 weeks',
  },
];
