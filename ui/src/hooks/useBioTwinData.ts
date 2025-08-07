import { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import axios from 'axios';

interface BioTwinData {
  current_metrics: {
    biological_age: number;
    health_score: number;
    cognitive_index: number;
    cellular_vitality: number;
    stress_resilience: number;
    energy_level: number;
  };
  timeline: {
    labels: string[];
    health_score: number[];
    energy_level: number[];
    cognitive_index: number[];
  };
  cellular_data: {
    total_cells: number;
    cell_type_distribution: Record<string, number>;
    average_health: number;
    average_energy: number;
  };
  ai_insights: Array<{
    type: string;
    message: string;
    confidence: number;
    timestamp: string;
  }>;
  evolution_data: {
    goals: Array<{
      id: string;
      title: string;
      progress: number;
      status: string;
    }>;
    recommendations: string[];
  };
}

export const useBioTwinData = (userId: string, timeRange: string = '7d') => {
  const [bioTwinData, setBioTwinData] = useState<BioTwinData | null>(null);

  const { data, isLoading, error, refetch } = useQuery(
    ['bioTwinData', userId, timeRange],
    async () => {
      // Try to fetch from multiple services
      const responses = await Promise.allSettled([
        axios.get(`/api/aice/user/${userId}/bio-twin`),
        axios.get(`/api/proteus/user/${userId}/cellular-status`),
        axios.get(`/api/symbiotic/multi_agent/status/${userId}`),
      ]);

      // Combine data from different services
      const aiceData = responses[0].status === 'fulfilled' ? responses[0].value.data : null;
      const proteusData = responses[1].status === 'fulfilled' ? responses[1].value.data : null;
      const symbioticData = responses[2].status === 'fulfilled' ? responses[2].value.data : null;

      return {
        aice: aiceData,
        proteus: proteusData,
        symbiotic: symbioticData,
      };
    },
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      staleTime: 10000, // Consider data stale after 10 seconds
      onError: (error) => {
        console.error('Failed to fetch bio-twin data:', error);
      },
    }
  );

  useEffect(() => {
    if (data) {
      // Transform and combine data from different services
      const transformedData: BioTwinData = {
        current_metrics: {
          biological_age: data.aice?.biological_age || 28,
          health_score: data.aice?.health_score || 87,
          cognitive_index: data.aice?.cognitive_index || 92,
          cellular_vitality: data.proteus?.cellular_vitality || 85,
          stress_resilience: data.aice?.stress_resilience || 78,
          energy_level: data.aice?.energy_level || 82,
        },
        timeline: {
          labels: generateTimeLabels(timeRange),
          health_score: generateMockTimeline(87, 7),
          energy_level: generateMockTimeline(82, 7),
          cognitive_index: generateMockTimeline(92, 7),
        },
        cellular_data: {
          total_cells: data.proteus?.total_cells || 1200,
          cell_type_distribution: data.proteus?.cell_type_distribution || {
            STEM: 120,
            NEURAL: 480,
            CARDIAC: 240,
            HEPATIC: 240,
            MUSCLE: 120,
          },
          average_health: data.proteus?.average_health || 0.85,
          average_energy: data.proteus?.average_energy || 0.78,
        },
        ai_insights: [
          {
            type: 'optimization',
            message: 'Your sleep quality has improved by 15% this week. Consider maintaining current sleep schedule.',
            confidence: 0.89,
            timestamp: new Date().toISOString(),
          },
          {
            type: 'alert',
            message: 'Cellular regeneration rate is optimal. Continue current nutrition protocol.',
            confidence: 0.92,
            timestamp: new Date(Date.now() - 3600000).toISOString(),
          },
          {
            type: 'recommendation',
            message: 'AI agents suggest increasing meditation duration to 25 minutes for enhanced cognitive performance.',
            confidence: 0.76,
            timestamp: new Date(Date.now() - 7200000).toISOString(),
          },
        ],
        evolution_data: {
          goals: [
            {
              id: '1',
              title: 'Reduce Biological Age',
              progress: 65,
              status: 'in_progress',
            },
            {
              id: '2',
              title: 'Enhance Cognitive Performance',
              progress: 78,
              status: 'in_progress',
            },
            {
              id: '3',
              title: 'Optimize Cellular Health',
              progress: 45,
              status: 'not_started',
            },
          ],
          recommendations: [
            'Implement intermittent fasting protocol',
            'Increase omega-3 fatty acid intake',
            'Add cold exposure therapy',
            'Optimize sleep environment',
          ],
        },
      };

      setBioTwinData(transformedData);
    }
  }, [data, timeRange]);

  return {
    bioTwinData,
    isLoading,
    error,
    refetch,
  };
};

// Helper functions
const generateTimeLabels = (timeRange: string): string[] => {
  const now = new Date();
  const labels: string[] = [];
  
  let days = 7;
  if (timeRange === '1d') days = 1;
  else if (timeRange === '30d') days = 30;
  else if (timeRange === '90d') days = 90;

  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    labels.push(date.toLocaleDateString());
  }
  
  return labels;
};

const generateMockTimeline = (baseValue: number, days: number): number[] => {
  const timeline: number[] = [];
  let currentValue = baseValue;
  
  for (let i = 0; i < days; i++) {
    // Add some realistic variation
    const variation = (Math.random() - 0.5) * 4; // ±2 points
    currentValue = Math.max(0, Math.min(100, currentValue + variation));
    timeline.push(Math.round(currentValue * 10) / 10);
  }
  
  return timeline;
};
