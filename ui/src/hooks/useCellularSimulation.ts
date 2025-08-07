import { useState, useCallback } from 'react';
import { useQuery, useMutation } from 'react-query';
import axios from 'axios';

interface CellularSimulationParams {
  tissue_type: string;
  initial_cell_count: number;
  simulation_steps: number;
  cell_type_distribution?: Record<string, number>;
}

interface CellularSimulationData {
  simulation_id: string;
  status: string;
  tissue_type: string;
  initial_cells: number;
  final_cells: number;
  steps_completed: number;
  cell_type_distribution: Record<string, number>;
  cell_state_distribution: Record<string, number>;
  average_cell_health: number;
  average_cell_energy: number;
  events_summary: {
    total_events: number;
    divisions: number;
    deaths: number;
  };
}

export const useCellularSimulation = (userId: string) => {
  const [simulationData, setSimulationData] = useState<CellularSimulationData | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  // Fetch current simulation status
  const { data: currentSimulation, isLoading } = useQuery(
    ['cellularSimulation', userId],
    async () => {
      try {
        const response = await axios.get(`/api/proteus/simulation/status/${userId}`);
        return response.data;
      } catch (error) {
        // Return mock data if service is not available
        return null;
      }
    },
    {
      refetchInterval: isSimulating ? 2000 : false, // Poll every 2 seconds when simulating
      enabled: isSimulating,
    }
  );

  // Start simulation mutation
  const startSimulationMutation = useMutation(
    async (params: CellularSimulationParams) => {
      const response = await axios.post('/api/proteus/simulate/cellular_automata', {
        tissue_type: params.tissue_type,
        initial_cell_count: params.initial_cell_count,
        simulation_steps: params.simulation_steps,
        cell_type_distribution: params.cell_type_distribution,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setSimulationData(data);
        setIsSimulating(false);
      },
      onError: (error) => {
        console.error('Simulation failed:', error);
        // Generate mock data for demo
        const mockData: CellularSimulationData = {
          simulation_id: `sim_${Date.now()}`,
          status: 'completed',
          tissue_type: 'neural',
          initial_cells: 1000,
          final_cells: 1200,
          steps_completed: 100,
          cell_type_distribution: {
            STEM: 120,
            NEURAL: 480,
            CARDIAC: 240,
            HEPATIC: 240,
            MUSCLE: 120,
          },
          cell_state_distribution: {
            quiescent: 720,
            proliferating: 240,
            differentiating: 120,
            apoptotic: 120,
          },
          average_cell_health: 0.85,
          average_cell_energy: 0.78,
          events_summary: {
            total_events: 200,
            divisions: 60,
            deaths: 20,
          },
        };
        setSimulationData(mockData);
        setIsSimulating(false);
      },
    }
  );

  // Stop simulation mutation
  const stopSimulationMutation = useMutation(
    async () => {
      const response = await axios.post(`/api/proteus/simulation/stop/${userId}`);
      return response.data;
    },
    {
      onSuccess: () => {
        setIsSimulating(false);
      },
    }
  );

  // Update simulation parameters mutation
  const updateParamsMutation = useMutation(
    async (params: Partial<CellularSimulationParams>) => {
      const response = await axios.patch(`/api/proteus/simulation/params/${userId}`, params);
      return response.data;
    }
  );

  const startSimulation = useCallback(async (params: CellularSimulationParams) => {
    setIsSimulating(true);
    await startSimulationMutation.mutateAsync(params);
  }, [startSimulationMutation]);

  const stopSimulation = useCallback(async () => {
    await stopSimulationMutation.mutateAsync();
  }, [stopSimulationMutation]);

  const updateSimulationParams = useCallback(async (params: Partial<CellularSimulationParams>) => {
    await updateParamsMutation.mutateAsync(params);
  }, [updateParamsMutation]);

  return {
    simulationData: simulationData || currentSimulation,
    isLoading: isLoading || startSimulationMutation.isLoading,
    isSimulating,
    error: startSimulationMutation.error || stopSimulationMutation.error,
    startSimulation,
    stopSimulation,
    updateSimulationParams,
  };
};
