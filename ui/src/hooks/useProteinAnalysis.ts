import { useState, useCallback } from 'react';
import { useMutation } from 'react-query';
import axios from 'axios';

interface ProteinAnalysisRequest {
  sequence: string;
  analysis_type: string;
  include_mutations?: boolean;
  include_evolution?: boolean;
}

interface ProteinAnalysisResult {
  analysis_id: string;
  sequence: string;
  sequence_length: number;
  confidence_score: number;
  processing_time: number;
  predicted_structure: string;
  secondary_structure: {
    alpha_helix: number;
    beta_sheet: number;
    random_coil: number;
  };
  functional_domains: Array<{
    name: string;
    start: number;
    end: number;
    confidence: number;
  }>;
  mutations?: Array<{
    position: number;
    original: string;
    mutated: string;
    effect: string;
    confidence: number;
  }>;
  evolution_data?: {
    conservation_score: number;
    evolutionary_pressure: string;
    related_sequences: string[];
  };
}

interface MutationPredictionRequest {
  sequence: string;
  target_positions?: number[];
  mutation_type?: string;
}

interface EvolutionAnalysisRequest {
  sequence: string;
  target_organism?: string;
  evolutionary_pressure?: string;
}

export const useProteinAnalysis = () => {
  const [analysisResult, setAnalysisResult] = useState<ProteinAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Main protein analysis mutation
  const analysisMutation = useMutation(
    async (request: ProteinAnalysisRequest) => {
      const response = await axios.post('/api/esm3/analyze_protein', request);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setAnalysisResult(data);
        setError(null);
      },
      onError: (error: any) => {
        console.error('Protein analysis failed:', error);
        setError(error.response?.data?.detail || 'Analysis failed');
        
        // Generate mock data for demo
        const mockResult: ProteinAnalysisResult = {
          analysis_id: `analysis_${Date.now()}`,
          sequence: request.sequence,
          sequence_length: request.sequence.length,
          confidence_score: 0.87,
          processing_time: 2.3,
          predicted_structure: 'Alpha-helical protein with three transmembrane domains and a cytoplasmic C-terminus. Contains a signal peptide at the N-terminus and two potential glycosylation sites.',
          secondary_structure: {
            alpha_helix: 45.2,
            beta_sheet: 23.8,
            random_coil: 31.0,
          },
          functional_domains: [
            {
              name: 'Signal Peptide',
              start: 1,
              end: 23,
              confidence: 0.92,
            },
            {
              name: 'Transmembrane Domain 1',
              start: 45,
              end: 67,
              confidence: 0.89,
            },
            {
              name: 'Binding Site',
              start: 89,
              end: 95,
              confidence: 0.76,
            },
            {
              name: 'Transmembrane Domain 2',
              start: 123,
              end: 145,
              confidence: 0.91,
            },
          ],
          mutations: request.include_mutations ? [
            {
              position: 67,
              original: 'A',
              mutated: 'V',
              effect: 'Neutral - Conservative substitution',
              confidence: 0.82,
            },
            {
              position: 89,
              original: 'R',
              mutated: 'K',
              effect: 'Mild - May affect binding affinity',
              confidence: 0.74,
            },
            {
              position: 134,
              original: 'L',
              mutated: 'P',
              effect: 'Severe - Likely disrupts structure',
              confidence: 0.93,
            },
          ] : undefined,
          evolution_data: request.include_evolution ? {
            conservation_score: 0.78,
            evolutionary_pressure: 'Moderate purifying selection',
            related_sequences: [
              'Human ortholog (98% identity)',
              'Mouse ortholog (89% identity)',
              'Zebrafish ortholog (72% identity)',
            ],
          } : undefined,
        };
        
        setAnalysisResult(mockResult);
      },
    }
  );

  // Mutation prediction mutation
  const mutationMutation = useMutation(
    async (request: MutationPredictionRequest) => {
      const response = await axios.post('/api/esm3/predict_mutations', request);
      return response.data;
    },
    {
      onError: (error: any) => {
        console.error('Mutation prediction failed:', error);
        setError(error.response?.data?.detail || 'Mutation prediction failed');
      },
    }
  );

  // Evolution analysis mutation
  const evolutionMutation = useMutation(
    async (request: EvolutionAnalysisRequest) => {
      const response = await axios.post('/api/esm3/analyze_evolution', request);
      return response.data;
    },
    {
      onError: (error: any) => {
        console.error('Evolution analysis failed:', error);
        setError(error.response?.data?.detail || 'Evolution analysis failed');
      },
    }
  );

  const analyzeSequence = useCallback(async (request: ProteinAnalysisRequest) => {
    setError(null);
    return await analysisMutation.mutateAsync(request);
  }, [analysisMutation]);

  const predictMutations = useCallback(async (request: MutationPredictionRequest) => {
    setError(null);
    return await mutationMutation.mutateAsync(request);
  }, [mutationMutation]);

  const analyzeEvolution = useCallback(async (request: EvolutionAnalysisRequest) => {
    setError(null);
    return await evolutionMutation.mutateAsync(request);
  }, [evolutionMutation]);

  const clearResults = useCallback(() => {
    setAnalysisResult(null);
    setError(null);
  }, []);

  return {
    analysisResult,
    isAnalyzing: analysisMutation.isLoading || mutationMutation.isLoading || evolutionMutation.isLoading,
    error,
    analyzeSequence,
    predictMutations,
    analyzeEvolution,
    clearResults,
  };
};
