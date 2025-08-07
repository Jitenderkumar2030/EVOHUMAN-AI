import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BeakerIcon,
  DocumentTextIcon,
  ChartBarIcon,
  CubeIcon,
  PlayIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

import { useProteinAnalysis } from '../../hooks/useProteinAnalysis';
import { SequenceEditor } from './SequenceEditor';
import { StructureViewer } from './StructureViewer';
import { MutationAnalysis } from './MutationAnalysis';
import { EvolutionPathways } from './EvolutionPathways';
import { BatchProcessor } from './BatchProcessor';

interface ProteinAnalysisInterfaceProps {
  userId: string;
}

export const ProteinAnalysisInterface: React.FC<ProteinAnalysisInterfaceProps> = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('sequence');
  const [currentSequence, setCurrentSequence] = useState('');
  const [analysisHistory, setAnalysisHistory] = useState<any[]>([]);

  const {
    analysisResult,
    isAnalyzing,
    error,
    analyzeSequence,
    predictMutations,
    analyzeEvolution,
    clearResults
  } = useProteinAnalysis();

  const tabs = [
    { id: 'sequence', name: 'Sequence Analysis', icon: DocumentTextIcon },
    { id: 'structure', name: '3D Structure', icon: CubeIcon },
    { id: 'mutations', name: 'Mutations', icon: BeakerIcon },
    { id: 'evolution', name: 'Evolution', icon: ChartBarIcon },
    { id: 'batch', name: 'Batch Processing', icon: ArrowPathIcon },
  ];

  const handleAnalyzeSequence = async () => {
    if (!currentSequence.trim()) return;

    try {
      const result = await analyzeSequence({
        sequence: currentSequence,
        analysis_type: 'structure_prediction',
        include_mutations: true,
        include_evolution: true,
      });

      setAnalysisHistory(prev => [result, ...prev.slice(0, 9)]); // Keep last 10
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const exampleSequences = [
    {
      name: 'Human Insulin',
      sequence: 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
      description: 'Human insulin precursor protein'
    },
    {
      name: 'Green Fluorescent Protein',
      sequence: 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
      description: 'Aequorea victoria green fluorescent protein'
    },
    {
      name: 'Human Hemoglobin Alpha',
      sequence: 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR',
      description: 'Human hemoglobin alpha chain'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Protein Analysis</h1>
              <p className="text-gray-600 mt-1">
                Advanced protein structure prediction and evolution analysis
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {analysisResult && (
                <div className="flex items-center text-sm text-green-600">
                  <CheckCircleIcon className="w-4 h-4 mr-1" />
                  Analysis Complete
                </div>
              )}
              
              {isAnalyzing && (
                <div className="flex items-center text-sm text-blue-600">
                  <ClockIcon className="w-4 h-4 mr-1 animate-spin" />
                  Analyzing...
                </div>
              )}
              
              <button
                onClick={clearResults}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Clear Results
              </button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-2" />
                    {tab.name}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sequence Input Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6 sticky top-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Protein Sequence</h3>
              
              {/* Example Sequences */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Example Sequences
                </label>
                <select
                  onChange={(e) => {
                    const selected = exampleSequences[parseInt(e.target.value)];
                    if (selected) {
                      setCurrentSequence(selected.sequence);
                    }
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">Select an example...</option>
                  {exampleSequences.map((seq, index) => (
                    <option key={index} value={index}>
                      {seq.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sequence Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Amino Acid Sequence
                </label>
                <textarea
                  value={currentSequence}
                  onChange={(e) => setCurrentSequence(e.target.value.toUpperCase())}
                  placeholder="Enter protein sequence (single letter amino acid codes)..."
                  className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md text-sm font-mono"
                  style={{ resize: 'vertical' }}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Length: {currentSequence.length} amino acids
                </p>
              </div>

              {/* Analysis Button */}
              <button
                onClick={handleAnalyzeSequence}
                disabled={!currentSequence.trim() || isAnalyzing}
                className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <ArrowPathIcon className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4 mr-2" />
                    Analyze Protein
                  </>
                )}
              </button>

              {/* Error Display */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"
                >
                  <div className="flex items-center">
                    <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </motion.div>
              )}

              {/* Analysis History */}
              {analysisHistory.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Recent Analyses</h4>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {analysisHistory.map((analysis, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentSequence(analysis.sequence)}
                        className="w-full text-left p-2 text-xs bg-gray-50 hover:bg-gray-100 rounded border"
                      >
                        <div className="font-medium truncate">
                          {analysis.sequence.substring(0, 20)}...
                        </div>
                        <div className="text-gray-500">
                          Confidence: {(analysis.confidence_score * 100).toFixed(1)}%
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Main Analysis Panel */}
          <div className="lg:col-span-3">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                {activeTab === 'sequence' && (
                  <SequenceAnalysisTab 
                    analysisResult={analysisResult}
                    sequence={currentSequence}
                  />
                )}
                
                {activeTab === 'structure' && (
                  <StructureViewer 
                    analysisResult={analysisResult}
                    sequence={currentSequence}
                  />
                )}
                
                {activeTab === 'mutations' && (
                  <MutationAnalysis 
                    analysisResult={analysisResult}
                    sequence={currentSequence}
                    onPredictMutations={predictMutations}
                  />
                )}
                
                {activeTab === 'evolution' && (
                  <EvolutionPathways 
                    analysisResult={analysisResult}
                    sequence={currentSequence}
                    onAnalyzeEvolution={analyzeEvolution}
                  />
                )}
                
                {activeTab === 'batch' && (
                  <BatchProcessor userId={userId} />
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
};

// Sequence Analysis Tab Component
const SequenceAnalysisTab: React.FC<{ 
  analysisResult: any; 
  sequence: string; 
}> = ({ analysisResult, sequence }) => {
  if (!analysisResult) {
    return (
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <BeakerIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Results</h3>
        <p className="text-gray-500">
          Enter a protein sequence and click "Analyze Protein" to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Analysis Summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {(analysisResult.confidence_score * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-blue-800">Confidence Score</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {analysisResult.sequence_length}
            </div>
            <div className="text-sm text-green-800">Amino Acids</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {analysisResult.processing_time?.toFixed(2)}s
            </div>
            <div className="text-sm text-purple-800">Processing Time</div>
          </div>
        </div>
      </div>

      {/* Structure Prediction */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Structure Prediction</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-gray-700">{analysisResult.predicted_structure}</p>
        </div>
      </div>

      {/* Sequence Visualization */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Sequence Visualization</h3>
        <SequenceEditor 
          sequence={sequence}
          analysisResult={analysisResult}
          readOnly={true}
        />
      </div>
    </div>
  );
};
