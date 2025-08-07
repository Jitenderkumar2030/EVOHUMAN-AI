import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Box } from '@react-three/drei';
import { Bar, Line } from 'react-chartjs-2';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon,
  AdjustmentsHorizontalIcon,
  EyeIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';

import { useCellularSimulation } from '../../hooks/useCellularSimulation';

interface CellularVisualizationProps {
  data?: any;
  userId: string;
}

export const CellularVisualization: React.FC<CellularVisualizationProps> = ({ 
  data, 
  userId 
}) => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [viewMode, setViewMode] = useState<'3d' | '2d' | 'charts'>('3d');
  const [selectedCellType, setSelectedCellType] = useState<string>('all');

  const {
    simulationData,
    isLoading,
    startSimulation,
    stopSimulation,
    updateSimulationParams
  } = useCellularSimulation(userId);

  const cellTypes = [
    { id: 'all', name: 'All Cells', color: '#6B7280' },
    { id: 'stem', name: 'Stem Cells', color: '#10B981' },
    { id: 'neural', name: 'Neural Cells', color: '#3B82F6' },
    { id: 'cardiac', name: 'Cardiac Cells', color: '#EF4444' },
    { id: 'hepatic', name: 'Hepatic Cells', color: '#F59E0B' },
    { id: 'muscle', name: 'Muscle Cells', color: '#8B5CF6' },
  ];

  const handleStartSimulation = async () => {
    setIsSimulating(true);
    await startSimulation({
      tissue_type: 'neural',
      initial_cell_count: 1000,
      simulation_steps: 100,
    });
  };

  const handleStopSimulation = () => {
    setIsSimulating(false);
    stopSimulation();
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <BeakerIcon className="w-6 h-6 mr-2 text-blue-600" />
            Cellular Simulation
          </h2>
          
          <div className="flex items-center space-x-4">
            {/* View Mode Toggle */}
            <div className="flex rounded-lg border border-gray-300">
              {[
                { id: '3d', name: '3D View' },
                { id: '2d', name: '2D View' },
                { id: 'charts', name: 'Charts' },
              ].map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setViewMode(mode.id as any)}
                  className={`px-3 py-2 text-sm font-medium ${
                    viewMode === mode.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {mode.name}
                </button>
              ))}
            </div>

            {/* Simulation Controls */}
            <div className="flex items-center space-x-2">
              {!isSimulating ? (
                <button
                  onClick={handleStartSimulation}
                  disabled={isLoading}
                  className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                >
                  <PlayIcon className="w-4 h-4 mr-2" />
                  Start
                </button>
              ) : (
                <button
                  onClick={handleStopSimulation}
                  className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  <StopIcon className="w-4 h-4 mr-2" />
                  Stop
                </button>
              )}
              
              <button className="p-2 text-gray-400 hover:text-gray-600">
                <AdjustmentsHorizontalIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Cell Type Filter */}
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700">Cell Type:</span>
          <div className="flex space-x-2">
            {cellTypes.map((type) => (
              <button
                key={type.id}
                onClick={() => setSelectedCellType(type.id)}
                className={`px-3 py-1 text-xs font-medium rounded-full border ${
                  selectedCellType === type.id
                    ? 'bg-blue-100 border-blue-300 text-blue-700'
                    : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                }`}
              >
                <span
                  className="inline-block w-2 h-2 rounded-full mr-2"
                  style={{ backgroundColor: type.color }}
                />
                {type.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 3D/2D Visualization */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="h-96">
              {viewMode === '3d' && (
                <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                  <ambientLight intensity={0.5} />
                  <pointLight position={[10, 10, 10]} />
                  <CellularScene 
                    simulationData={simulationData}
                    selectedCellType={selectedCellType}
                  />
                  <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
                </Canvas>
              )}
              
              {viewMode === '2d' && (
                <Cellular2DView 
                  simulationData={simulationData}
                  selectedCellType={selectedCellType}
                />
              )}
              
              {viewMode === 'charts' && (
                <CellularCharts simulationData={simulationData} />
              )}
            </div>
          </div>
        </div>

        {/* Statistics Panel */}
        <div className="space-y-6">
          {/* Current Stats */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Live Statistics</h3>
            <div className="space-y-4">
              {simulationData?.cell_type_distribution && Object.entries(simulationData.cell_type_distribution).map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 capitalize">{type.replace('_', ' ')}</span>
                  <span className="font-semibold text-gray-900">{count as number}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Cell Health Metrics */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Health Metrics</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Average Health</span>
                <div className="flex items-center">
                  <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${(simulationData?.average_cell_health || 0.75) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold">
                    {((simulationData?.average_cell_health || 0.75) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Average Energy</span>
                <div className="flex items-center">
                  <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(simulationData?.average_cell_energy || 0.68) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold">
                    {((simulationData?.average_cell_energy || 0.68) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Simulation Progress */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Progress</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Steps Completed</span>
                <span className="font-semibold">
                  {simulationData?.steps_completed || 0} / 100
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((simulationData?.steps_completed || 0) / 100) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// 3D Scene Component
const CellularScene: React.FC<{ simulationData: any; selectedCellType: string }> = ({ 
  simulationData, 
  selectedCellType 
}) => {
  const cellTypeColors = {
    stem: '#10B981',
    neural: '#3B82F6',
    cardiac: '#EF4444',
    hepatic: '#F59E0B',
    muscle: '#8B5CF6',
  };

  // Generate cell positions (mock data for visualization)
  const cells = React.useMemo(() => {
    const cellArray = [];
    const cellTypes = Object.keys(cellTypeColors);
    
    for (let i = 0; i < 100; i++) {
      const cellType = cellTypes[Math.floor(Math.random() * cellTypes.length)];
      cellArray.push({
        id: i,
        type: cellType,
        position: [
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 10,
        ],
        size: Math.random() * 0.3 + 0.1,
        health: Math.random(),
      });
    }
    return cellArray;
  }, [simulationData]);

  return (
    <>
      {cells
        .filter(cell => selectedCellType === 'all' || cell.type === selectedCellType)
        .map((cell) => (
          <Sphere
            key={cell.id}
            position={cell.position as [number, number, number]}
            args={[cell.size, 16, 16]}
          >
            <meshStandardMaterial 
              color={cellTypeColors[cell.type as keyof typeof cellTypeColors]}
              opacity={0.7 + cell.health * 0.3}
              transparent
            />
          </Sphere>
        ))}
      
      {/* Grid */}
      <gridHelper args={[10, 10]} />
      
      {/* Labels */}
      <Text
        position={[0, 5, 0]}
        fontSize={0.5}
        color="black"
        anchorX="center"
        anchorY="middle"
      >
        Cellular Environment
      </Text>
    </>
  );
};

// 2D View Component
const Cellular2DView: React.FC<{ simulationData: any; selectedCellType: string }> = ({ 
  simulationData, 
  selectedCellType 
}) => {
  return (
    <div className="w-full h-full bg-gray-100 rounded-lg flex items-center justify-center">
      <div className="text-center">
        <EyeIcon className="w-12 h-12 text-gray-400 mx-auto mb-2" />
        <p className="text-gray-500">2D Cellular View</p>
        <p className="text-sm text-gray-400">Coming Soon</p>
      </div>
    </div>
  );
};

// Charts Component
const CellularCharts: React.FC<{ simulationData: any }> = ({ simulationData }) => {
  const chartData = {
    labels: ['Stem', 'Neural', 'Cardiac', 'Hepatic', 'Muscle'],
    datasets: [
      {
        label: 'Cell Count',
        data: [
          simulationData?.cell_type_distribution?.STEM || 100,
          simulationData?.cell_type_distribution?.NEURAL || 400,
          simulationData?.cell_type_distribution?.CARDIAC || 200,
          simulationData?.cell_type_distribution?.HEPATIC || 200,
          simulationData?.cell_type_distribution?.MUSCLE || 100,
        ],
        backgroundColor: [
          '#10B981',
          '#3B82F6',
          '#EF4444',
          '#F59E0B',
          '#8B5CF6',
        ],
      },
    ],
  };

  return (
    <div className="h-full">
      <Bar
        data={chartData}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false,
            },
            title: {
              display: true,
              text: 'Cell Type Distribution',
            },
          },
        }}
      />
    </div>
  );
};
