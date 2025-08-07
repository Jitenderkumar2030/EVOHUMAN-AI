import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import {
  SignalIcon,
  WifiIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  ChartBarIcon,
  HeartIcon,
  BrainIcon,
  BeakerIcon,
} from '@heroicons/react/24/outline';

import { useWebSocket } from '../../hooks/useWebSocket';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface RealTimeDataDashboardProps {
  userId: string;
}

interface DataPoint {
  timestamp: Date;
  value: number;
  type: string;
}

interface StreamingData {
  bioMetrics: DataPoint[];
  cellularData: DataPoint[];
  cognitiveData: DataPoint[];
  agentActivity: DataPoint[];
}

export const RealTimeDataDashboard: React.FC<RealTimeDataDashboardProps> = ({ userId }) => {
  const [isStreaming, setIsStreaming] = useState(true);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['health', 'cognitive', 'cellular']);
  const [timeWindow, setTimeWindow] = useState(300); // 5 minutes in seconds
  const [streamingData, setStreamingData] = useState<StreamingData>({
    bioMetrics: [],
    cellularData: [],
    cognitiveData: [],
    agentActivity: [],
  });

  const dataBufferRef = useRef<Map<string, DataPoint[]>>(new Map());
  const maxDataPoints = 100;

  // WebSocket connections for different data streams
  const bioMetricsWS = useWebSocket(`/bio-twin/${userId}`, { autoConnect: isStreaming });
  const cellularWS = useWebSocket(`/cellular/${userId}`, { autoConnect: isStreaming });
  const agentWS = useWebSocket(`/agents/${userId}`, { autoConnect: isStreaming });

  // Handle incoming real-time data
  useEffect(() => {
    if (bioMetricsWS.lastMessage?.type === 'bio_metrics') {
      const data = bioMetricsWS.lastMessage.data;
      addDataPoint('bioMetrics', data.health_score, 'health');
      addDataPoint('bioMetrics', data.energy_level, 'energy');
      addDataPoint('bioMetrics', data.stress_level, 'stress');
    }
  }, [bioMetricsWS.lastMessage]);

  useEffect(() => {
    if (cellularWS.lastMessage?.type === 'cellular_simulation') {
      const data = cellularWS.lastMessage.data;
      addDataPoint('cellularData', data.average_health * 100, 'cellular_health');
      addDataPoint('cellularData', data.total_cells, 'cell_count');
    }
  }, [cellularWS.lastMessage]);

  useEffect(() => {
    if (agentWS.lastMessage?.type === 'agent_system') {
      const data = agentWS.lastMessage.data;
      addDataPoint('agentActivity', data.system_performance * 100, 'performance');
      addDataPoint('agentActivity', data.actions_taken, 'actions');
    }
  }, [agentWS.lastMessage]);

  const addDataPoint = (category: keyof StreamingData, value: number, type: string) => {
    const now = new Date();
    const newPoint: DataPoint = { timestamp: now, value, type };

    setStreamingData(prev => {
      const updated = { ...prev };
      updated[category] = [...updated[category], newPoint].slice(-maxDataPoints);
      return updated;
    });
  };

  // Generate mock data for demonstration
  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(() => {
      // Mock bio metrics
      addDataPoint('bioMetrics', 85 + Math.random() * 10, 'health');
      addDataPoint('bioMetrics', 78 + Math.random() * 15, 'energy');
      addDataPoint('bioMetrics', 20 + Math.random() * 30, 'stress');

      // Mock cellular data
      addDataPoint('cellularData', 75 + Math.random() * 20, 'cellular_health');
      addDataPoint('cellularData', 1000 + Math.random() * 200, 'cell_count');

      // Mock cognitive data
      addDataPoint('cognitiveData', 88 + Math.random() * 12, 'cognitive_score');
      addDataPoint('cognitiveData', 92 + Math.random() * 8, 'memory_score');

      // Mock agent activity
      addDataPoint('agentActivity', 70 + Math.random() * 25, 'performance');
      addDataPoint('agentActivity', Math.floor(Math.random() * 5), 'actions');
    }, 2000);

    return () => clearInterval(interval);
  }, [isStreaming]);

  const toggleStreaming = () => {
    setIsStreaming(!isStreaming);
  };

  const clearData = () => {
    setStreamingData({
      bioMetrics: [],
      cellularData: [],
      cognitiveData: [],
      agentActivity: [],
    });
  };

  const getConnectionStatus = () => {
    const connections = [bioMetricsWS, cellularWS, agentWS];
    const connected = connections.filter(ws => ws.isConnected).length;
    const total = connections.length;
    return { connected, total, percentage: (connected / total) * 100 };
  };

  const connectionStatus = getConnectionStatus();

  const metricOptions = [
    { id: 'health', name: 'Health Metrics', icon: HeartIcon, color: 'red' },
    { id: 'cognitive', name: 'Cognitive Data', icon: BrainIcon, color: 'blue' },
    { id: 'cellular', name: 'Cellular Activity', icon: BeakerIcon, color: 'green' },
    { id: 'agents', name: 'AI Agents', icon: ChartBarIcon, color: 'purple' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Real-Time Data Stream</h1>
              <p className="text-gray-600 mt-1">
                Live bio-intelligence monitoring and visualization
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center text-sm">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  connectionStatus.connected === connectionStatus.total 
                    ? 'bg-green-400 animate-pulse' 
                    : connectionStatus.connected > 0 
                    ? 'bg-yellow-400' 
                    : 'bg-red-400'
                }`} />
                <span className="text-gray-600">
                  {connectionStatus.connected}/{connectionStatus.total} Connected
                </span>
              </div>

              {/* Time Window Selector */}
              <select
                value={timeWindow}
                onChange={(e) => setTimeWindow(parseInt(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value={60}>1 Minute</option>
                <option value={300}>5 Minutes</option>
                <option value={900}>15 Minutes</option>
                <option value={1800}>30 Minutes</option>
              </select>

              {/* Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={toggleStreaming}
                  className={`flex items-center px-4 py-2 rounded-lg text-white ${
                    isStreaming 
                      ? 'bg-red-600 hover:bg-red-700' 
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  {isStreaming ? (
                    <>
                      <PauseIcon className="w-4 h-4 mr-2" />
                      Pause
                    </>
                  ) : (
                    <>
                      <PlayIcon className="w-4 h-4 mr-2" />
                      Start
                    </>
                  )}
                </button>
                
                <button
                  onClick={clearData}
                  className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  <ArrowPathIcon className="w-4 h-4 mr-2" />
                  Clear
                </button>
              </div>
            </div>
          </div>

          {/* Metric Filters */}
          <div className="pb-4">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-700">Show Metrics:</span>
              <div className="flex space-x-2">
                {metricOptions.map((metric) => {
                  const Icon = metric.icon;
                  const isSelected = selectedMetrics.includes(metric.id);
                  return (
                    <button
                      key={metric.id}
                      onClick={() => {
                        setSelectedMetrics(prev => 
                          isSelected 
                            ? prev.filter(m => m !== metric.id)
                            : [...prev, metric.id]
                        );
                      }}
                      className={`flex items-center px-3 py-1 text-xs font-medium rounded-full border ${
                        isSelected
                          ? 'bg-blue-100 border-blue-300 text-blue-700'
                          : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="w-3 h-3 mr-1" />
                      {metric.name}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Dashboard */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Real-time Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Bio Metrics Chart */}
            {selectedMetrics.includes('health') && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-lg shadow p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                    <HeartIcon className="w-5 h-5 mr-2 text-red-500" />
                    Bio Metrics
                  </h3>
                  <div className="flex items-center text-sm text-gray-500">
                    <SignalIcon className="w-4 h-4 mr-1" />
                    Live
                  </div>
                </div>
                <div className="h-64">
                  <Line
                    data={{
                      datasets: [
                        {
                          label: 'Health Score',
                          data: streamingData.bioMetrics
                            .filter(d => d.type === 'health')
                            .map(d => ({ x: d.timestamp, y: d.value })),
                          borderColor: 'rgb(239, 68, 68)',
                          backgroundColor: 'rgba(239, 68, 68, 0.1)',
                          tension: 0.4,
                        },
                        {
                          label: 'Energy Level',
                          data: streamingData.bioMetrics
                            .filter(d => d.type === 'energy')
                            .map(d => ({ x: d.timestamp, y: d.value })),
                          borderColor: 'rgb(34, 197, 94)',
                          backgroundColor: 'rgba(34, 197, 94, 0.1)',
                          tension: 0.4,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      animation: { duration: 0 },
                      scales: {
                        x: {
                          type: 'time',
                          time: {
                            displayFormats: {
                              second: 'HH:mm:ss',
                              minute: 'HH:mm',
                            },
                          },
                        },
                        y: {
                          beginAtZero: true,
                          max: 100,
                        },
                      },
                      plugins: {
                        legend: {
                          position: 'top' as const,
                        },
                      },
                    }}
                  />
                </div>
              </motion.div>
            )}

            {/* Cellular Activity Chart */}
            {selectedMetrics.includes('cellular') && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-lg shadow p-6"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                    <BeakerIcon className="w-5 h-5 mr-2 text-green-500" />
                    Cellular Activity
                  </h3>
                  <div className="flex items-center text-sm text-gray-500">
                    <SignalIcon className="w-4 h-4 mr-1" />
                    Live
                  </div>
                </div>
                <div className="h-64">
                  <Line
                    data={{
                      datasets: [
                        {
                          label: 'Cellular Health',
                          data: streamingData.cellularData
                            .filter(d => d.type === 'cellular_health')
                            .map(d => ({ x: d.timestamp, y: d.value })),
                          borderColor: 'rgb(34, 197, 94)',
                          backgroundColor: 'rgba(34, 197, 94, 0.1)',
                          tension: 0.4,
                          yAxisID: 'y',
                        },
                        {
                          label: 'Cell Count',
                          data: streamingData.cellularData
                            .filter(d => d.type === 'cell_count')
                            .map(d => ({ x: d.timestamp, y: d.value })),
                          borderColor: 'rgb(59, 130, 246)',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          tension: 0.4,
                          yAxisID: 'y1',
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      animation: { duration: 0 },
                      scales: {
                        x: {
                          type: 'time',
                          time: {
                            displayFormats: {
                              second: 'HH:mm:ss',
                              minute: 'HH:mm',
                            },
                          },
                        },
                        y: {
                          type: 'linear',
                          display: true,
                          position: 'left',
                          beginAtZero: true,
                          max: 100,
                        },
                        y1: {
                          type: 'linear',
                          display: true,
                          position: 'right',
                          beginAtZero: true,
                          grid: {
                            drawOnChartArea: false,
                          },
                        },
                      },
                      plugins: {
                        legend: {
                          position: 'top' as const,
                        },
                      },
                    }}
                  />
                </div>
              </motion.div>
            )}
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* Connection Status Panel */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Connection Status</h3>
              <div className="space-y-3">
                <ConnectionStatusItem
                  name="Bio-Twin Service"
                  isConnected={bioMetricsWS.isConnected}
                  error={bioMetricsWS.error}
                />
                <ConnectionStatusItem
                  name="Cellular Service"
                  isConnected={cellularWS.isConnected}
                  error={cellularWS.error}
                />
                <ConnectionStatusItem
                  name="Agent System"
                  isConnected={agentWS.isConnected}
                  error={agentWS.error}
                />
              </div>
            </div>

            {/* Live Statistics */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Live Statistics</h3>
              <div className="space-y-4">
                <StatItem
                  label="Data Points"
                  value={Object.values(streamingData).reduce((sum, arr) => sum + arr.length, 0)}
                />
                <StatItem
                  label="Update Rate"
                  value={isStreaming ? "2.0 Hz" : "0 Hz"}
                />
                <StatItem
                  label="Buffer Size"
                  value={`${maxDataPoints} points`}
                />
                <StatItem
                  label="Time Window"
                  value={`${timeWindow / 60} min`}
                />
              </div>
            </div>

            {/* Current Values */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Values</h3>
              <div className="space-y-3">
                {streamingData.bioMetrics.length > 0 && (
                  <CurrentValueItem
                    label="Health Score"
                    value={streamingData.bioMetrics
                      .filter(d => d.type === 'health')
                      .slice(-1)[0]?.value || 0}
                    unit="%"
                    color="red"
                  />
                )}
                {streamingData.cellularData.length > 0 && (
                  <CurrentValueItem
                    label="Cellular Health"
                    value={streamingData.cellularData
                      .filter(d => d.type === 'cellular_health')
                      .slice(-1)[0]?.value || 0}
                    unit="%"
                    color="green"
                  />
                )}
                {streamingData.agentActivity.length > 0 && (
                  <CurrentValueItem
                    label="AI Performance"
                    value={streamingData.agentActivity
                      .filter(d => d.type === 'performance')
                      .slice(-1)[0]?.value || 0}
                    unit="%"
                    color="purple"
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper Components
const ConnectionStatusItem: React.FC<{
  name: string;
  isConnected: boolean;
  error: string | null;
}> = ({ name, isConnected, error }) => (
  <div className="flex items-center justify-between">
    <span className="text-sm text-gray-600">{name}</span>
    <div className="flex items-center">
      {isConnected ? (
        <>
          <WifiIcon className="w-4 h-4 text-green-500 mr-1" />
          <span className="text-xs text-green-600">Connected</span>
        </>
      ) : (
        <>
          <ExclamationTriangleIcon className="w-4 h-4 text-red-500 mr-1" />
          <span className="text-xs text-red-600">
            {error ? 'Error' : 'Disconnected'}
          </span>
        </>
      )}
    </div>
  </div>
);

const StatItem: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <div className="flex justify-between">
    <span className="text-sm text-gray-600">{label}</span>
    <span className="text-sm font-semibold text-gray-900">{value}</span>
  </div>
);

const CurrentValueItem: React.FC<{
  label: string;
  value: number;
  unit: string;
  color: string;
}> = ({ label, value, unit, color }) => {
  const colorClasses = {
    red: 'text-red-600',
    green: 'text-green-600',
    blue: 'text-blue-600',
    purple: 'text-purple-600',
  };

  return (
    <div className="flex justify-between items-center">
      <span className="text-sm text-gray-600">{label}</span>
      <span className={`text-sm font-bold ${colorClasses[color as keyof typeof colorClasses]}`}>
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
};
