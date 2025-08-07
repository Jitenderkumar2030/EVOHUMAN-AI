import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  HeartIcon, 
  BrainIcon, 
  BeakerIcon, 
  ChartBarIcon,
  ClockIcon,
  FireIcon,
  ShieldCheckIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'chart.js';

import { useBioTwinData } from '../../hooks/useBioTwinData';
import { useWebSocket } from '../../hooks/useWebSocket';
import { BioMetricCard } from './BioMetricCard';
import { EvolutionTimeline } from './EvolutionTimeline';
import { CellularVisualization } from './CellularVisualization';
import { AIInsightsPanel } from './AIInsightsPanel';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
);

interface BioTwinDashboardProps {
  userId: string;
}

export const BioTwinDashboard: React.FC<BioTwinDashboardProps> = ({ userId }) => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [activeTab, setActiveTab] = useState('overview');
  
  // Fetch bio-twin data
  const { 
    bioTwinData, 
    isLoading, 
    error, 
    refetch 
  } = useBioTwinData(userId, selectedTimeRange);

  // WebSocket for real-time updates
  const { 
    isConnected, 
    lastMessage, 
    sendMessage 
  } = useWebSocket(`/bio-twin/${userId}`);

  // Real-time data state
  const [realTimeMetrics, setRealTimeMetrics] = useState<any>(null);

  useEffect(() => {
    if (lastMessage) {
      setRealTimeMetrics(lastMessage);
    }
  }, [lastMessage]);

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'cellular', name: 'Cellular', icon: BeakerIcon },
    { id: 'cognitive', name: 'Cognitive', icon: BrainIcon },
    { id: 'evolution', name: 'Evolution', icon: ArrowTrendingUpIcon },
  ];

  const timeRanges = [
    { id: '1d', name: '24 Hours' },
    { id: '7d', name: '7 Days' },
    { id: '30d', name: '30 Days' },
    { id: '90d', name: '90 Days' },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-red-500 text-center">
          <p className="text-xl font-semibold">Error loading bio-twin data</p>
          <p className="text-sm mt-2">{error.message}</p>
          <button 
            onClick={() => refetch()}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Bio-Digital Twin</h1>
              <p className="text-gray-600 mt-1">
                Real-time biological intelligence dashboard
                {isConnected && (
                  <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    <span className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
                    Live
                  </span>
                )}
              </p>
            </div>
            
            {/* Time Range Selector */}
            <div className="flex space-x-2">
              {timeRanges.map((range) => (
                <button
                  key={range.id}
                  onClick={() => setSelectedTimeRange(range.id)}
                  className={`px-3 py-2 text-sm font-medium rounded-md ${
                    selectedTimeRange === range.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {range.name}
                </button>
              ))}
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
        {activeTab === 'overview' && (
          <OverviewTab 
            bioTwinData={bioTwinData} 
            realTimeMetrics={realTimeMetrics}
          />
        )}
        
        {activeTab === 'cellular' && (
          <CellularTab 
            bioTwinData={bioTwinData}
            userId={userId}
          />
        )}
        
        {activeTab === 'cognitive' && (
          <CognitiveTab 
            bioTwinData={bioTwinData}
            userId={userId}
          />
        )}
        
        {activeTab === 'evolution' && (
          <EvolutionTab 
            bioTwinData={bioTwinData}
            userId={userId}
          />
        )}
      </div>
    </div>
  );
};

// Overview Tab Component
const OverviewTab: React.FC<{ bioTwinData: any; realTimeMetrics: any }> = ({ 
  bioTwinData, 
  realTimeMetrics 
}) => {
  const currentMetrics = realTimeMetrics || bioTwinData?.current_metrics || {};
  
  const keyMetrics = [
    {
      name: 'Biological Age',
      value: currentMetrics.biological_age || 28,
      unit: 'years',
      change: -0.5,
      icon: ClockIcon,
      color: 'blue',
    },
    {
      name: 'Health Score',
      value: currentMetrics.health_score || 87,
      unit: '%',
      change: +2.3,
      icon: HeartIcon,
      color: 'green',
    },
    {
      name: 'Cognitive Index',
      value: currentMetrics.cognitive_index || 92,
      unit: '%',
      change: +1.8,
      icon: BrainIcon,
      color: 'purple',
    },
    {
      name: 'Cellular Vitality',
      value: currentMetrics.cellular_vitality || 85,
      unit: '%',
      change: +0.7,
      icon: BeakerIcon,
      color: 'orange',
    },
    {
      name: 'Stress Resilience',
      value: currentMetrics.stress_resilience || 78,
      unit: '%',
      change: +3.2,
      icon: ShieldCheckIcon,
      color: 'indigo',
    },
    {
      name: 'Energy Level',
      value: currentMetrics.energy_level || 82,
      unit: '%',
      change: +1.5,
      icon: FireIcon,
      color: 'red',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {keyMetrics.map((metric, index) => (
          <motion.div
            key={metric.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <BioMetricCard metric={metric} />
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Health Trends Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Health Trends</h3>
          <div className="h-64">
            <Line
              data={{
                labels: bioTwinData?.timeline?.labels || ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
                datasets: [
                  {
                    label: 'Health Score',
                    data: bioTwinData?.timeline?.health_score || [82, 84, 83, 86, 87, 85, 87],
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    tension: 0.4,
                  },
                  {
                    label: 'Energy Level',
                    data: bioTwinData?.timeline?.energy_level || [78, 80, 79, 82, 84, 81, 82],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: false,
                    min: 70,
                    max: 100,
                  },
                },
              }}
            />
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
          <div className="h-64">
            <Doughnut
              data={{
                labels: ['Optimal', 'Good', 'Needs Attention'],
                datasets: [
                  {
                    data: [65, 25, 10],
                    backgroundColor: [
                      'rgb(34, 197, 94)',
                      'rgb(251, 191, 36)',
                      'rgb(239, 68, 68)',
                    ],
                    borderWidth: 0,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom' as const,
                  },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* AI Insights Panel */}
      <AIInsightsPanel insights={bioTwinData?.ai_insights} />
    </div>
  );
};

// Placeholder components for other tabs
const CellularTab: React.FC<{ bioTwinData: any; userId: string }> = ({ bioTwinData, userId }) => (
  <CellularVisualization data={bioTwinData?.cellular_data} userId={userId} />
);

const CognitiveTab: React.FC<{ bioTwinData: any; userId: string }> = ({ bioTwinData, userId }) => (
  <div>Cognitive Tab - Coming Soon</div>
);

const EvolutionTab: React.FC<{ bioTwinData: any; userId: string }> = ({ bioTwinData, userId }) => (
  <EvolutionTimeline data={bioTwinData?.evolution_data} userId={userId} />
);
