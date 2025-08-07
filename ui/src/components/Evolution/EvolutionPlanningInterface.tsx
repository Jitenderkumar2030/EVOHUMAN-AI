import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  RocketLaunchIcon,
  ChartBarIcon,
  ClockIcon,
  StarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  PlusIcon,
  TrashIcon,
  AdjustmentsHorizontalIcon,
  ArrowTrendingUpIcon,
  HeartIcon,
  BrainIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { Line, Radar, Bar } from 'react-chartjs-2';

import { useEvolutionPlanning } from '../../hooks/useEvolutionPlanning';
import { useMultiAgentSystem } from '../../hooks/useMultiAgentSystem';
import { GoalCard } from './GoalCard';
import { ProgressTimeline } from './ProgressTimeline';
import { RecommendationPanel } from './RecommendationPanel';
import { MetricsOverview } from './MetricsOverview';

interface EvolutionPlanningInterfaceProps {
  userId: string;
}

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

export const EvolutionPlanningInterface: React.FC<EvolutionPlanningInterfaceProps> = ({ userId }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [goals, setGoals] = useState<EvolutionGoal[]>([]);
  const [showAddGoal, setShowAddGoal] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('3months');

  const {
    evolutionPlan,
    recommendations,
    isLoading,
    createEvolutionPlan,
    updateGoal,
    getRecommendations
  } = useEvolutionPlanning(userId);

  const {
    agentStatus,
    executeStep,
    processHumanFeedback
  } = useMultiAgentSystem(userId);

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'goals', name: 'Goals', icon: StarIcon },
    { id: 'progress', name: 'Progress', icon: ArrowTrendingUpIcon },
    { id: 'recommendations', name: 'AI Insights', icon: BrainIcon },
  ];

  const timeframes = [
    { id: '1month', name: '1 Month', days: 30 },
    { id: '3months', name: '3 Months', days: 90 },
    { id: '6months', name: '6 Months', days: 180 },
    { id: '1year', name: '1 Year', days: 365 },
  ];

  const goalCategories = [
    { id: 'health', name: 'Health & Wellness', icon: HeartIcon, color: 'red' },
    { id: 'cognitive', name: 'Cognitive Enhancement', icon: BrainIcon, color: 'blue' },
    { id: 'longevity', name: 'Longevity', icon: ClockIcon, color: 'green' },
    { id: 'performance', name: 'Performance', icon: RocketLaunchIcon, color: 'purple' },
  ];

  useEffect(() => {
    // Initialize with sample goals
    const sampleGoals: EvolutionGoal[] = [
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
      {
        id: '3',
        title: 'Optimize Cellular Health',
        description: 'Improve mitochondrial function and cellular repair mechanisms',
        category: 'health',
        priority: 'medium',
        targetValue: 90,
        currentValue: 75,
        unit: '%',
        deadline: '2025-01-31',
        status: 'not_started',
        milestones: [
          { id: '3a', title: 'NAD+ supplementation', targetDate: '2024-10-01', completed: false },
          { id: '3b', title: 'Heat shock therapy', targetDate: '2024-11-01', completed: false },
          { id: '3c', title: 'Cellular analysis', targetDate: '2024-12-01', completed: false },
        ]
      }
    ];
    setGoals(sampleGoals);
  }, []);

  const handleCreateGoal = async (goalData: Partial<EvolutionGoal>) => {
    const newGoal: EvolutionGoal = {
      id: Date.now().toString(),
      title: goalData.title || '',
      description: goalData.description || '',
      category: goalData.category || 'health',
      priority: goalData.priority || 'medium',
      targetValue: goalData.targetValue || 0,
      currentValue: goalData.currentValue || 0,
      unit: goalData.unit || '',
      deadline: goalData.deadline || '',
      status: 'not_started',
      milestones: []
    };

    setGoals(prev => [...prev, newGoal]);
    setShowAddGoal(false);

    // Process through multi-agent system
    await processHumanFeedback({
      goals: [newGoal.title],
      satisfaction: 0.8,
      feedback: { new_goal: goalData }
    });
  };

  const handleUpdateGoal = async (goalId: string, updates: Partial<EvolutionGoal>) => {
    setGoals(prev => prev.map(goal => 
      goal.id === goalId ? { ...goal, ...updates } : goal
    ));

    await updateGoal(goalId, updates);
  };

  const getProgressPercentage = (goal: EvolutionGoal) => {
    if (goal.targetValue === goal.currentValue) return 100;
    return Math.min(100, Math.max(0, 
      ((goal.currentValue - (goal.currentValue * 0.8)) / (goal.targetValue - (goal.currentValue * 0.8))) * 100
    ));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Evolution Planning</h1>
              <p className="text-gray-600 mt-1">
                AI-powered human enhancement and longevity optimization
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Multi-Agent Status */}
              {agentStatus && (
                <div className="flex items-center text-sm">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                  <span className="text-gray-600">
                    AI Agents Active ({agentStatus.agent_count})
                  </span>
                </div>
              )}

              {/* Timeframe Selector */}
              <select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                {timeframes.map(tf => (
                  <option key={tf.id} value={tf.id}>{tf.name}</option>
                ))}
              </select>

              <button
                onClick={() => setShowAddGoal(true)}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <PlusIcon className="w-4 h-4 mr-2" />
                Add Goal
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
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {activeTab === 'overview' && (
              <OverviewTab 
                goals={goals}
                agentStatus={agentStatus}
                selectedTimeframe={selectedTimeframe}
              />
            )}
            
            {activeTab === 'goals' && (
              <GoalsTab 
                goals={goals}
                onUpdateGoal={handleUpdateGoal}
                onCreateGoal={handleCreateGoal}
                showAddGoal={showAddGoal}
                setShowAddGoal={setShowAddGoal}
                goalCategories={goalCategories}
              />
            )}
            
            {activeTab === 'progress' && (
              <ProgressTab 
                goals={goals}
                selectedTimeframe={selectedTimeframe}
              />
            )}
            
            {activeTab === 'recommendations' && (
              <RecommendationsTab 
                recommendations={recommendations}
                agentStatus={agentStatus}
                goals={goals}
                onExecuteStep={executeStep}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Add Goal Modal */}
      <AddGoalModal
        isOpen={showAddGoal}
        onClose={() => setShowAddGoal(false)}
        onSubmit={handleCreateGoal}
        categories={goalCategories}
      />
    </div>
  );
};

// Overview Tab Component
const OverviewTab: React.FC<{ 
  goals: EvolutionGoal[]; 
  agentStatus: any; 
  selectedTimeframe: string; 
}> = ({ goals, agentStatus, selectedTimeframe }) => {
  const activeGoals = goals.filter(g => g.status === 'in_progress').length;
  const completedGoals = goals.filter(g => g.status === 'completed').length;
  const totalProgress = goals.reduce((sum, goal) => {
    const progress = Math.min(100, Math.max(0, 
      ((goal.currentValue - (goal.currentValue * 0.8)) / (goal.targetValue - (goal.currentValue * 0.8))) * 100
    ));
    return sum + progress;
  }, 0) / goals.length;

  return (
    <div className="space-y-8">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Active Goals"
          value={activeGoals}
          icon={StarIcon}
          color="blue"
          change={+2}
        />
        <MetricCard
          title="Completed Goals"
          value={completedGoals}
          icon={CheckCircleIcon}
          color="green"
          change={+1}
        />
        <MetricCard
          title="Overall Progress"
          value={`${totalProgress.toFixed(0)}%`}
          icon={ArrowTrendingUpIcon}
          color="purple"
          change={+5.2}
        />
        <MetricCard
          title="AI Agents"
          value={agentStatus?.agent_count || 0}
          icon={BrainIcon}
          color="orange"
          change={0}
        />
      </div>

      {/* Progress Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Goal Progress Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Goal Progress</h3>
          <div className="h-64">
            <Bar
              data={{
                labels: goals.map(g => g.title.substring(0, 15) + '...'),
                datasets: [
                  {
                    label: 'Progress %',
                    data: goals.map(goal => {
                      const progress = Math.min(100, Math.max(0, 
                        ((goal.currentValue - (goal.currentValue * 0.8)) / (goal.targetValue - (goal.currentValue * 0.8))) * 100
                      ));
                      return progress;
                    }),
                    backgroundColor: goals.map(goal => {
                      const colors = {
                        health: 'rgba(239, 68, 68, 0.8)',
                        cognitive: 'rgba(59, 130, 246, 0.8)',
                        longevity: 'rgba(34, 197, 94, 0.8)',
                        performance: 'rgba(147, 51, 234, 0.8)',
                      };
                      return colors[goal.category];
                    }),
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { display: false },
                },
                scales: {
                  y: { beginAtZero: true, max: 100 },
                },
              }}
            />
          </div>
        </div>

        {/* Category Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Goal Categories</h3>
          <div className="h-64">
            <Radar
              data={{
                labels: ['Health', 'Cognitive', 'Longevity', 'Performance'],
                datasets: [
                  {
                    label: 'Current Progress',
                    data: [
                      goals.filter(g => g.category === 'health').reduce((sum, g) => sum + ((g.currentValue / g.targetValue) * 100), 0) / Math.max(1, goals.filter(g => g.category === 'health').length),
                      goals.filter(g => g.category === 'cognitive').reduce((sum, g) => sum + ((g.currentValue / g.targetValue) * 100), 0) / Math.max(1, goals.filter(g => g.category === 'cognitive').length),
                      goals.filter(g => g.category === 'longevity').reduce((sum, g) => sum + ((g.currentValue / g.targetValue) * 100), 0) / Math.max(1, goals.filter(g => g.category === 'longevity').length),
                      goals.filter(g => g.category === 'performance').reduce((sum, g) => sum + ((g.currentValue / g.targetValue) * 100), 0) / Math.max(1, goals.filter(g => g.category === 'performance').length),
                    ],
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    pointBackgroundColor: 'rgba(59, 130, 246, 1)',
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    beginAtZero: true,
                    max: 100,
                  },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* Recent Goals */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Active Goals</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {goals.filter(g => g.status === 'in_progress').slice(0, 3).map((goal) => (
              <GoalCard key={goal.id} goal={goal} compact={true} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Metric Card Component
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ComponentType<any>;
  color: string;
  change: number;
}> = ({ title, value, icon: Icon, color, change }) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    purple: 'bg-purple-50 text-purple-600',
    orange: 'bg-orange-50 text-orange-600',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
          <Icon className="w-6 h-6" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <div className="flex items-baseline">
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {change !== 0 && (
              <span className={`ml-2 text-sm ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {change > 0 ? '+' : ''}{change}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Placeholder components for other tabs
const GoalsTab: React.FC<any> = ({ goals, onUpdateGoal, goalCategories }) => (
  <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
    {goals.map((goal: EvolutionGoal) => (
      <GoalCard key={goal.id} goal={goal} onUpdate={onUpdateGoal} />
    ))}
  </div>
);

const ProgressTab: React.FC<any> = ({ goals, selectedTimeframe }) => (
  <ProgressTimeline goals={goals} timeframe={selectedTimeframe} />
);

const RecommendationsTab: React.FC<any> = ({ recommendations, agentStatus, goals, onExecuteStep }) => (
  <RecommendationPanel 
    recommendations={recommendations}
    agentStatus={agentStatus}
    goals={goals}
    onExecuteStep={onExecuteStep}
  />
);

// Add Goal Modal Component
const AddGoalModal: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (goal: Partial<EvolutionGoal>) => void;
  categories: any[];
}> = ({ isOpen, onClose, onSubmit, categories }) => {
  const [formData, setFormData] = useState<Partial<EvolutionGoal>>({
    category: 'health',
    priority: 'medium',
  });

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
    setFormData({ category: 'health', priority: 'medium' });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Add New Goal</h3>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
            <input
              type="text"
              value={formData.title || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              value={formData.description || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
              <select
                value={formData.category}
                onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                {categories.map(cat => (
                  <option key={cat.id} value={cat.id}>{cat.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
              <select
                value={formData.priority}
                onChange={(e) => setFormData(prev => ({ ...prev, priority: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Create Goal
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
