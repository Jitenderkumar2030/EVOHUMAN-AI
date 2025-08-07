import React from 'react';
import { motion } from 'framer-motion';
import { 
  StarIcon, 
  ClockIcon, 
  CheckCircleIcon,
  PlayIcon,
  PauseIcon,
  ExclamationTriangleIcon,
  HeartIcon,
  BrainIcon,
  BeakerIcon,
  RocketLaunchIcon
} from '@heroicons/react/24/outline';

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

interface GoalCardProps {
  goal: EvolutionGoal;
  compact?: boolean;
  onUpdate?: (goalId: string, updates: Partial<EvolutionGoal>) => void;
}

export const GoalCard: React.FC<GoalCardProps> = ({ goal, compact = false, onUpdate }) => {
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'health':
        return HeartIcon;
      case 'cognitive':
        return BrainIcon;
      case 'longevity':
        return ClockIcon;
      case 'performance':
        return RocketLaunchIcon;
      default:
        return BeakerIcon;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'health':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'cognitive':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'longevity':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'performance':
        return 'text-purple-600 bg-purple-50 border-purple-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'text-red-700 bg-red-100';
      case 'medium':
        return 'text-yellow-700 bg-yellow-100';
      case 'low':
        return 'text-green-700 bg-green-100';
      default:
        return 'text-gray-700 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return CheckCircleIcon;
      case 'in_progress':
        return PlayIcon;
      case 'paused':
        return PauseIcon;
      default:
        return ClockIcon;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'in_progress':
        return 'text-blue-600 bg-blue-100';
      case 'paused':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getProgressPercentage = () => {
    if (goal.targetValue === goal.currentValue) return 100;
    
    // For goals where we want to decrease the value (like biological age)
    if (goal.targetValue < goal.currentValue) {
      const totalReduction = goal.currentValue - goal.targetValue;
      const currentReduction = goal.currentValue - goal.currentValue; // This would be dynamic in real app
      return Math.min(100, Math.max(0, (currentReduction / totalReduction) * 100));
    }
    
    // For goals where we want to increase the value
    return Math.min(100, Math.max(0, (goal.currentValue / goal.targetValue) * 100));
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
  };

  const CategoryIcon = getCategoryIcon(goal.category);
  const StatusIcon = getStatusIcon(goal.status);
  const progressPercentage = getProgressPercentage();

  if (compact) {
    return (
      <motion.div
        className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
        whileHover={{ scale: 1.02 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center flex-1">
            <div className={`p-2 rounded-lg ${getCategoryColor(goal.category)}`}>
              <CategoryIcon className="w-4 h-4" />
            </div>
            <div className="ml-3 flex-1">
              <h4 className="text-sm font-semibold text-gray-900">{goal.title}</h4>
              <div className="flex items-center mt-1">
                <div className="w-16 bg-gray-200 rounded-full h-1.5 mr-2">
                  <div 
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${progressPercentage}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500">
                  {progressPercentage.toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(goal.status)}`}>
            <StatusIcon className="w-3 h-3 mr-1" />
            {goal.status.replace('_', ' ')}
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="bg-white rounded-lg shadow border border-gray-200 p-6 hover:shadow-lg transition-shadow"
      whileHover={{ scale: 1.02 }}
      layout
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center">
          <div className={`p-3 rounded-lg ${getCategoryColor(goal.category)}`}>
            <CategoryIcon className="w-6 h-6" />
          </div>
          <div className="ml-4">
            <h3 className="text-lg font-semibold text-gray-900">{goal.title}</h3>
            <p className="text-sm text-gray-600 mt-1">{goal.description}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(goal.priority)}`}>
            <StarIcon className="w-3 h-3 mr-1" />
            {goal.priority}
          </div>
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(goal.status)}`}>
            <StatusIcon className="w-3 h-3 mr-1" />
            {goal.status.replace('_', ' ')}
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Progress</span>
          <span className="text-sm text-gray-500">
            {goal.currentValue} / {goal.targetValue} {goal.unit}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            className="bg-blue-500 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progressPercentage}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>{progressPercentage.toFixed(0)}% complete</span>
          <span>Due: {formatDate(goal.deadline)}</span>
        </div>
      </div>

      {/* Milestones */}
      {goal.milestones && goal.milestones.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Milestones</h4>
          <div className="space-y-2">
            {goal.milestones.slice(0, 3).map((milestone) => (
              <div key={milestone.id} className="flex items-center text-sm">
                <div className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center ${
                  milestone.completed 
                    ? 'bg-green-100 text-green-600' 
                    : 'bg-gray-100 text-gray-400'
                }`}>
                  {milestone.completed && <CheckCircleIcon className="w-3 h-3" />}
                </div>
                <span className={milestone.completed ? 'text-gray-900 line-through' : 'text-gray-700'}>
                  {milestone.title}
                </span>
                <span className="ml-auto text-xs text-gray-500">
                  {formatDate(milestone.targetDate)}
                </span>
              </div>
            ))}
            {goal.milestones.length > 3 && (
              <div className="text-xs text-gray-500 ml-7">
                +{goal.milestones.length - 3} more milestones
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-200">
        <div className="flex items-center space-x-2">
          {goal.status === 'not_started' && (
            <button
              onClick={() => onUpdate?.(goal.id, { status: 'in_progress' })}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              Start Goal
            </button>
          )}
          {goal.status === 'in_progress' && (
            <button
              onClick={() => onUpdate?.(goal.id, { status: 'paused' })}
              className="text-sm text-yellow-600 hover:text-yellow-800 font-medium"
            >
              Pause
            </button>
          )}
          {goal.status === 'paused' && (
            <button
              onClick={() => onUpdate?.(goal.id, { status: 'in_progress' })}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              Resume
            </button>
          )}
        </div>
        
        <div className="text-xs text-gray-500">
          Category: {goal.category}
        </div>
      </div>
    </motion.div>
  );
};
