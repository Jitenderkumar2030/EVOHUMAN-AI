import React from 'react';
import { motion } from 'framer-motion';
import { 
  CheckCircleIcon, 
  ClockIcon, 
  PlayIcon,
  PauseIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';

interface EvolutionTimelineProps {
  data?: any;
  userId: string;
}

interface TimelineEvent {
  id: string;
  title: string;
  description: string;
  date: string;
  status: 'completed' | 'in_progress' | 'upcoming' | 'paused';
  category: 'health' | 'cognitive' | 'longevity' | 'performance';
  impact: number;
}

export const EvolutionTimeline: React.FC<EvolutionTimelineProps> = ({ data, userId }) => {
  // Mock timeline data
  const timelineEvents: TimelineEvent[] = [
    {
      id: '1',
      title: 'Sleep Optimization Protocol',
      description: 'Implemented circadian rhythm optimization with blue light filtering and temperature control.',
      date: '2024-08-01',
      status: 'completed',
      category: 'health',
      impact: 0.23,
    },
    {
      id: '2',
      title: 'Cognitive Enhancement Training',
      description: 'Started dual n-back training and working memory exercises.',
      date: '2024-08-05',
      status: 'completed',
      category: 'cognitive',
      impact: 0.18,
    },
    {
      id: '3',
      title: 'Intermittent Fasting Protocol',
      description: 'Began 16:8 time-restricted eating window for metabolic optimization.',
      date: '2024-08-10',
      status: 'in_progress',
      category: 'longevity',
      impact: 0.31,
    },
    {
      id: '4',
      title: 'Advanced Biomarker Testing',
      description: 'Comprehensive blood panel including inflammatory markers and hormone levels.',
      date: '2024-08-15',
      status: 'in_progress',
      category: 'health',
      impact: 0.15,
    },
    {
      id: '5',
      title: 'Meditation Practice Expansion',
      description: 'Extended daily meditation to 25 minutes with focused attention techniques.',
      date: '2024-08-20',
      status: 'upcoming',
      category: 'cognitive',
      impact: 0.22,
    },
    {
      id: '6',
      title: 'NAD+ Supplementation',
      description: 'Begin NAD+ precursor supplementation for cellular energy optimization.',
      date: '2024-08-25',
      status: 'upcoming',
      category: 'longevity',
      impact: 0.28,
    },
  ];

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

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'health':
        return 'bg-red-50 text-red-700 border-red-200';
      case 'cognitive':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'longevity':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'performance':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <ArrowTrendingUpIcon className="w-5 h-5 mr-2 text-blue-500" />
          Evolution Timeline
        </h3>
        <div className="text-sm text-gray-500">
          {timelineEvents.filter(e => e.status === 'completed').length} of {timelineEvents.length} completed
        </div>
      </div>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200"></div>

        <div className="space-y-6">
          {timelineEvents.map((event, index) => {
            const StatusIcon = getStatusIcon(event.status);
            const statusColor = getStatusColor(event.status);
            const categoryColor = getCategoryColor(event.category);

            return (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative flex items-start"
              >
                {/* Timeline dot */}
                <div className={`relative z-10 flex items-center justify-center w-8 h-8 rounded-full ${statusColor}`}>
                  <StatusIcon className="w-4 h-4" />
                </div>

                {/* Content */}
                <div className="ml-4 flex-1">
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-gray-900 mb-1">
                          {event.title}
                        </h4>
                        <p className="text-sm text-gray-600 mb-2">
                          {event.description}
                        </p>
                      </div>
                      <div className="ml-4 text-right">
                        <div className="text-xs text-gray-500 mb-1">
                          {formatDate(event.date)}
                        </div>
                        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${categoryColor}`}>
                          {event.category}
                        </div>
                      </div>
                    </div>

                    {/* Impact indicator */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center text-xs text-gray-500">
                        <span className="mr-2">Expected Impact:</span>
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-1.5 mr-2">
                            <div 
                              className="bg-blue-500 h-1.5 rounded-full"
                              style={{ width: `${event.impact * 100}%` }}
                            />
                          </div>
                          <span className="font-medium">
                            {(event.impact * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      {/* Status badge */}
                      <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusColor}`}>
                        <StatusIcon className="w-3 h-3 mr-1" />
                        {event.status.replace('_', ' ')}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Summary stats */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {timelineEvents.filter(e => e.status === 'completed').length}
            </div>
            <div className="text-xs text-gray-500">Completed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {timelineEvents.filter(e => e.status === 'in_progress').length}
            </div>
            <div className="text-xs text-gray-500">In Progress</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-600">
              {timelineEvents.filter(e => e.status === 'upcoming').length}
            </div>
            <div className="text-xs text-gray-500">Upcoming</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {(timelineEvents.reduce((sum, e) => sum + e.impact, 0) * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500">Total Impact</div>
          </div>
        </div>
      </div>
    </div>
  );
};
