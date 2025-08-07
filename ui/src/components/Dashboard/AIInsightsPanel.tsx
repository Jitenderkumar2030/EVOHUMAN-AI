import React from 'react';
import { motion } from 'framer-motion';
import { 
  LightBulbIcon, 
  ExclamationTriangleIcon, 
  CheckCircleIcon,
  InformationCircleIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

interface AIInsight {
  type: string;
  message: string;
  confidence: number;
  timestamp: string;
}

interface AIInsightsPanelProps {
  insights?: AIInsight[];
}

export const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({ insights = [] }) => {
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'optimization':
        return LightBulbIcon;
      case 'alert':
        return ExclamationTriangleIcon;
      case 'recommendation':
        return SparklesIcon;
      case 'success':
        return CheckCircleIcon;
      default:
        return InformationCircleIcon;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'optimization':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'alert':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'recommendation':
        return 'text-purple-600 bg-purple-50 border-purple-200';
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <SparklesIcon className="w-5 h-5 mr-2 text-purple-500" />
          AI Insights
        </h3>
        <span className="text-sm text-gray-500">
          {insights.length} insights
        </span>
      </div>

      <div className="space-y-4">
        {insights.length === 0 ? (
          <div className="text-center py-8">
            <SparklesIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No AI insights available yet.</p>
            <p className="text-sm text-gray-400 mt-1">
              Insights will appear as your bio-twin analyzes your data.
            </p>
          </div>
        ) : (
          insights.map((insight, index) => {
            const Icon = getInsightIcon(insight.type);
            const colorClasses = getInsightColor(insight.type);
            
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border ${colorClasses}`}
              >
                <div className="flex items-start">
                  <Icon className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-sm font-medium mb-1 capitalize">
                      {insight.type} Insight
                    </p>
                    <p className="text-sm text-gray-700 mb-2">
                      {insight.message}
                    </p>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">
                        {formatTimestamp(insight.timestamp)}
                      </span>
                      <div className="flex items-center">
                        <span className="text-gray-500 mr-1">Confidence:</span>
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-1.5 mr-2">
                            <div 
                              className="bg-current h-1.5 rounded-full"
                              style={{ width: `${insight.confidence * 100}%` }}
                            />
                          </div>
                          <span className="font-medium">
                            {(insight.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
};
