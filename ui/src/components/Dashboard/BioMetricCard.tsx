import React from 'react';
import { motion } from 'framer-motion';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';

interface BioMetric {
  name: string;
  value: number;
  unit: string;
  change: number;
  icon: React.ComponentType<any>;
  color: string;
}

interface BioMetricCardProps {
  metric: BioMetric;
}

export const BioMetricCard: React.FC<BioMetricCardProps> = ({ metric }) => {
  const { name, value, unit, change, icon: Icon, color } = metric;

  const colorClasses = {
    blue: {
      bg: 'bg-blue-50',
      icon: 'text-blue-600',
      accent: 'bg-blue-500',
    },
    green: {
      bg: 'bg-green-50',
      icon: 'text-green-600',
      accent: 'bg-green-500',
    },
    purple: {
      bg: 'bg-purple-50',
      icon: 'text-purple-600',
      accent: 'bg-purple-500',
    },
    orange: {
      bg: 'bg-orange-50',
      icon: 'text-orange-600',
      accent: 'bg-orange-500',
    },
    indigo: {
      bg: 'bg-indigo-50',
      icon: 'text-indigo-600',
      accent: 'bg-indigo-500',
    },
    red: {
      bg: 'bg-red-50',
      icon: 'text-red-600',
      accent: 'bg-red-500',
    },
  };

  const colors = colorClasses[color as keyof typeof colorClasses] || colorClasses.blue;

  const isPositiveChange = change > 0;
  const changeColor = isPositiveChange ? 'text-green-600' : 'text-red-600';
  const ChangeIcon = isPositiveChange ? ArrowUpIcon : ArrowDownIcon;

  return (
    <motion.div
      className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow duration-200"
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <div className={`p-3 rounded-lg ${colors.bg}`}>
            <Icon className={`w-6 h-6 ${colors.icon}`} />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">{name}</p>
            <div className="flex items-baseline">
              <p className="text-2xl font-bold text-gray-900">
                {typeof value === 'number' ? value.toFixed(1) : value}
              </p>
              <p className="ml-1 text-sm text-gray-500">{unit}</p>
            </div>
          </div>
        </div>
        
        {/* Change Indicator */}
        <div className="flex items-center">
          <div className={`flex items-center ${changeColor}`}>
            <ChangeIcon className="w-4 h-4 mr-1" />
            <span className="text-sm font-medium">
              {Math.abs(change).toFixed(1)}
            </span>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-4">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Progress</span>
          <span>{value}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            className={`h-2 rounded-full ${colors.accent}`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, Math.max(0, value))}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Trend Sparkline (simplified) */}
      <div className="mt-3 flex items-center justify-between">
        <div className="flex space-x-1">
          {[...Array(7)].map((_, i) => (
            <div
              key={i}
              className={`w-1 rounded-full ${colors.accent}`}
              style={{
                height: `${Math.random() * 16 + 4}px`,
                opacity: 0.3 + (i * 0.1),
              }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-400">7 days</span>
      </div>
    </motion.div>
  );
};
