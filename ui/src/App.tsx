import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Provider } from 'react-redux';
import { store } from './store/store';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';

// Components
import Layout from './components/Layout/Layout';
import ProtectedRoute from './components/Auth/ProtectedRoute';

// Pages
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import Dashboard from './pages/Dashboard/Dashboard';
import BioTwin from './pages/BioTwin/BioTwin';
import Evolution from './pages/Evolution/Evolution';
import Insights from './pages/Insights/Insights';
import Settings from './pages/Settings/Settings';

// New Advanced Components
import { BioTwinDashboard } from './components/Dashboard/BioTwinDashboard';
import { ProteinAnalysisInterface } from './components/ProteinAnalysis/ProteinAnalysisInterface';
import { EvolutionPlanningInterface } from './components/Evolution/EvolutionPlanningInterface';
import { RealTimeDataDashboard } from './components/RealTime/RealTimeDataDashboard';

// Styles
import './App.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <AuthProvider>
            <Router>
              <div className="App min-h-screen bg-gray-50 dark:bg-gray-900">
                <Routes>
                  {/* Public routes */}
                  <Route path="/login" element={<Login />} />
                  <Route path="/register" element={<Register />} />
                  
                  {/* Protected routes */}
                  <Route path="/" element={
                    <ProtectedRoute>
                      <Layout />
                    </ProtectedRoute>
                  }>
                    <Route index element={<Navigate to="/dashboard" replace />} />
                    <Route path="dashboard" element={<Dashboard />} />
                    <Route path="bio-twin" element={<BioTwinDashboard userId="demo_user_001" />} />
                    <Route path="protein-analysis" element={<ProteinAnalysisInterface userId="demo_user_001" />} />
                    <Route path="evolution" element={<EvolutionPlanningInterface userId="demo_user_001" />} />
                    <Route path="real-time" element={<RealTimeDataDashboard userId="demo_user_001" />} />
                    <Route path="insights" element={<Insights />} />
                    <Route path="settings" element={<Settings />} />
                  </Route>
                  
                  {/* Catch all route */}
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </div>
            </Router>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  );
}

export default App;
