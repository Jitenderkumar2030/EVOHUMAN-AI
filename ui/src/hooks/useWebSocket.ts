import { useState, useEffect, useRef, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastMessage: any;
  connectionId: string | null;
}

export const useWebSocket = (
  endpoint: string,
  options: UseWebSocketOptions = {}
) => {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options;

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastMessage: null,
    connectionId: null,
  });

  const socketRef = useRef<Socket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }));

    try {
      const socket = io(`ws://localhost:8000${endpoint}`, {
        transports: ['websocket'],
        upgrade: true,
        rememberUpgrade: true,
        timeout: 10000,
      });

      socket.on('connect', () => {
        console.log('WebSocket connected:', endpoint);
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
          connectionId: socket.id,
        }));
        reconnectCountRef.current = 0;
      });

      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          connectionId: null,
        }));

        // Attempt reconnection if not manually disconnected
        if (reason !== 'io client disconnect' && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      });

      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          error: error.message,
        }));
      });

      // Listen for real-time data updates
      socket.on('bio_metrics_update', (data) => {
        setState(prev => ({ ...prev, lastMessage: { type: 'bio_metrics', data } }));
      });

      socket.on('cellular_simulation_update', (data) => {
        setState(prev => ({ ...prev, lastMessage: { type: 'cellular_simulation', data } }));
      });

      socket.on('agent_system_update', (data) => {
        setState(prev => ({ ...prev, lastMessage: { type: 'agent_system', data } }));
      });

      socket.on('protein_analysis_update', (data) => {
        setState(prev => ({ ...prev, lastMessage: { type: 'protein_analysis', data } }));
      });

      socket.on('evolution_progress_update', (data) => {
        setState(prev => ({ ...prev, lastMessage: { type: 'evolution_progress', data } }));
      });

      socketRef.current = socket;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        isConnecting: false,
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }
  }, [endpoint, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
      connectionId: null,
    }));
  }, []);

  const sendMessage = useCallback((event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
      return true;
    }
    return false;
  }, []);

  const subscribe = useCallback((event: string, callback: (data: any) => void) => {
    if (socketRef.current) {
      socketRef.current.on(event, callback);
      return () => {
        socketRef.current?.off(event, callback);
      };
    }
    return () => {};
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    subscribe,
  };
};
