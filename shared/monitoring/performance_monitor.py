"""
Performance Monitoring System
Comprehensive APM and metrics collection for EvoHuman.AI services
"""
import time
import psutil
import asyncio
import structlog
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps
import json
import aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import numpy as np


logger = structlog.get_logger("performance-monitor")


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = "ms"


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    timestamp: datetime


@dataclass
class ServiceMetrics:
    """Service-specific performance metrics"""
    service_name: str
    request_count: int
    error_count: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    active_connections: int
    timestamp: datetime


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, service_name: str, redis_url: str = "redis://localhost:6379"):
        self.service_name = service_name
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.system_cpu = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        self.system_memory = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Internal metrics storage
        self.metrics_buffer: List[PerformanceMetric] = []
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.request_counts: Dict[str, int] = {}
        
        # Configuration
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds
        self.alert_thresholds = {
            'response_time_p95': 2000,  # ms
            'error_rate': 0.05,  # 5%
            'cpu_usage': 80,  # %
            'memory_usage': 85,  # %
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        
        logger.info("Performance monitor initialized", service=service_name)
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            self._flush_task = asyncio.create_task(self._metrics_flush_loop())
            
            logger.info("Performance monitoring started")
        except Exception as e:
            logger.error("Failed to initialize performance monitoring", error=str(e))
    
    async def cleanup(self):
        """Cleanup monitoring resources"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._flush_task:
            self._flush_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Performance monitoring stopped")
    
    @asynccontextmanager
    async def track_request(self, method: str, endpoint: str):
        """Context manager to track HTTP request performance"""
        start_time = time.time()
        status_code = "200"
        
        try:
            yield
        except Exception as e:
            status_code = "500"
            self.record_error(f"{method} {endpoint}", str(e))
            raise
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            # Record metrics
            self.record_request(method, endpoint, duration, status_code)
    
    def record_request(self, method: str, endpoint: str, duration_ms: float, status_code: str):
        """Record HTTP request metrics"""
        # Prometheus metrics
        self.request_counter.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration_ms / 1000)
        
        # Internal metrics
        self.response_times.append(duration_ms)
        if len(self.response_times) > self.buffer_size:
            self.response_times = self.response_times[-self.buffer_size:]
        
        key = f"{method}:{endpoint}"
        self.request_counts[key] = self.request_counts.get(key, 0) + 1
        
        # Create metric object
        metric = PerformanceMetric(
            name="http_request_duration",
            value=duration_ms,
            timestamp=datetime.utcnow(),
            labels={
                "method": method,
                "endpoint": endpoint,
                "status": status_code,
                "service": self.service_name
            }
        )
        
        self.metrics_buffer.append(metric)
        
        # Check for performance alerts
        asyncio.create_task(self._check_performance_alerts())
    
    def record_error(self, operation: str, error_message: str):
        """Record error occurrence"""
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        
        metric = PerformanceMetric(
            name="error_count",
            value=1,
            timestamp=datetime.utcnow(),
            labels={
                "operation": operation,
                "service": self.service_name,
                "error": error_message[:100]  # Truncate long error messages
            }
        )
        
        self.metrics_buffer.append(metric)
        
        logger.error("Operation error recorded", 
                    operation=operation, 
                    error=error_message)
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record custom application metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels={**(labels or {}), "service": self.service_name}
        )
        
        self.metrics_buffer.append(metric)
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io,
            timestamp=datetime.utcnow()
        )
    
    async def get_service_metrics(self) -> ServiceMetrics:
        """Get current service performance metrics"""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        # Calculate response time percentiles
        if self.response_times:
            avg_response_time = np.mean(self.response_times)
            p95_response_time = np.percentile(self.response_times, 95)
            p99_response_time = np.percentile(self.response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        return ServiceMetrics(
            service_name=self.service_name,
            request_count=total_requests,
            error_count=total_errors,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            active_connections=0,  # Would be set by web server
            timestamp=datetime.utcnow()
        )
    
    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def _system_monitoring_loop(self):
        """Background task to collect system metrics"""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self.get_system_metrics()
                
                # Update Prometheus gauges
                self.system_cpu.set(system_metrics.cpu_percent)
                self.system_memory.set(system_metrics.memory_percent)
                
                # Store in Redis if available
                if self.redis_client:
                    await self._store_system_metrics(system_metrics)
                
                # Check for system alerts
                await self._check_system_alerts(system_metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("System monitoring error", error=str(e))
                await asyncio.sleep(30)
    
    async def _metrics_flush_loop(self):
        """Background task to flush metrics to storage"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics flush error", error=str(e))
    
    async def _flush_metrics(self):
        """Flush buffered metrics to Redis"""
        if not self.redis_client or not self.metrics_buffer:
            return
        
        try:
            # Prepare metrics for storage
            metrics_data = []
            for metric in self.metrics_buffer:
                metrics_data.append({
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "labels": metric.labels,
                    "unit": metric.unit
                })
            
            # Store in Redis with expiration
            key = f"metrics:{self.service_name}:{int(time.time())}"
            await self.redis_client.setex(
                key, 
                86400,  # 24 hours
                json.dumps(metrics_data)
            )
            
            # Clear buffer
            self.metrics_buffer.clear()
            
            logger.debug("Metrics flushed to storage", count=len(metrics_data))
            
        except Exception as e:
            logger.error("Failed to flush metrics", error=str(e))
    
    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in Redis"""
        try:
            key = f"system_metrics:{self.service_name}"
            data = {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_mb": metrics.memory_used_mb,
                "disk_usage_percent": metrics.disk_usage_percent,
                "network_io": metrics.network_io,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            await self.redis_client.setex(key, 300, json.dumps(data))  # 5 minutes
            
        except Exception as e:
            logger.error("Failed to store system metrics", error=str(e))
    
    async def _check_performance_alerts(self):
        """Check for performance threshold violations"""
        try:
            if not self.response_times:
                return
            
            # Check response time P95
            p95_time = np.percentile(self.response_times, 95)
            if p95_time > self.alert_thresholds['response_time_p95']:
                await self._send_alert(
                    "high_response_time",
                    f"P95 response time is {p95_time:.2f}ms (threshold: {self.alert_thresholds['response_time_p95']}ms)"
                )
            
            # Check error rate
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            
            if total_requests > 0:
                error_rate = total_errors / total_requests
                if error_rate > self.alert_thresholds['error_rate']:
                    await self._send_alert(
                        "high_error_rate",
                        f"Error rate is {error_rate:.2%} (threshold: {self.alert_thresholds['error_rate']:.2%})"
                    )
        
        except Exception as e:
            logger.error("Performance alert check failed", error=str(e))
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system resource threshold violations"""
        try:
            # Check CPU usage
            if metrics.cpu_percent > self.alert_thresholds['cpu_usage']:
                await self._send_alert(
                    "high_cpu_usage",
                    f"CPU usage is {metrics.cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_usage']}%)"
                )
            
            # Check memory usage
            if metrics.memory_percent > self.alert_thresholds['memory_usage']:
                await self._send_alert(
                    "high_memory_usage",
                    f"Memory usage is {metrics.memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_usage']}%)"
                )
        
        except Exception as e:
            logger.error("System alert check failed", error=str(e))
    
    async def _send_alert(self, alert_type: str, message: str):
        """Send performance alert"""
        alert_data = {
            "service": self.service_name,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "warning"
        }
        
        # Store alert in Redis
        if self.redis_client:
            try:
                key = f"alerts:{self.service_name}:{alert_type}:{int(time.time())}"
                await self.redis_client.setex(key, 3600, json.dumps(alert_data))  # 1 hour
            except Exception as e:
                logger.error("Failed to store alert", error=str(e))
        
        logger.warning("Performance alert", **alert_data)


def monitor_performance(monitor: PerformanceMonitor):
    """Decorator to monitor function performance"""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    monitor.record_custom_metric(
                        f"function_duration_{func.__name__}",
                        duration,
                        {"function": func.__name__}
                    )
                    return result
                except Exception as e:
                    monitor.record_error(func.__name__, str(e))
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    monitor.record_custom_metric(
                        f"function_duration_{func.__name__}",
                        duration,
                        {"function": func.__name__}
                    )
                    return result
                except Exception as e:
                    monitor.record_error(func.__name__, str(e))
                    raise
            return sync_wrapper
    return decorator
