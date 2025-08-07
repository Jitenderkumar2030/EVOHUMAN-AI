"""
Performance Benchmark Suite for EvoHuman.AI
Comprehensive performance testing and benchmarking
"""
import asyncio
import aiohttp
import time
import statistics
import json
import psutil
import structlog
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


logger = structlog.get_logger("performance-benchmark")


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_name: str
    duration_ms: float
    success: bool
    response_size: int = 0
    status_code: int = 0
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BenchmarkSummary:
    """Benchmark summary statistics"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    total_duration: float
    error_rate: float


class PerformanceBenchmark:
    """Performance benchmark suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[BenchmarkResult] = []
        
        # Benchmark configuration
        self.concurrent_users = [1, 5, 10, 25, 50, 100]
        self.test_duration = 60  # seconds
        self.ramp_up_time = 10  # seconds
        
        # Test data
        self.test_protein_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        self.test_user_id = "benchmark_user"
    
    async def setup(self):
        """Setup benchmark environment"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(limit=200, limit_per_host=50)
        )
        
        # Setup test data
        await self._setup_test_data()
        
        logger.info("Performance benchmark setup complete")
    
    async def teardown(self):
        """Cleanup benchmark environment"""
        if self.session:
            await self.session.close()
        
        # Generate reports
        await self._generate_reports()
        
        logger.info("Performance benchmark teardown complete")
    
    async def _setup_test_data(self):
        """Setup test data for benchmarks"""
        try:
            # Create test user
            await self.session.post(
                f"{self.base_url.replace('8000', '8001')}/users",
                json={
                    "id": self.test_user_id,
                    "name": "Benchmark User",
                    "email": "benchmark@test.com",
                    "biological_age": 30,
                    "chronological_age": 35,
                }
            )
            
            # Initialize bio-twin
            await self.session.post(
                f"{self.base_url.replace('8000', '8005')}/bio-twin/initialize",
                json={
                    "user_id": self.test_user_id,
                    "initial_metrics": {
                        "health_score": 85,
                        "energy_level": 78,
                        "cognitive_index": 92,
                    }
                }
            )
            
            logger.info("Benchmark test data setup complete")
            
        except Exception as e:
            logger.warning(f"Test data setup failed: {e}, using mock data")
    
    async def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        logger.info("Starting comprehensive performance benchmarks")
        
        # System resource benchmarks
        await self.benchmark_system_resources()
        
        # Service-specific benchmarks
        await self.benchmark_bio_twin_service()
        await self.benchmark_protein_analysis()
        await self.benchmark_cellular_simulation()
        await self.benchmark_multi_agent_system()
        
        # Load testing
        await self.benchmark_concurrent_load()
        
        # Stress testing
        await self.benchmark_stress_limits()
        
        logger.info("All performance benchmarks completed")
    
    async def benchmark_system_resources(self):
        """Benchmark system resource usage"""
        logger.info("Running system resource benchmarks")
        
        # Monitor system resources during idle state
        idle_metrics = await self._collect_system_metrics(duration=30)
        
        # Monitor system resources during load
        load_task = asyncio.create_task(self._generate_load())
        load_metrics = await self._collect_system_metrics(duration=60)
        await load_task
        
        # Calculate resource utilization
        cpu_increase = load_metrics['cpu_avg'] - idle_metrics['cpu_avg']
        memory_increase = load_metrics['memory_avg'] - idle_metrics['memory_avg']
        
        logger.info(
            "System resource benchmark results",
            idle_cpu=idle_metrics['cpu_avg'],
            load_cpu=load_metrics['cpu_avg'],
            cpu_increase=cpu_increase,
            idle_memory=idle_metrics['memory_avg'],
            load_memory=load_metrics['memory_avg'],
            memory_increase=memory_increase
        )
    
    async def benchmark_bio_twin_service(self):
        """Benchmark bio-twin service performance"""
        logger.info("Running bio-twin service benchmarks")
        
        test_cases = [
            {
                "name": "bio_twin_get",
                "method": "GET",
                "url": f"{self.base_url.replace('8000', '8005')}/bio-twin/{self.test_user_id}",
                "expected_response_time": 100  # ms
            },
            {
                "name": "bio_twin_update",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8005')}/bio-twin/{self.test_user_id}/update",
                "json": {"health_score": 88, "energy_level": 82},
                "expected_response_time": 200  # ms
            },
            {
                "name": "cognitive_assessment",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8001')}/cognitive/assess",
                "json": {"user_id": self.test_user_id, "assessment_type": "comprehensive"},
                "expected_response_time": 500  # ms
            }
        ]
        
        for test_case in test_cases:
            await self._run_single_benchmark(test_case, iterations=100)
    
    async def benchmark_protein_analysis(self):
        """Benchmark protein analysis performance"""
        logger.info("Running protein analysis benchmarks")
        
        test_cases = [
            {
                "name": "protein_analysis_small",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8003')}/analyze_protein",
                "json": {
                    "sequence": self.test_protein_sequence[:100],
                    "analysis_type": "structure_prediction"
                },
                "expected_response_time": 2000  # ms
            },
            {
                "name": "protein_analysis_medium",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8003')}/analyze_protein",
                "json": {
                    "sequence": self.test_protein_sequence,
                    "analysis_type": "structure_prediction"
                },
                "expected_response_time": 5000  # ms
            },
            {
                "name": "protein_analysis_with_mutations",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8003')}/analyze_protein",
                "json": {
                    "sequence": self.test_protein_sequence,
                    "analysis_type": "structure_prediction",
                    "include_mutations": True,
                    "include_evolution": True
                },
                "expected_response_time": 8000  # ms
            }
        ]
        
        for test_case in test_cases:
            await self._run_single_benchmark(test_case, iterations=20)
    
    async def benchmark_cellular_simulation(self):
        """Benchmark cellular simulation performance"""
        logger.info("Running cellular simulation benchmarks")
        
        test_cases = [
            {
                "name": "cellular_automata_small",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8002')}/simulate/cellular_automata",
                "json": {
                    "tissue_type": "neural",
                    "initial_cell_count": 100,
                    "simulation_steps": 10
                },
                "expected_response_time": 1000  # ms
            },
            {
                "name": "cellular_automata_medium",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8002')}/simulate/cellular_automata",
                "json": {
                    "tissue_type": "neural",
                    "initial_cell_count": 1000,
                    "simulation_steps": 50
                },
                "expected_response_time": 5000  # ms
            },
            {
                "name": "wound_healing_simulation",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8002')}/simulate/wound_healing",
                "json": {
                    "wound_type": "acute",
                    "wound_size": [5, 5, 2],
                    "simulation_days": 7,
                    "patient_age": 35
                },
                "expected_response_time": 3000  # ms
            }
        ]
        
        for test_case in test_cases:
            await self._run_single_benchmark(test_case, iterations=10)
    
    async def benchmark_multi_agent_system(self):
        """Benchmark multi-agent system performance"""
        logger.info("Running multi-agent system benchmarks")
        
        test_cases = [
            {
                "name": "agent_status",
                "method": "GET",
                "url": f"{self.base_url.replace('8000', '8004')}/multi_agent/status/{self.test_user_id}",
                "expected_response_time": 100  # ms
            },
            {
                "name": "agent_step",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8004')}/multi_agent/step",
                "json": {
                    "user_id": self.test_user_id,
                    "human_input": {"satisfaction": 0.8, "goals": ["improve_health"]}
                },
                "expected_response_time": 500  # ms
            },
            {
                "name": "human_feedback",
                "method": "POST",
                "url": f"{self.base_url.replace('8000', '8004')}/multi_agent/human_feedback",
                "json": {
                    "user_id": self.test_user_id,
                    "feedback_data": {"satisfaction": 0.7, "goals": ["better_sleep"]}
                },
                "expected_response_time": 300  # ms
            }
        ]
        
        for test_case in test_cases:
            await self._run_single_benchmark(test_case, iterations=50)
    
    async def benchmark_concurrent_load(self):
        """Benchmark system under concurrent load"""
        logger.info("Running concurrent load benchmarks")
        
        for user_count in self.concurrent_users:
            logger.info(f"Testing with {user_count} concurrent users")
            
            # Create concurrent tasks
            tasks = []
            for _ in range(user_count):
                task = asyncio.create_task(self._simulate_user_session())
                tasks.append(task)
            
            # Run concurrent load test
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_sessions = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful_sessions) / len(results)
            
            logger.info(
                f"Concurrent load test results",
                users=user_count,
                success_rate=success_rate,
                duration=end_time - start_time,
                successful_sessions=len(successful_sessions)
            )
    
    async def benchmark_stress_limits(self):
        """Benchmark system stress limits"""
        logger.info("Running stress limit benchmarks")
        
        # Gradually increase load until failure
        current_load = 10
        max_successful_load = 0
        
        while current_load <= 500:
            logger.info(f"Testing stress limit with {current_load} requests/second")
            
            # Generate load for 30 seconds
            success_rate = await self._generate_sustained_load(
                requests_per_second=current_load,
                duration=30
            )
            
            if success_rate >= 0.95:  # 95% success rate threshold
                max_successful_load = current_load
                current_load = int(current_load * 1.5)
            else:
                logger.info(f"Stress limit reached at {max_successful_load} requests/second")
                break
        
        logger.info(f"Maximum sustainable load: {max_successful_load} requests/second")
    
    async def _run_single_benchmark(self, test_case: Dict[str, Any], iterations: int = 100):
        """Run a single benchmark test case"""
        logger.info(f"Running benchmark: {test_case['name']}")
        
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                if test_case['method'] == 'GET':
                    async with self.session.get(test_case['url']) as response:
                        await response.read()
                        success = response.status == 200
                        status_code = response.status
                        response_size = len(await response.read()) if success else 0
                else:  # POST
                    async with self.session.post(
                        test_case['url'], 
                        json=test_case.get('json', {})
                    ) as response:
                        await response.read()
                        success = response.status == 200
                        status_code = response.status
                        response_size = len(await response.read()) if success else 0
                
                duration_ms = (time.time() - start_time) * 1000
                
                result = BenchmarkResult(
                    test_name=test_case['name'],
                    duration_ms=duration_ms,
                    success=success,
                    response_size=response_size,
                    status_code=status_code
                )
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                result = BenchmarkResult(
                    test_name=test_case['name'],
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
            self.results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(test_case['name'], results)
        
        # Check performance expectations
        expected_time = test_case.get('expected_response_time', 1000)
        performance_ok = summary.p95_response_time <= expected_time
        
        logger.info(
            f"Benchmark results: {test_case['name']}",
            success_rate=f"{summary.success_rate:.2%}",
            avg_response_time=f"{summary.avg_response_time:.2f}ms",
            p95_response_time=f"{summary.p95_response_time:.2f}ms",
            requests_per_second=f"{summary.requests_per_second:.2f}",
            performance_ok=performance_ok,
            expected_time=expected_time
        )
    
    async def _simulate_user_session(self):
        """Simulate a typical user session"""
        try:
            # Get bio-twin data
            async with self.session.get(
                f"{self.base_url.replace('8000', '8005')}/bio-twin/{self.test_user_id}"
            ) as response:
                await response.read()
            
            # Analyze a protein
            async with self.session.post(
                f"{self.base_url.replace('8000', '8003')}/analyze_protein",
                json={
                    "sequence": self.test_protein_sequence[:200],
                    "analysis_type": "structure_prediction"
                }
            ) as response:
                await response.read()
            
            # Check agent status
            async with self.session.get(
                f"{self.base_url.replace('8000', '8004')}/multi_agent/status/{self.test_user_id}"
            ) as response:
                await response.read()
            
            return True
            
        except Exception as e:
            logger.error(f"User session simulation failed: {e}")
            return False
    
    async def _generate_load(self):
        """Generate load for resource monitoring"""
        tasks = []
        for _ in range(20):
            task = asyncio.create_task(self._simulate_user_session())
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _generate_sustained_load(self, requests_per_second: int, duration: int) -> float:
        """Generate sustained load and return success rate"""
        total_requests = requests_per_second * duration
        interval = 1.0 / requests_per_second
        
        tasks = []
        start_time = time.time()
        
        for i in range(total_requests):
            if time.time() - start_time >= duration:
                break
            
            task = asyncio.create_task(self._simulate_user_session())
            tasks.append(task)
            
            await asyncio.sleep(interval)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = [r for r in results if r is True]
        
        return len(successful) / len(results) if results else 0
    
    async def _collect_system_metrics(self, duration: int) -> Dict[str, float]:
        """Collect system metrics over a duration"""
        cpu_samples = []
        memory_samples = []
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            cpu_samples.append(psutil.cpu_percent())
            memory_samples.append(psutil.virtual_memory().percent)
            await asyncio.sleep(1)
        
        return {
            'cpu_avg': statistics.mean(cpu_samples),
            'cpu_max': max(cpu_samples),
            'memory_avg': statistics.mean(memory_samples),
            'memory_max': max(memory_samples)
        }
    
    def _calculate_summary(self, test_name: str, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Calculate summary statistics for benchmark results"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            response_times = [r.duration_ms for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = np.percentile(response_times, 50)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        total_duration = max(r.timestamp for r in results) - min(r.timestamp for r in results)
        total_duration_seconds = total_duration.total_seconds() if total_duration.total_seconds() > 0 else 1
        
        return BenchmarkSummary(
            test_name=test_name,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            success_rate=len(successful_results) / len(results),
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=len(results) / total_duration_seconds,
            total_duration=total_duration_seconds,
            error_rate=len(failed_results) / len(results)
        )
    
    async def _generate_reports(self):
        """Generate performance benchmark reports"""
        logger.info("Generating performance benchmark reports")
        
        # Group results by test name
        test_groups = {}
        for result in self.results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        # Generate summary report
        summaries = []
        for test_name, results in test_groups.items():
            summary = self._calculate_summary(test_name, results)
            summaries.append(summary)
        
        # Save to JSON
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summaries": [
                {
                    "test_name": s.test_name,
                    "total_requests": s.total_requests,
                    "success_rate": s.success_rate,
                    "avg_response_time": s.avg_response_time,
                    "p95_response_time": s.p95_response_time,
                    "p99_response_time": s.p99_response_time,
                    "requests_per_second": s.requests_per_second,
                    "error_rate": s.error_rate
                }
                for s in summaries
            ]
        }
        
        with open("benchmark_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Generate CSV for detailed analysis
        df = pd.DataFrame([
            {
                "test_name": r.test_name,
                "duration_ms": r.duration_ms,
                "success": r.success,
                "response_size": r.response_size,
                "status_code": r.status_code,
                "timestamp": r.timestamp.isoformat()
            }
            for r in self.results
        ])
        
        df.to_csv("benchmark_results.csv", index=False)
        
        logger.info("Performance benchmark reports generated")


async def main():
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    
    try:
        await benchmark.setup()
        await benchmark.run_all_benchmarks()
    finally:
        await benchmark.teardown()


if __name__ == "__main__":
    asyncio.run(main())
