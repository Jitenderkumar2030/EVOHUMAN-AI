"""
Enhanced Batch Job Processor for ExoStack
Handles batch ESM3 and AiCE workloads with retry logic and monitoring
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import httpx
import structlog
import yaml
from enum import Enum

from shared.models import ModelMetadata, ComputeJob, ComputeNode
from shared.utils import setup_logging, generate_id, utc_now
from shared.constants import MODEL_VERSIONS


class JobType(str, Enum):
    ESM3_PROTEIN_ANALYSIS = "esm3_protein_analysis"
    ESM3_BATCH_ANALYSIS = "esm3_batch_analysis"
    ESM3_MUTATION_PREDICTION = "esm3_mutation_prediction"
    AICE_COGNITIVE_ENHANCEMENT = "aice_cognitive_enhancement"
    AICE_MEMORY_CONSOLIDATION = "aice_memory_consolidation"
    PROTEUS_REGENERATION = "proteus_regeneration"
    BIOTWINS_ANALYSIS = "biotwins_analysis"
    CUSTOM_COMPUTE = "custom_compute"


class JobStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class BatchJobProcessor:
    """Enhanced batch job processor with monitoring and retry capabilities"""
    
    def __init__(self, redis_client=None):
        self.logger = setup_logging("batch_processor")
        
        # Service clients for different AI models
        self.service_clients = {
            "esm3": httpx.AsyncClient(
                base_url=os.getenv("ESM3_SERVICE_URL", "http://esm3-service:8000"),
                timeout=300.0  # 5 minutes timeout for batch jobs
            ),
            "aice": httpx.AsyncClient(
                base_url=os.getenv("AICE_SERVICE_URL", "http://aice-service:8000"),
                timeout=180.0
            ),
            "proteus": httpx.AsyncClient(
                base_url=os.getenv("PROTEUS_SERVICE_URL", "http://proteus-service:8000"),
                timeout=240.0
            ),
            "bio_twin": httpx.AsyncClient(
                base_url=os.getenv("BIO_TWIN_URL", "http://bio-twin:8000"),
                timeout=120.0
            )
        }
        
        # Job processing configuration
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
        self.job_timeout = int(os.getenv("JOB_TIMEOUT_SECONDS", "1800"))  # 30 minutes
        self.max_retries = int(os.getenv("MAX_JOB_RETRIES", "3"))
        self.retry_delays = [30, 60, 120]  # Progressive retry delays in seconds
        
        # Job monitoring
        self.job_metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
            "average_duration": 0.0
        }
        
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_history: List[Dict[str, Any]] = []
    
    async def process_esm3_batch(
        self, 
        user_id: str, 
        sequences: List[str], 
        analysis_type: str = "structure_prediction",
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a batch of ESM3 protein analysis jobs"""
        
        job_id = job_id or generate_id()
        start_time = time.time()
        
        self.logger.info(
            "Starting ESM3 batch processing",
            job_id=job_id,
            user_id=user_id,
            sequence_count=len(sequences),
            analysis_type=analysis_type
        )
        
        # Create batch job record
        batch_job = {
            "job_id": job_id,
            "job_type": JobType.ESM3_BATCH_ANALYSIS,
            "user_id": user_id,
            "status": JobStatus.RUNNING,
            "total_sequences": len(sequences),
            "completed_sequences": 0,
            "failed_sequences": 0,
            "results": [],
            "errors": [],
            "started_at": utc_now().isoformat(),
            "progress": 0.0
        }
        
        self.active_jobs[job_id] = batch_job
        
        try:
            # Process sequences in smaller batches to avoid overwhelming the service
            batch_size = min(20, len(sequences))  # Process up to 20 at a time
            results = []
            errors = []
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                # Process current batch
                batch_results = await self._process_esm3_sequence_batch(
                    user_id, batch, analysis_type, job_id, i // batch_size + 1
                )
                
                # Collect results and errors
                for result in batch_results:
                    if result.get("status") == "success":
                        results.append(result)
                        batch_job["completed_sequences"] += 1
                    else:
                        errors.append(result)
                        batch_job["failed_sequences"] += 1
                
                # Update progress
                progress = (i + len(batch)) / len(sequences)
                batch_job["progress"] = progress
                batch_job["results"] = results
                batch_job["errors"] = errors
                
                self.logger.info(
                    "ESM3 batch progress update",
                    job_id=job_id,
                    progress=f"{progress:.1%}",
                    completed=batch_job["completed_sequences"],
                    failed=batch_job["failed_sequences"]
                )
                
                # Small delay between batches to prevent overwhelming
                if i + batch_size < len(sequences):
                    await asyncio.sleep(2)
            
            # Finalize job
            duration = time.time() - start_time
            batch_job.update({
                "status": JobStatus.COMPLETED,
                "completed_at": utc_now().isoformat(),
                "duration": duration,
                "success_rate": batch_job["completed_sequences"] / len(sequences) * 100
            })
            
            # Update metrics
            self._update_job_metrics(batch_job)
            
            self.logger.info(
                "ESM3 batch processing completed",
                job_id=job_id,
                duration=f"{duration:.2f}s",
                success_rate=f"{batch_job['success_rate']:.1f}%",
                total_results=len(results)
            )
            
            return {
                "job_id": job_id,
                "status": "completed",
                "total_sequences": len(sequences),
                "successful_analyses": len(results),
                "failed_analyses": len(errors),
                "results": results,
                "errors": errors,
                "duration": duration,
                "success_rate": batch_job["success_rate"]
            }
            
        except Exception as e:
            # Handle job failure
            duration = time.time() - start_time
            batch_job.update({
                "status": JobStatus.FAILED,
                "failed_at": utc_now().isoformat(),
                "duration": duration,
                "error": str(e)
            })
            
            self.logger.error(
                "ESM3 batch processing failed",
                job_id=job_id,
                error=str(e),
                duration=f"{duration:.2f}s"
            )
            
            raise
        
        finally:
            # Move job to history
            self.job_history.append(self.active_jobs.pop(job_id, batch_job))
            # Keep only last 1000 jobs in history
            self.job_history = self.job_history[-1000:]
    
    async def _process_esm3_sequence_batch(
        self, 
        user_id: str, 
        sequences: List[str], 
        analysis_type: str,
        parent_job_id: str,
        batch_number: int
    ) -> List[Dict[str, Any]]:
        """Process a single batch of sequences"""
        
        try:
            # Prepare batch request
            batch_data = {
                "user_id": user_id,
                "sequences": [
                    {
                        "sequence": seq,
                        "analysis_type": analysis_type,
                        "include_mutations": True,
                        "include_evolution": True
                    }
                    for seq in sequences
                ],
                "batch_id": f"{parent_job_id}_batch_{batch_number}"
            }
            
            # Send request to ESM3 service
            response = await self.service_clients["esm3"].post(
                "/batch_analyze",
                json=batch_data
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Process and format results
                results = []
                for i, sequence in enumerate(sequences):
                    if i < len(result_data.get("results", [])):
                        result = result_data["results"][i]
                        results.append({
                            "status": "success",
                            "sequence": sequence,
                            "sequence_length": len(sequence),
                            "analysis_result": result,
                            "processing_time": result.get("processing_time", 0),
                            "confidence_score": result.get("confidence_score", 0.0)
                        })
                    else:
                        results.append({
                            "status": "error",
                            "sequence": sequence,
                            "error": "No result returned for sequence"
                        })
                
                return results
            else:
                # Handle service error
                error_msg = f"ESM3 service error: {response.status_code}"
                return [
                    {
                        "status": "error",
                        "sequence": seq,
                        "error": error_msg
                    }
                    for seq in sequences
                ]
                
        except Exception as e:
            # Handle request error
            error_msg = f"Request failed: {str(e)}"
            return [
                {
                    "status": "error",
                    "sequence": seq,
                    "error": error_msg
                }
                for seq in sequences
            ]
    
    async def process_aice_cognitive_batch(
        self,
        user_id: str,
        cognitive_tasks: List[Dict[str, Any]],
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a batch of AiCE cognitive enhancement tasks"""
        
        job_id = job_id or generate_id()
        start_time = time.time()
        
        self.logger.info(
            "Starting AiCE cognitive batch processing",
            job_id=job_id,
            user_id=user_id,
            task_count=len(cognitive_tasks)
        )
        
        batch_job = {
            "job_id": job_id,
            "job_type": JobType.AICE_COGNITIVE_ENHANCEMENT,
            "user_id": user_id,
            "status": JobStatus.RUNNING,
            "total_tasks": len(cognitive_tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "results": [],
            "errors": [],
            "started_at": utc_now().isoformat(),
            "progress": 0.0
        }
        
        self.active_jobs[job_id] = batch_job
        
        try:
            results = []
            errors = []
            
            # Process tasks concurrently but with limits
            semaphore = asyncio.Semaphore(5)  # Limit concurrent tasks
            
            async def process_single_task(task_data: Dict[str, Any], index: int):
                async with semaphore:
                    try:
                        # Add user context
                        task_request = {
                            "user_id": user_id,
                            "task_type": task_data.get("task_type", "cognitive_enhancement"),
                            "parameters": task_data.get("parameters", {}),
                            "context": task_data.get("context", {}),
                            "job_id": job_id,
                            "task_index": index
                        }
                        
                        response = await self.service_clients["aice"].post(
                            "/cognitive/enhance",
                            json=task_request
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            batch_job["completed_tasks"] += 1
                            return {
                                "status": "success",
                                "task_index": index,
                                "task_type": task_data.get("task_type"),
                                "result": result,
                                "processing_time": result.get("processing_time", 0)
                            }
                        else:
                            batch_job["failed_tasks"] += 1
                            return {
                                "status": "error",
                                "task_index": index,
                                "error": f"AiCE service error: {response.status_code}"
                            }
                            
                    except Exception as e:
                        batch_job["failed_tasks"] += 1
                        return {
                            "status": "error",
                            "task_index": index,
                            "error": str(e)
                        }
            
            # Process all tasks
            task_results = await asyncio.gather(
                *[process_single_task(task, i) for i, task in enumerate(cognitive_tasks)],
                return_exceptions=True
            )
            
            # Collect results
            for result in task_results:
                if isinstance(result, Exception):
                    errors.append({"error": str(result)})
                elif result.get("status") == "success":
                    results.append(result)
                else:
                    errors.append(result)
            
            # Finalize job
            duration = time.time() - start_time
            success_rate = len(results) / len(cognitive_tasks) * 100
            
            batch_job.update({
                "status": JobStatus.COMPLETED,
                "completed_at": utc_now().isoformat(),
                "duration": duration,
                "results": results,
                "errors": errors,
                "success_rate": success_rate,
                "progress": 1.0
            })
            
            self._update_job_metrics(batch_job)
            
            self.logger.info(
                "AiCE cognitive batch processing completed",
                job_id=job_id,
                duration=f"{duration:.2f}s",
                success_rate=f"{success_rate:.1f}%"
            )
            
            return {
                "job_id": job_id,
                "status": "completed",
                "total_tasks": len(cognitive_tasks),
                "successful_tasks": len(results),
                "failed_tasks": len(errors),
                "results": results,
                "errors": errors,
                "duration": duration,
                "success_rate": success_rate
            }
            
        except Exception as e:
            duration = time.time() - start_time
            batch_job.update({
                "status": JobStatus.FAILED,
                "failed_at": utc_now().isoformat(),
                "duration": duration,
                "error": str(e)
            })
            
            self.logger.error(
                "AiCE cognitive batch processing failed",
                job_id=job_id,
                error=str(e)
            )
            
            raise
        
        finally:
            self.job_history.append(self.active_jobs.pop(job_id, batch_job))
            self.job_history = self.job_history[-1000:]
    
    async def retry_failed_job(self, job_id: str) -> Dict[str, Any]:
        """Retry a failed job with exponential backoff"""
        
        # Find job in history
        job = None
        for historical_job in self.job_history:
            if historical_job.get("job_id") == job_id:
                job = historical_job
                break
        
        if not job:
            raise ValueError(f"Job {job_id} not found in history")
        
        if job.get("status") != JobStatus.FAILED:
            raise ValueError(f"Job {job_id} is not in failed state")
        
        retries = job.get("retries", 0)
        if retries >= self.max_retries:
            raise ValueError(f"Job {job_id} has exceeded maximum retries ({self.max_retries})")
        
        # Calculate retry delay
        delay = self.retry_delays[min(retries, len(self.retry_delays) - 1)]
        
        self.logger.info(
            "Retrying failed job",
            job_id=job_id,
            retry_attempt=retries + 1,
            delay=delay
        )
        
        # Wait before retrying
        await asyncio.sleep(delay)
        
        # Update job for retry
        job.update({
            "status": JobStatus.RETRYING,
            "retries": retries + 1,
            "retry_started_at": utc_now().isoformat()
        })
        
        try:
            # Retry based on job type
            if job["job_type"] == JobType.ESM3_BATCH_ANALYSIS:
                # Extract original parameters and retry
                original_sequences = [r.get("sequence") for r in job.get("results", []) + job.get("errors", [])]
                result = await self.process_esm3_batch(
                    job["user_id"],
                    original_sequences,
                    job.get("analysis_type", "structure_prediction"),
                    job_id
                )
            elif job["job_type"] == JobType.AICE_COGNITIVE_ENHANCEMENT:
                # Retry cognitive tasks
                original_tasks = job.get("original_tasks", [])
                result = await self.process_aice_cognitive_batch(
                    job["user_id"],
                    original_tasks,
                    job_id
                )
            else:
                raise ValueError(f"Retry not supported for job type: {job['job_type']}")
            
            self.job_metrics["retries"] += 1
            return result
            
        except Exception as e:
            self.logger.error(
                "Job retry failed",
                job_id=job_id,
                retry_attempt=retries + 1,
                error=str(e)
            )
            
            # Update job with retry failure
            job.update({
                "status": JobStatus.FAILED,
                "retry_failed_at": utc_now().isoformat(),
                "last_error": str(e)
            })
            
            raise
    
    async def monitor_job_health(self) -> Dict[str, Any]:
        """Monitor health of all active jobs and detect timeouts"""
        
        current_time = utc_now()
        timeout_threshold = timedelta(seconds=self.job_timeout)
        
        timed_out_jobs = []
        healthy_jobs = []
        
        for job_id, job in self.active_jobs.items():
            started_at = datetime.fromisoformat(job["started_at"])
            elapsed = current_time - started_at
            
            if elapsed > timeout_threshold:
                timed_out_jobs.append(job_id)
                job["status"] = JobStatus.FAILED
                job["failed_at"] = current_time.isoformat()
                job["error"] = f"Job timeout after {elapsed.total_seconds():.0f} seconds"
                
                self.logger.warning(
                    "Job timed out",
                    job_id=job_id,
                    elapsed_time=f"{elapsed.total_seconds():.0f}s",
                    timeout_threshold=f"{timeout_threshold.total_seconds():.0f}s"
                )
            else:
                healthy_jobs.append(job_id)
        
        # Move timed out jobs to history
        for job_id in timed_out_jobs:
            if job_id in self.active_jobs:
                self.job_history.append(self.active_jobs.pop(job_id))
        
        return {
            "timestamp": current_time.isoformat(),
            "active_jobs": len(healthy_jobs),
            "timed_out_jobs": len(timed_out_jobs),
            "timed_out_job_ids": timed_out_jobs,
            "job_timeout_seconds": self.job_timeout,
            "max_concurrent_jobs": self.max_concurrent_jobs
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific job"""
        
        # Check active jobs first
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id].copy()
            
            # Add real-time progress calculation
            if job["status"] == JobStatus.RUNNING:
                started_at = datetime.fromisoformat(job["started_at"])
                elapsed = (utc_now() - started_at).total_seconds()
                job["elapsed_time"] = elapsed
                job["estimated_completion"] = None  # Would require better tracking
            
            return job
        
        # Check job history
        for job in self.job_history:
            if job.get("job_id") == job_id:
                return job.copy()
        
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        
        # Calculate recent success rate (last 100 jobs)
        recent_jobs = self.job_history[-100:]
        recent_success_rate = 0.0
        if recent_jobs:
            successful = len([j for j in recent_jobs if j.get("status") == JobStatus.COMPLETED])
            recent_success_rate = successful / len(recent_jobs) * 100
        
        # Calculate average processing time
        completed_jobs = [j for j in self.job_history if j.get("duration")]
        avg_duration = 0.0
        if completed_jobs:
            avg_duration = sum(j["duration"] for j in completed_jobs) / len(completed_jobs)
        
        return {
            "total_processed": self.job_metrics["total_processed"],
            "successful": self.job_metrics["successful"],
            "failed": self.job_metrics["failed"],
            "retries": self.job_metrics["retries"],
            "active_jobs": len(self.active_jobs),
            "recent_success_rate": round(recent_success_rate, 1),
            "average_duration": round(avg_duration, 2),
            "job_history_size": len(self.job_history),
            "service_health": await self._check_service_health()
        }
    
    async def _check_service_health(self) -> Dict[str, bool]:
        """Check health of all connected AI services"""
        
        health_status = {}
        
        for service_name, client in self.service_clients.items():
            try:
                response = await client.get("/health", timeout=5.0)
                health_status[service_name] = response.status_code == 200
            except Exception:
                health_status[service_name] = False
        
        return health_status
    
    def _update_job_metrics(self, job: Dict[str, Any]):
        """Update internal job metrics"""
        
        self.job_metrics["total_processed"] += 1
        
        if job.get("status") == JobStatus.COMPLETED:
            self.job_metrics["successful"] += 1
        elif job.get("status") == JobStatus.FAILED:
            self.job_metrics["failed"] += 1
        
        # Update average duration
        if job.get("duration"):
            current_avg = self.job_metrics["average_duration"]
            total = self.job_metrics["total_processed"]
            
            new_avg = ((current_avg * (total - 1)) + job["duration"]) / total
            self.job_metrics["average_duration"] = new_avg
    
    async def cleanup(self):
        """Cleanup resources"""
        for client in self.service_clients.values():
            await client.aclose()
