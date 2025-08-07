"""
EvoHuman.AI ExoStack Distributed Compute Service
Lightweight distributed AI compute orchestration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import httpx
from typing import Dict, Any, List, Optional
import structlog
import asyncio
import json
import yaml
from datetime import datetime, timedelta
import uuid

from shared.models import ModelMetadata, ExplanationData, BioMetric, TwinInsight
from shared.constants import MODEL_VERSIONS, RISK_LEVELS
from shared.utils import setup_logging, create_health_check_response, generate_id, utc_now
from .batch_processor import BatchJobProcessor, JobType, JobStatus


# Setup logging
logger = setup_logging("exostack-service")

# Global state
node_registry: Dict[str, Dict[str, Any]] = {}
job_queue: Dict[str, Dict[str, Any]] = {}
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}
batch_processor: Optional[BatchJobProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global node_registry, job_queue, batch_processor
    
    logger.info("Starting ExoStack Distributed Compute Service")
    
    # Initialize node registry
    node_registry = {}
    job_queue = {}
    active_jobs = {}
    completed_jobs = {}
    
    # Initialize batch processor
    batch_processor = BatchJobProcessor()
    
    # Load existing node configurations if available
    try:
        node_config_path = "/app/configs/nodes.yaml"
        if os.path.exists(node_config_path):
            with open(node_config_path, 'r') as f:
                config = yaml.safe_load(f)
                node_registry.update(config.get('nodes', {}))
                logger.info(f"Loaded {len(node_registry)} registered nodes")
    except Exception as e:
        logger.warning(f"Failed to load node registry: {e}")
    
    # Start background tasks
    asyncio.create_task(_job_scheduler())
    asyncio.create_task(_node_health_checker())
    asyncio.create_task(_batch_job_monitor())
    
    yield
    
    logger.info("Shutting down ExoStack service")
    if batch_processor:
        await batch_processor.cleanup()


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI ExoStack Distributed Compute Service",
    description="Lightweight distributed AI compute orchestration",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {
        "node_registry": len(node_registry) > 0,
        "job_queue_active": len(job_queue) >= 0,
        "active_jobs": len(active_jobs),
        "completed_jobs": len(completed_jobs)
    }
    
    return create_health_check_response("exostack-service", dependencies)


# NODE MANAGEMENT

@app.post("/nodes/register")
async def register_node(node_info: Dict[str, Any]):
    """Register a new compute node"""
    
    try:
        node_id = node_info.get("node_id", generate_id())
        node_name = node_info.get("name", f"node-{node_id[:8]}")
        capabilities = node_info.get("capabilities", {})
        endpoint = node_info.get("endpoint", "")
        
        # Validate node info
        if not endpoint:
            raise HTTPException(status_code=400, detail="Node endpoint is required")
        
        # Test node connectivity
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/health")
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Node health check failed")
        except Exception as e:
            logger.warning(f"Node connectivity test failed: {e}")
        
        node_registry[node_id] = {
            "node_id": node_id,
            "name": node_name,
            "endpoint": endpoint,
            "capabilities": capabilities,
            "status": "active",
            "last_seen": utc_now().isoformat(),
            "registered_at": utc_now().isoformat(),
            "jobs_completed": 0,
            "total_compute_hours": 0.0
        }
        
        logger.info(f"Registered new node: {node_name} ({node_id})")
        
        return {
            "node_id": node_id,
            "status": "registered",
            "name": node_name,
            "capabilities": capabilities,
            "message": "Node successfully registered"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Node registration failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to register node")


@app.get("/nodes")
async def list_nodes(
    status: Optional[str] = None,
    capability: Optional[str] = None
):
    """List all registered nodes with optional filtering"""
    
    nodes = []
    for node_id, node_info in node_registry.items():
        # Apply filters
        if status and node_info.get("status") != status:
            continue
        if capability and capability not in node_info.get("capabilities", {}):
            continue
            
        nodes.append({
            **node_info,
            "uptime": _calculate_uptime(node_info.get("registered_at"))
        })
    
    return {
        "nodes": nodes,
        "total_nodes": len(nodes),
        "active_nodes": len([n for n in nodes if n["status"] == "active"]),
        "timestamp": utc_now().isoformat()
    }


@app.get("/nodes/{node_id}")
async def get_node_info(node_id: str):
    """Get detailed information about a specific node"""
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node_info = node_registry[node_id]
    
    # Get recent jobs for this node
    recent_jobs = []
    for job_id, job in completed_jobs.items():
        if job.get("assigned_node") == node_id:
            recent_jobs.append({
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "completed_at": job.get("completed_at"),
                "duration": job.get("duration", 0)
            })
    
    recent_jobs.sort(key=lambda x: x["completed_at"], reverse=True)
    
    return {
        **node_info,
        "recent_jobs": recent_jobs[:10],
        "uptime": _calculate_uptime(node_info.get("registered_at"))
    }


@app.delete("/nodes/{node_id}")
async def unregister_node(node_id: str):
    """Unregister a compute node"""
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Cancel any active jobs on this node
    for job_id, job in active_jobs.items():
        if job.get("assigned_node") == node_id:
            job["status"] = "cancelled"
            job["error"] = "Node unregistered"
            completed_jobs[job_id] = job
    
    # Remove from active jobs
    active_jobs = {k: v for k, v in active_jobs.items() if v.get("assigned_node") != node_id}
    
    # Remove from registry
    removed_node = node_registry.pop(node_id)
    
    logger.info(f"Unregistered node: {removed_node.get('name')} ({node_id})")
    
    return {
        "message": f"Node {node_id} successfully unregistered",
        "cancelled_jobs": len([j for j in active_jobs.values() if j.get("assigned_node") == node_id])
    }


# JOB MANAGEMENT

@app.post("/jobs/submit")
async def submit_job(job_request: Dict[str, Any]):
    """Submit a new compute job to the queue"""
    
    try:
        job_id = generate_id()
        job_type = job_request.get("job_type", "compute")
        priority = job_request.get("priority", "normal")
        requirements = job_request.get("requirements", {})
        
        # Validate job request
        if not job_request.get("task"):
            raise HTTPException(status_code=400, detail="Job task is required")
        
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "task": job_request["task"],
            "parameters": job_request.get("parameters", {}),
            "requirements": requirements,
            "priority": priority,
            "status": "queued",
            "submitted_at": utc_now().isoformat(),
            "estimated_duration": job_request.get("estimated_duration", 300),
            "max_retries": job_request.get("max_retries", 3),
            "retries": 0
        }
        
        # Add to queue
        job_queue[job_id] = job
        
        logger.info(f"Job submitted: {job_type} ({job_id})")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position_in_queue": len(job_queue),
            "estimated_wait_time": _estimate_wait_time(),
            "message": "Job successfully submitted to queue"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Job submission failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to submit job")


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50
):
    """List jobs with optional filtering"""
    
    all_jobs = {**job_queue, **active_jobs, **completed_jobs}
    
    jobs = []
    for job_id, job in all_jobs.items():
        # Apply filters
        if status and job.get("status") != status:
            continue
        if job_type and job.get("job_type") != job_type:
            continue
            
        jobs.append({
            "job_id": job_id,
            "job_type": job.get("job_type"),
            "status": job.get("status"),
            "submitted_at": job.get("submitted_at"),
            "assigned_node": job.get("assigned_node"),
            "progress": job.get("progress", 0),
            "estimated_completion": job.get("estimated_completion")
        })
    
    # Sort by submission time (newest first)
    jobs.sort(key=lambda x: x["submitted_at"], reverse=True)
    
    return {
        "jobs": jobs[:limit],
        "total_jobs": len(jobs),
        "queued": len(job_queue),
        "active": len(active_jobs),
        "completed": len(completed_jobs),
        "timestamp": utc_now().isoformat()
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get detailed status of a specific job"""
    
    # Check all job stores
    job = None
    if job_id in job_queue:
        job = job_queue[job_id]
    elif job_id in active_jobs:
        job = active_jobs[job_id]
    elif job_id in completed_jobs:
        job = completed_jobs[job_id]
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Add runtime information
    job_info = {**job}
    
    if job.get("status") == "running":
        started_at = datetime.fromisoformat(job.get("started_at"))
        elapsed = (utc_now() - started_at).total_seconds()
        job_info["elapsed_time"] = elapsed
        
        if job.get("estimated_duration"):
            progress = min(elapsed / job["estimated_duration"], 1.0)
            job_info["estimated_progress"] = progress
    
    return job_info


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job"""
    
    # Check if job exists and can be cancelled
    if job_id in completed_jobs:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    
    job = None
    if job_id in job_queue:
        job = job_queue.pop(job_id)
    elif job_id in active_jobs:
        job = active_jobs.pop(job_id)
    else:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update job status
    job["status"] = "cancelled"
    job["cancelled_at"] = utc_now().isoformat()
    completed_jobs[job_id] = job
    
    # If job was running, notify the node
    if job.get("assigned_node"):
        asyncio.create_task(_notify_node_cancel(job.get("assigned_node"), job_id))
    
    logger.info(f"Job cancelled: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job successfully cancelled"
    }


# JOB EXECUTION ENDPOINTS (for nodes to report back)

@app.post("/jobs/{job_id}/start")
async def start_job_execution(job_id: str, node_info: Dict[str, Any]):
    """Node reports that job execution has started"""
    
    if job_id not in job_queue:
        raise HTTPException(status_code=404, detail="Job not found in queue")
    
    job = job_queue.pop(job_id)
    job["status"] = "running"
    job["started_at"] = utc_now().isoformat()
    job["assigned_node"] = node_info.get("node_id")
    
    active_jobs[job_id] = job
    
    logger.info(f"Job started: {job_id} on node {node_info.get('node_id')}")
    
    return {"status": "acknowledged", "job_id": job_id}


@app.post("/jobs/{job_id}/progress")
async def update_job_progress(job_id: str, progress_info: Dict[str, Any]):
    """Node reports job progress"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Active job not found")
    
    job = active_jobs[job_id]
    job["progress"] = progress_info.get("progress", job.get("progress", 0))
    job["last_update"] = utc_now().isoformat()
    
    if progress_info.get("intermediate_results"):
        job["intermediate_results"] = progress_info["intermediate_results"]
    
    return {"status": "acknowledged", "job_id": job_id}


@app.post("/jobs/{job_id}/complete")
async def complete_job(job_id: str, completion_info: Dict[str, Any]):
    """Node reports job completion"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Active job not found")
    
    job = active_jobs.pop(job_id)
    job["status"] = "completed"
    job["completed_at"] = utc_now().isoformat()
    job["results"] = completion_info.get("results", {})
    job["duration"] = completion_info.get("duration", 0)
    
    completed_jobs[job_id] = job
    
    # Update node statistics
    node_id = job.get("assigned_node")
    if node_id and node_id in node_registry:
        node_registry[node_id]["jobs_completed"] += 1
        node_registry[node_id]["total_compute_hours"] += job["duration"] / 3600
        node_registry[node_id]["last_seen"] = utc_now().isoformat()
    
    logger.info(f"Job completed: {job_id} in {job['duration']}s")
    
    return {"status": "acknowledged", "job_id": job_id}


@app.post("/jobs/{job_id}/error")
async def report_job_error(job_id: str, error_info: Dict[str, Any]):
    """Node reports job error"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Active job not found")
    
    job = active_jobs.pop(job_id)
    job["status"] = "failed"
    job["failed_at"] = utc_now().isoformat()
    job["error"] = error_info.get("error", "Unknown error")
    job["error_details"] = error_info.get("details", {})
    
    # Check if we should retry
    if job["retries"] < job.get("max_retries", 3):
        job["retries"] += 1
        job["status"] = "queued"
        job_queue[job_id] = job
        logger.info(f"Job failed, retrying: {job_id} (attempt {job['retries']})")
    else:
        completed_jobs[job_id] = job
        logger.error(f"Job failed permanently: {job_id}")
    
    return {"status": "acknowledged", "job_id": job_id}


# CLUSTER MANAGEMENT

@app.get("/cluster/status")
async def get_cluster_status():
    """Get overall cluster status and metrics"""
    
    # Calculate cluster metrics
    total_nodes = len(node_registry)
    active_nodes = len([n for n in node_registry.values() if n["status"] == "active"])
    
    # Job statistics
    total_jobs = len(job_queue) + len(active_jobs) + len(completed_jobs)
    completed_job_count = len(completed_jobs)
    success_rate = completed_job_count / max(total_jobs, 1) * 100
    
    # Compute capacity
    total_capacity = sum(
        n.get("capabilities", {}).get("max_concurrent_jobs", 1) 
        for n in node_registry.values() if n["status"] == "active"
    )
    
    current_utilization = len(active_jobs) / max(total_capacity, 1) * 100
    
    return {
        "cluster_id": "evohuman-exostack",
        "status": "operational" if active_nodes > 0 else "no_nodes",
        "nodes": {
            "total": total_nodes,
            "active": active_nodes,
            "inactive": total_nodes - active_nodes
        },
        "jobs": {
            "queued": len(job_queue),
            "active": len(active_jobs),
            "completed": len(completed_jobs),
            "total": total_jobs,
            "success_rate": round(success_rate, 1)
        },
        "capacity": {
            "total_slots": total_capacity,
            "used_slots": len(active_jobs),
            "utilization": round(current_utilization, 1)
        },
        "uptime": _calculate_uptime(utc_now().isoformat()),
        "timestamp": utc_now().isoformat()
    }


@app.post("/cluster/scale")
async def scale_cluster(scale_request: Dict[str, Any]):
    """Request cluster scaling (add/remove nodes)"""
    
    target_nodes = scale_request.get("target_nodes", len(node_registry))
    current_nodes = len(node_registry)
    
    # This is a placeholder - actual implementation would integrate with
    # cloud providers or container orchestration systems
    
    return {
        "current_nodes": current_nodes,
        "target_nodes": target_nodes,
        "scaling_action": "requested",
        "message": "Cluster scaling request submitted",
        "timestamp": utc_now().isoformat()
    }


# HELPER FUNCTIONS

def _calculate_uptime(start_time_iso: str) -> str:
    """Calculate uptime from start time"""
    try:
        start_time = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        uptime = utc_now() - start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{days}d {hours}h {minutes}m"
    except:
        return "unknown"


def _estimate_wait_time() -> str:
    """Estimate wait time for new jobs"""
    if not job_queue:
        return "0 minutes"
    
    # Simple estimation based on queue length and active nodes
    active_nodes = len([n for n in node_registry.values() if n["status"] == "active"])
    if active_nodes == 0:
        return "unknown (no active nodes)"
    
    avg_job_time = 300  # 5 minutes average
    queue_position = len(job_queue)
    estimated_seconds = (queue_position * avg_job_time) / max(active_nodes, 1)
    
    if estimated_seconds < 60:
        return "< 1 minute"
    elif estimated_seconds < 3600:
        return f"~{int(estimated_seconds / 60)} minutes"
    else:
        return f"~{int(estimated_seconds / 3600)} hours"


async def _notify_node_cancel(node_id: str, job_id: str):
    """Notify node that job was cancelled"""
    if node_id not in node_registry:
        return
    
    node_endpoint = node_registry[node_id]["endpoint"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{node_endpoint}/jobs/{job_id}/cancel")
    except Exception as e:
        logger.warning(f"Failed to notify node {node_id} of job cancellation: {e}")


# BACKGROUND TASKS

async def _job_scheduler():
    """Background task to assign jobs to available nodes"""
    while True:
        try:
            # Get available nodes
            available_nodes = [
                (node_id, node) for node_id, node in node_registry.items()
                if node["status"] == "active"
            ]
            
            if not available_nodes or not job_queue:
                await asyncio.sleep(5)
                continue
            
            # Assign jobs to nodes (simple FIFO for now)
            for job_id, job in list(job_queue.items()):
                # Find suitable node
                suitable_node = None
                for node_id, node in available_nodes:
                    capabilities = node.get("capabilities", {})
                    requirements = job.get("requirements", {})
                    
                    # Check if node can handle this job type
                    if requirements.get("job_type") and requirements["job_type"] not in capabilities.get("supported_types", []):
                        continue
                    
                    # Check resource requirements
                    if requirements.get("min_memory") and requirements["min_memory"] > capabilities.get("memory_gb", 0):
                        continue
                    
                    suitable_node = (node_id, node)
                    break
                
                if suitable_node:
                    node_id, node = suitable_node
                    
                    # Assign job to node
                    try:
                        endpoint = node["endpoint"]
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.post(
                                f"{endpoint}/execute",
                                json={
                                    "job_id": job_id,
                                    "task": job["task"],
                                    "parameters": job.get("parameters", {})
                                }
                            )
                            
                            if response.status_code == 200:
                                # Move job from queue to active
                                job_queue.pop(job_id)
                                job["status"] = "assigned"
                                job["assigned_node"] = node_id
                                job["assigned_at"] = utc_now().isoformat()
                                active_jobs[job_id] = job
                                
                                logger.info(f"Job {job_id} assigned to node {node_id}")
                            else:
                                logger.warning(f"Node {node_id} rejected job {job_id}: {response.status_code}")
                                
                    except Exception as e:
                        logger.error(f"Failed to assign job {job_id} to node {node_id}: {e}")
                        # Mark node as inactive if connection fails
                        node_registry[node_id]["status"] = "inactive"
                
                # Break after assigning one job per cycle
                break
                
        except Exception as e:
            logger.error(f"Job scheduler error: {e}")
        
        await asyncio.sleep(2)


async def _node_health_checker():
    """Background task to check node health"""
    while True:
        try:
            for node_id, node in list(node_registry.items()):
                if node["status"] != "active":
                    continue
                
                try:
                    endpoint = node["endpoint"]
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{endpoint}/health")
                        if response.status_code == 200:
                            node["last_seen"] = utc_now().isoformat()
                        else:
                            node["status"] = "unhealthy"
                            logger.warning(f"Node {node_id} health check failed: {response.status_code}")
                            
                except Exception as e:
                    node["status"] = "inactive"
                    logger.warning(f"Node {node_id} became inactive: {e}")
        
        except Exception as e:
            logger.error(f"Node health checker error: {e}")
        
        await asyncio.sleep(30)


async def _batch_job_monitor():
    """Background task to monitor batch job health"""
    while True:
        try:
            if batch_processor:
                health_report = await batch_processor.monitor_job_health()
                if health_report.get("timed_out_jobs", 0) > 0:
                    logger.warning(
                        "Batch jobs timed out",
                        timed_out_count=health_report["timed_out_jobs"]
                    )
        except Exception as e:
            logger.error(f"Batch job monitor error: {e}")
        
        await asyncio.sleep(60)  # Check every minute


# BATCH PROCESSING ENDPOINTS

@app.post("/batch/esm3/protein_analysis")
async def submit_esm3_batch_job(
    user_id: str,
    sequences: List[str],
    analysis_type: str = "structure_prediction"
):
    """Submit batch ESM3 protein analysis job"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    try:
        result = await batch_processor.process_esm3_batch(
            user_id=user_id,
            sequences=sequences,
            analysis_type=analysis_type
        )
        
        return {
            "status": "success",
            "batch_job_submitted": True,
            **result
        }
        
    except Exception as e:
        logger.error(
            "ESM3 batch job submission failed",
            user_id=user_id,
            sequence_count=len(sequences),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Batch job failed: {str(e)}")


@app.post("/batch/aice/cognitive_tasks")
async def submit_aice_batch_job(
    user_id: str,
    cognitive_tasks: List[Dict[str, Any]]
):
    """Submit batch AiCE cognitive enhancement job"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    try:
        result = await batch_processor.process_aice_cognitive_batch(
            user_id=user_id,
            cognitive_tasks=cognitive_tasks
        )
        
        return {
            "status": "success",
            "batch_job_submitted": True,
            **result
        }
        
    except Exception as e:
        logger.error(
            "AiCE batch job submission failed",
            user_id=user_id,
            task_count=len(cognitive_tasks),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Batch job failed: {str(e)}")


@app.get("/batch/jobs/{job_id}/status")
async def get_batch_job_status(job_id: str):
    """Get status of a specific batch job"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    job_status = batch_processor.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    return job_status


@app.post("/batch/jobs/{job_id}/retry")
async def retry_batch_job(job_id: str):
    """Retry a failed batch job"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    try:
        result = await batch_processor.retry_failed_job(job_id)
        
        return {
            "status": "success",
            "job_retried": True,
            "job_id": job_id,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Batch job retry failed",
            job_id=job_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")


@app.get("/batch/metrics")
async def get_batch_metrics():
    """Get comprehensive batch processing metrics"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    try:
        metrics = await batch_processor.get_metrics_summary()
        return metrics
    except Exception as e:
        logger.error("Failed to get batch metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.get("/batch/health")
async def check_batch_health():
    """Check health of batch processing system"""
    if not batch_processor:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")
    
    try:
        health_report = await batch_processor.monitor_job_health()
        service_health = await batch_processor._check_service_health()
        
        return {
            "batch_processor_status": "healthy",
            "job_health": health_report,
            "ai_services": service_health,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Batch health check failed", error=str(e))
        return {
            "batch_processor_status": "unhealthy",
            "error": str(e),
            "timestamp": utc_now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
