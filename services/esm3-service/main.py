"""
EvoHuman.AI ESM3 Bio-Intelligence Service
Facebook ESM3 protein modeling integration for structure prediction and evolution analysis
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
import torch
import numpy as np
from datetime import datetime
import yaml

from shared.models import (
    AIServiceRequest, AIServiceResponse, ESM3AnalysisResult, 
    ModelMetadata, ExplanationData
)
from shared.constants import MODEL_VERSIONS, RISK_LEVELS
from shared.utils import setup_logging, create_health_check_response, generate_id, utc_now
from .esm3_engine import ESM3Engine
from .protein_analyzer import ProteinAnalyzer
from .evolution_predictor import EvolutionPredictor


# Setup logging
logger = setup_logging("esm3-service")

# Global components
esm3_engine: Optional[ESM3Engine] = None
protein_analyzer: Optional[ProteinAnalyzer] = None
evolution_predictor: Optional[EvolutionPredictor] = None
exostack_client: Optional[httpx.AsyncClient] = None
config: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global esm3_engine, protein_analyzer, evolution_predictor, exostack_client, config
    
    logger.info("Starting ESM3 Bio-Intelligence Service")
    
    # Load configuration
    try:
        with open("/app/configs/models.yaml", "r") as f:
            full_config = yaml.safe_load(f)
            config = full_config.get("esm3", {})
        logger.info("Configuration loaded", config=config)
    except Exception as e:
        logger.warning("Failed to load config, using defaults", error=str(e))
        config = {
            "model_name": "esm3_sm_open_v1",
            "model_path": "/app/models/esm3",
            "parameters": {
                "max_sequence_length": 1024,
                "batch_size": 8,
                "temperature": 0.7,
                "top_k": 50
            }
        }
    
    # Initialize ExoStack client
    exostack_url = os.getenv("EXOSTACK_SERVICE_URL", "http://exostack-service:8000")
    exostack_client = httpx.AsyncClient(base_url=exostack_url, timeout=300.0)
    
    # Initialize ESM3 components
    try:
        esm3_engine = ESM3Engine(config)
        await esm3_engine.initialize()
        
        protein_analyzer = ProteinAnalyzer(esm3_engine, config)
        evolution_predictor = EvolutionPredictor(esm3_engine, config)
        
        logger.info("ESM3 service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize ESM3 service", error=str(e))
        # Continue without model for development/testing
        logger.warning("Running in development mode without ESM3 model")
    
    yield
    
    # Cleanup
    logger.info("Shutting down ESM3 service")
    if exostack_client:
        await exostack_client.aclose()
    if esm3_engine:
        await esm3_engine.cleanup()


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI ESM3 Bio-Intelligence Service",
    description="Facebook ESM3 protein modeling for structure prediction and evolution analysis",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {
        "esm3_model": esm3_engine is not None and esm3_engine.is_ready(),
        "protein_analyzer": protein_analyzer is not None,
        "evolution_predictor": evolution_predictor is not None,
        "gpu_available": torch.cuda.is_available(),
        "exostack_connection": exostack_client is not None
    }
    
    # Check ExoStack connectivity
    if exostack_client:
        try:
            response = await exostack_client.get("/health", timeout=5.0)
            dependencies["exostack_service"] = response.status_code == 200
        except Exception:
            dependencies["exostack_service"] = False
    
    return create_health_check_response("esm3-service", dependencies)


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_protein_sequence(
    sequence: str,
    analysis_type: str = "structure_prediction",
    include_mutations: bool = False,
    include_evolution: bool = False,
    user_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Analyze protein sequence using ESM3 model
    
    Args:
        sequence: Protein amino acid sequence
        analysis_type: Type of analysis (structure_prediction, mutation_analysis, evolution_analysis)
        include_mutations: Whether to include mutation effect analysis
        include_evolution: Whether to include evolutionary pathway analysis
        user_id: User ID for tracking
    
    Returns:
        JSON response with analysis results
    """
    if not esm3_engine or not protein_analyzer:
        # Return mock response for development
        return await _mock_protein_analysis(sequence, analysis_type, include_mutations, include_evolution)
    
    try:
        # Validate sequence
        if not _validate_protein_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        if len(sequence) > config.get("parameters", {}).get("max_sequence_length", 1024):
            raise HTTPException(status_code=400, detail="Sequence too long")
        
        # Generate analysis ID
        analysis_id = generate_id()
        
        logger.info("Starting protein analysis", 
                   analysis_id=analysis_id, 
                   sequence_length=len(sequence),
                   analysis_type=analysis_type,
                   user_id=user_id)
        
        # Run analysis
        result = await protein_analyzer.analyze_sequence(
            sequence=sequence,
            analysis_type=analysis_type,
            include_mutations=include_mutations,
            include_evolution=include_evolution,
            analysis_id=analysis_id
        )
        
        # Add metadata
        result.update({
            "sequence_id": analysis_id,
            "timestamp": utc_now().isoformat(),
            "service": "esm3",
            "model_version": config.get("model_name", "esm3_sm_open_v1"),
            "user_id": user_id
        })
        
        logger.info("Protein analysis completed", 
                   analysis_id=analysis_id,
                   confidence_score=result.get("confidence_score", 0.0))
        
        return result
        
    except Exception as e:
        logger.error("Protein analysis failed", error=str(e), sequence_length=len(sequence))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch_analyze")
async def batch_analyze_proteins(
    sequences: List[str],
    analysis_type: str = "structure_prediction",
    user_id: Optional[str] = None,
    use_exostack: bool = False
):
    """Batch analyze multiple protein sequences"""
    if len(sequences) > 100:
        raise HTTPException(status_code=400, detail="Too many sequences (max 100)")
    
    if use_exostack and exostack_client:
        # Submit to ExoStack for distributed processing
        return await _submit_batch_to_exostack(sequences, analysis_type, user_id)
    
    # Process locally
    results = []
    for i, sequence in enumerate(sequences):
        try:
            result = await analyze_protein_sequence(
                sequence=sequence,
                analysis_type=analysis_type,
                user_id=user_id
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze sequence {i}", error=str(e))
            results.append({
                "sequence_id": f"batch_{i}",
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "batch_id": generate_id(),
        "total_sequences": len(sequences),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results
    }


@app.post("/predict_mutations")
async def predict_mutation_effects(
    sequence: str,
    mutations: List[Dict[str, Any]],
    user_id: Optional[str] = None
):
    """Predict effects of specific mutations on protein structure/function"""
    if not esm3_engine or not protein_analyzer:
        return await _mock_mutation_analysis(sequence, mutations)
    
    try:
        result = await protein_analyzer.predict_mutation_effects(
            sequence=sequence,
            mutations=mutations
        )
        
        result.update({
            "sequence_id": generate_id(),
            "timestamp": utc_now().isoformat(),
            "user_id": user_id
        })
        
        return result
        
    except Exception as e:
        logger.error("Mutation prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Mutation prediction failed: {str(e)}")


@app.post("/evolution_analysis")
async def analyze_evolutionary_pathways(
    sequence: str,
    target_properties: Dict[str, Any],
    user_id: Optional[str] = None
):
    """Analyze evolutionary pathways for protein optimization"""
    if not evolution_predictor:
        return await _mock_evolution_analysis(sequence, target_properties)
    
    try:
        result = await evolution_predictor.analyze_pathways(
            sequence=sequence,
            target_properties=target_properties
        )
        
        result.update({
            "sequence_id": generate_id(),
            "timestamp": utc_now().isoformat(),
            "user_id": user_id
        })
        
        return result
        
    except Exception as e:
        logger.error("Evolution analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Evolution analysis failed: {str(e)}")


@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded ESM3 model"""
    if not esm3_engine:
        return {
            "status": "not_loaded",
            "message": "ESM3 model not loaded (development mode)"
        }
    
    return {
        "model_name": config.get("model_name", "unknown"),
        "model_path": config.get("model_path", "unknown"),
        "parameters": config.get("parameters", {}),
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": esm3_engine.is_ready(),
        "supported_tasks": config.get("tasks", [])
    }


# Helper functions
def _validate_protein_sequence(sequence: str) -> bool:
    """Validate protein amino acid sequence"""
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa.upper() in valid_amino_acids for aa in sequence if aa.isalpha())


async def _mock_protein_analysis(sequence: str, analysis_type: str, include_mutations: bool, include_evolution: bool) -> Dict[str, Any]:
    """Mock protein analysis for development/testing"""
    await asyncio.sleep(0.1)  # Simulate processing time
    
    confidence_score = 0.85 + np.random.random() * 0.1
    
    # Create model metadata
    model_metadata = ModelMetadata(
        model_name=config.get("model_name", "esm3_mock"),
        model_version=MODEL_VERSIONS["esm3"],
        timestamp=utc_now(),
        processing_time=0.1
    )
    
    # Create explanation
    contributing_factors = [
        f"Protein sequence analysis of {len(sequence)} amino acids",
        "ESM3 evolutionary language model prediction",
        "Structural fold prediction using transformer architecture",
        "Conservation analysis across protein families"
    ]
    
    if include_mutations:
        contributing_factors.append("Mutation effect prediction using energy landscape modeling")
    
    if include_evolution:
        contributing_factors.append("Evolutionary pathway optimization analysis")
    
    explanation = ExplanationData(
        reason=f"The {analysis_type} analysis was performed using ESM3's deep learning model trained on protein sequences. The model analyzes amino acid patterns, secondary structure propensities, and evolutionary conservation to predict protein properties.",
        model_metadata=model_metadata,
        confidence_score=confidence_score,
        risk_level=RISK_LEVELS["LOW"] if confidence_score > 0.8 else RISK_LEVELS["MEDIUM"],
        biological_relevance="High correlation with experimental protein structure data. ESM3 predictions are based on evolutionary patterns learned from millions of protein sequences, providing biologically meaningful insights.",
        contributing_factors=contributing_factors,
        alternative_explanations=[
            "Experimental structure determination could provide more precise structural details",
            "Molecular dynamics simulations might reveal dynamic properties not captured in static predictions",
            "Laboratory mutagenesis experiments could validate predicted mutation effects"
        ]
    )
    
    return {
        "sequence_id": generate_id(),
        "sequence": sequence,
        "sequence_length": len(sequence),
        "predicted_structure": f"Mock structure prediction for {len(sequence)} residues",
        "confidence_score": confidence_score,
        "analysis_type": analysis_type,
        "mutation_analysis": {
            "hotspots": [10, 25, 67, 89] if include_mutations else None,
            "stability_effects": "Mock mutation stability analysis" if include_mutations else None
        } if include_mutations else None,
        "evolution_suggestion": {
            "pathways": ["pathway_1", "pathway_2"],
            "optimization_targets": ["stability", "activity"]
        } if include_evolution else None,
        "processing_time": 0.1,
        "timestamp": utc_now(),
        "status": "completed",
        "mock": True,
        # Enhanced fields
        "explanation": explanation.model_dump(),
        "model_metadata": model_metadata.model_dump()
    }


async def _mock_mutation_analysis(sequence: str, mutations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mock mutation analysis"""
    await asyncio.sleep(0.05)
    
    return {
        "sequence_id": generate_id(),
        "mutation_effects": [
            {
                "mutation": mut,
                "stability_change": np.random.uniform(-2.0, 2.0),
                "activity_change": np.random.uniform(-1.0, 1.0),
                "confidence": np.random.uniform(0.7, 0.95)
            }
            for mut in mutations
        ],
        "overall_assessment": "Mock mutation analysis completed",
        "mock": True
    }


async def _mock_evolution_analysis(sequence: str, target_properties: Dict[str, Any]) -> Dict[str, Any]:
    """Mock evolution analysis"""
    await asyncio.sleep(0.2)
    
    return {
        "sequence_id": generate_id(),
        "evolutionary_pathways": [
            {
                "pathway_id": f"pathway_{i}",
                "mutations": [f"A{10+i}V", f"L{20+i}F"],
                "predicted_improvement": np.random.uniform(0.1, 0.8),
                "confidence": np.random.uniform(0.6, 0.9)
            }
            for i in range(3)
        ],
        "optimization_suggestions": [
            "Increase thermal stability through core mutations",
            "Enhance catalytic activity via active site modifications",
            "Improve solubility through surface charge optimization"
        ],
        "mock": True
    }


async def _submit_batch_to_exostack(sequences: List[str], analysis_type: str, user_id: Optional[str]) -> Dict[str, Any]:
    """Submit batch job to ExoStack for distributed processing"""
    if not exostack_client:
        raise HTTPException(status_code=503, detail="ExoStack service not available")
    
    job_config = {
        "job_type": "esm3_batch_analysis",
        "sequences": sequences,
        "analysis_type": analysis_type,
        "user_id": user_id,
        "config": config
    }
    
    try:
        response = await exostack_client.post("/jobs/submit", json=job_config)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Failed to submit batch job to ExoStack", error=str(e))
        raise HTTPException(status_code=503, detail="Failed to submit batch job")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
