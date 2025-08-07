"""
EvoHuman.AI API Gateway
Main entry point for the platform API
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import httpx
from typing import Dict, Any, Optional, List
import structlog

from shared.models import User, UserCreate, AIServiceRequest, AIServiceResponse
from shared.utils import setup_logging, create_health_check_response, utc_now
from .auth import AuthManager
from .database import DatabaseManager


# Setup logging
logger = setup_logging("gateway")

# Global clients
auth_manager: Optional[AuthManager] = None
db_manager: Optional[DatabaseManager] = None
service_clients: Dict[str, httpx.AsyncClient] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global auth_manager, db_manager, service_clients
    
    logger.info("Starting EvoHuman.AI Gateway")
    
    # Initialize managers
    auth_manager = AuthManager()
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Initialize service clients
    services = {
        "bio-twin": "http://bio-twin:8000",
        "esm3": "http://esm3-service:8000", 
        "proteus": "http://proteus-service:8000",
        "aice": "http://aice-service:8000",
        "symbiotic": "http://symbiotic-service:8000",
        "exostack": "http://exostack-service:8000"
    }
    
    for name, url in services.items():
        service_clients[name] = httpx.AsyncClient(base_url=url, timeout=30.0)
    
    logger.info("Gateway initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Gateway")
    for client in service_clients.values():
        await client.aclose()
    await db_manager.close()


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI Gateway",
    description="API Gateway for the EvoHuman.AI Bio-Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Auth manager not initialized")
    
    user = await auth_manager.get_user_from_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {}
    
    # Check database
    try:
        if db_manager:
            await db_manager.health_check()
            dependencies["database"] = True
        else:
            dependencies["database"] = False
    except Exception:
        dependencies["database"] = False
    
    # Check services
    for name, client in service_clients.items():
        try:
            response = await client.get("/health", timeout=5.0)
            dependencies[f"service_{name}"] = response.status_code == 200
        except Exception:
            dependencies[f"service_{name}"] = False
    
    return create_health_check_response("gateway", dependencies)


# Authentication endpoints
@app.post("/auth/register", response_model=Dict[str, str])
async def register(user_data: UserCreate):
    """Register a new user"""
    if not auth_manager or not db_manager:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        user = await auth_manager.create_user(user_data)
        token = await auth_manager.create_access_token(user.id)
        
        logger.info("User registered", user_id=user.id, email=user.email)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user.id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=Dict[str, str])
async def login(email: str, password: str):
    """Login user"""
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Auth manager not initialized")
    
    user = await auth_manager.authenticate_user(email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    token = await auth_manager.create_access_token(user.id)
    
    logger.info("User logged in", user_id=user.id, email=user.email)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id
    }


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user


# Bio-Twin endpoints
@app.get("/bio-twin/snapshot")
async def get_bio_twin_snapshot(current_user: User = Depends(get_current_user)):
    """Get user's current bio-digital twin snapshot"""
    try:
        response = await service_clients["bio-twin"].get(
            f"/snapshot/{current_user.id}"
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("Bio-twin service error", error=str(e))
        raise HTTPException(status_code=503, detail="Bio-twin service unavailable")


@app.post("/bio-twin/analyze")
async def analyze_bio_twin(
    analysis_request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Request bio-twin analysis"""
    try:
        payload = {
            "user_id": current_user.id,
            **analysis_request
        }
        response = await service_clients["bio-twin"].post("/analyze", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("Bio-twin analysis error", error=str(e))
        raise HTTPException(status_code=503, detail="Analysis service unavailable")


# AI Service proxy endpoints
@app.post("/ai/{service_name}/process")
async def process_ai_request(
    service_name: str,
    request: AIServiceRequest,
    current_user: User = Depends(get_current_user)
):
    """Proxy AI service requests"""
    if service_name not in service_clients:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

    try:
        request.user_id = current_user.id
        response = await service_clients[service_name].post("/process", json=request.dict())
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("AI service error", service=service_name, error=str(e))
        raise HTTPException(status_code=503, detail=f"{service_name} service unavailable")


# ESM3 Protein Analysis Endpoints
@app.post("/esm3/analyze")
async def analyze_protein_sequence(
    sequence: str,
    analysis_type: str = "structure_prediction",
    include_mutations: bool = False,
    include_evolution: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Analyze protein sequence using ESM3 model"""
    try:
        payload = {
            "sequence": sequence,
            "analysis_type": analysis_type,
            "include_mutations": include_mutations,
            "include_evolution": include_evolution,
            "user_id": current_user.id
        }
        response = await service_clients["esm3"].post("/analyze", json=payload)
        response.raise_for_status()

        result = response.json()
        logger.info("ESM3 analysis completed",
                   user_id=current_user.id,
                   sequence_length=len(sequence),
                   confidence=result.get("confidence_score", 0.0))

        return result
    except httpx.HTTPError as e:
        logger.error("ESM3 analysis error", error=str(e))
        raise HTTPException(status_code=503, detail="ESM3 service unavailable")


@app.post("/esm3/batch_analyze")
async def batch_analyze_proteins(
    sequences: List[str],
    analysis_type: str = "structure_prediction",
    use_exostack: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Batch analyze multiple protein sequences"""
    if len(sequences) > 100:
        raise HTTPException(status_code=400, detail="Too many sequences (max 100)")

    try:
        payload = {
            "sequences": sequences,
            "analysis_type": analysis_type,
            "use_exostack": use_exostack,
            "user_id": current_user.id
        }
        response = await service_clients["esm3"].post("/batch_analyze", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("ESM3 batch analysis error", error=str(e))
        raise HTTPException(status_code=503, detail="ESM3 batch service unavailable")


@app.post("/esm3/predict_mutations")
async def predict_mutation_effects(
    sequence: str,
    mutations: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """Predict effects of specific mutations on protein"""
    try:
        payload = {
            "sequence": sequence,
            "mutations": mutations,
            "user_id": current_user.id
        }
        response = await service_clients["esm3"].post("/predict_mutations", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("ESM3 mutation prediction error", error=str(e))
        raise HTTPException(status_code=503, detail="ESM3 mutation service unavailable")


@app.post("/esm3/evolution_analysis")
async def analyze_evolutionary_pathways(
    sequence: str,
    target_properties: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Analyze evolutionary pathways for protein optimization"""
    try:
        payload = {
            "sequence": sequence,
            "target_properties": target_properties,
            "user_id": current_user.id
        }
        response = await service_clients["esm3"].post("/evolution_analysis", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("ESM3 evolution analysis error", error=str(e))
        raise HTTPException(status_code=503, detail="ESM3 evolution service unavailable")


@app.get("/esm3/model_info")
async def get_esm3_model_info(current_user: User = Depends(get_current_user)):
    """Get information about the ESM3 model"""
    try:
        response = await service_clients["esm3"].get("/model_info")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("ESM3 model info error", error=str(e))
        raise HTTPException(status_code=503, detail="ESM3 service unavailable")


if __name__ == "__main__":

# SymbioticAIS Endpoints
@app.post("/symbiotic/analyze")
async def analyze_symbiotic(
    input_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Analyze user input using SymbioticAIS"""
    try:
        payload = {
            "user_id": current_user.id,
            **input_data
        }
        response = await service_clients["symbiotic"].post("/symbiotic/analyze", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("SymbioticAIS analysis error", error=str(e))
        raise HTTPException(status_code=503, detail="SymbioticAIS service unavailable")

@app.get("/symbiotic/state/{user_id}")
async def get_symbiotic_state(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get user's SymbioticAIS learning state"""
    if user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's data")
    
    try:
        response = await service_clients["symbiotic"].get(f"/symbiotic/state/{user_id}")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("SymbioticAIS state error", error=str(e))
        raise HTTPException(status_code=503, detail="SymbioticAIS service unavailable")

@app.post("/symbiotic/train")
async def train_symbiotic(
    training_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Submit training feedback to SymbioticAIS"""
    try:
        payload = {
            "user_id": current_user.id,
            **training_data
        }
        response = await service_clients["symbiotic"].post("/symbiotic/train", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error("SymbioticAIS training error", error=str(e))
        raise HTTPException(status_code=503, detail="SymbioticAIS training service unavailable")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
