"""
EvoHuman.AI Evolution API Routes
Handles evolution and longevity-focused endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
import structlog
from ..auth import get_current_user
from ...core.agent_orchestrator import (
    get_orchestrator,
    EvolutionRequest,
    EvolutionResponse
)
from ...shared.models import User

# Configure logging
logger = structlog.get_logger("evolution_api")

router = APIRouter(prefix="/evolve", tags=["evolution"])

@router.post("/", response_model=EvolutionResponse)
async def evolve_human(
    request: EvolutionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate personalized longevity evolution insights and recommendations
    """
    try:
        # Validate user matches request
        if request.user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this user's evolution data"
            )

        # Get orchestrator instance
        orchestrator = await get_orchestrator()

        # Generate evolution insights
        response = await orchestrator.orchestrate_evolution(request)

        logger.info(
            "Evolution insights generated",
            user_id=current_user.id,
            longevity_score=response.longevity_score
        )

        return response

    except Exception as e:
        logger.error(
            "Evolution analysis failed",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate evolution insights"
        )

@router.get("/status/{user_id}")
async def get_evolution_status(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get current evolution and training status for a user
    """
    try:
        if user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this user's status"
            )

        orchestrator = await get_orchestrator()
        cached = await orchestrator.get_cached_insights(user_id)

        if not cached:
            return {
                "status": "no_data",
                "message": "No evolution data available"
            }

        return {
            "status": "active",
            "last_update": cached["updated_at"],
            "longevity_score": cached["longevity_score"]
        }

    except Exception as e:
        logger.error(
            "Status check failed",
            user_id=user_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to check evolution status"
        )