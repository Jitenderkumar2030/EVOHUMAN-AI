"""
EvoHuman.AI Notification Service
Manages user notifications and message queues
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import structlog

from shared.models import NotificationMessage, NotificationQueue
from shared.utils import setup_logging, create_health_check_response, utc_now
from .queue_manager import NotificationQueueManager


# Setup logging
logger = setup_logging("notification-service")

# Global components
queue_manager: Optional[NotificationQueueManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global queue_manager
    
    logger.info("Starting Notification Service")
    
    # Initialize queue manager
    queue_manager = NotificationQueueManager()
    
    logger.info("Notification Service initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Notification Service")


# Create FastAPI app
app = FastAPI(
    title="EvoHuman.AI Notification Service",
    description="User notification and message queue management",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dependencies = {
        "queue_manager": queue_manager is not None,
        "redis": True  # Will be tested by queue_manager
    }
    
    # Test Redis connection through queue manager
    if queue_manager:
        try:
            test_queue = await queue_manager._get_user_queue("health_check_test")
            dependencies["redis"] = test_queue is not None
        except Exception:
            dependencies["redis"] = False
    
    return create_health_check_response("notification-service", dependencies)


@app.get("/notifications/{user_id}")
async def get_user_notifications(
    user_id: str,
    limit: int = 20,
    unread_only: bool = False,
    message_types: Optional[str] = None
) -> List[NotificationMessage]:
    """Get user's notifications with filtering"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        # Parse message types filter
        message_type_list = message_types.split(",") if message_types else None
        
        notifications = await queue_manager.get_user_notifications(
            user_id=user_id,
            limit=limit,
            unread_only=unread_only,
            message_types=message_type_list
        )
        
        return notifications
        
    except Exception as e:
        logger.error("Failed to get user notifications", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")


@app.post("/notifications/{user_id}")
async def add_notification(
    user_id: str,
    notification_data: Dict[str, Any]
) -> NotificationMessage:
    """Add a new notification for user"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        # Extract notification details
        message_type = notification_data.get("message_type", "general")
        title = notification_data.get("title", "")
        content = notification_data.get("content", "")
        priority = notification_data.get("priority", "medium")
        actions = notification_data.get("actions")
        expires_at = notification_data.get("expires_at")
        
        if not title or not content:
            raise HTTPException(
                status_code=400, 
                detail="title and content are required"
            )
        
        # Parse expires_at if provided
        if expires_at:
            from datetime import datetime
            expires_at = datetime.fromisoformat(expires_at)
        
        notification = await queue_manager.add_notification(
            user_id=user_id,
            message_type=message_type,
            title=title,
            content=content,
            priority=priority,
            expires_at=expires_at,
            actions=actions
        )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add notification", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add notification")


@app.put("/notifications/{user_id}/{notification_id}/read")
async def mark_notification_read(
    user_id: str,
    notification_id: str
):
    """Mark a notification as read"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        success = await queue_manager.mark_notification_read(user_id, notification_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"status": "success", "message": "Notification marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to mark notification as read", 
                    user_id=user_id, notification_id=notification_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")


@app.put("/notifications/{user_id}/mark-all-read")
async def mark_all_notifications_read(user_id: str):
    """Mark all notifications as read for a user"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        marked_count = await queue_manager.mark_all_read(user_id)
        
        return {
            "status": "success", 
            "message": f"Marked {marked_count} notifications as read"
        }
        
    except Exception as e:
        logger.error("Failed to mark all notifications as read", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to mark all notifications as read")


@app.delete("/notifications/{user_id}/{notification_id}")
async def delete_notification(
    user_id: str,
    notification_id: str
):
    """Delete a specific notification"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        success = await queue_manager.delete_notification(user_id, notification_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"status": "success", "message": "Notification deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete notification", 
                    user_id=user_id, notification_id=notification_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete notification")


@app.get("/notifications/{user_id}/unread-count")
async def get_unread_count(user_id: str):
    """Get count of unread notifications"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        count = await queue_manager.get_unread_count(user_id)
        
        return {
            "user_id": user_id,
            "unread_count": count,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get unread count", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get unread count")


@app.post("/notifications/{user_id}/cleanup")
async def cleanup_expired_notifications(user_id: str):
    """Clean up expired notifications for user"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        removed_count = await queue_manager.cleanup_expired_notifications(user_id)
        
        return {
            "status": "success", 
            "message": f"Cleaned up {removed_count} expired notifications"
        }
        
    except Exception as e:
        logger.error("Failed to cleanup expired notifications", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cleanup notifications")


# Specialized notification endpoints

@app.post("/notifications/{user_id}/insight")
async def add_insight_notification(
    user_id: str,
    insight_data: Dict[str, Any]
):
    """Add notification for new AI insight"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        insight_type = insight_data.get("insight_type", "general")
        insight_content = insight_data.get("insight_content", "")
        confidence_score = insight_data.get("confidence_score", 0.5)
        priority = insight_data.get("priority", "medium")
        
        if not insight_content:
            raise HTTPException(status_code=400, detail="insight_content is required")
        
        notification = await queue_manager.add_insight_notification(
            user_id=user_id,
            insight_type=insight_type,
            insight_content=insight_content,
            confidence_score=confidence_score,
            priority=priority
        )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add insight notification", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add insight notification")


@app.post("/notifications/{user_id}/reminder")
async def add_reminder_notification(
    user_id: str,
    reminder_data: Dict[str, Any]
):
    """Add reminder notification"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        reminder_type = reminder_data.get("reminder_type", "general")
        reminder_content = reminder_data.get("reminder_content", "")
        due_date_str = reminder_data.get("due_date")
        
        if not reminder_content:
            raise HTTPException(status_code=400, detail="reminder_content is required")
        
        # Parse due_date if provided
        due_date = None
        if due_date_str:
            from datetime import datetime
            due_date = datetime.fromisoformat(due_date_str)
        
        notification = await queue_manager.add_reminder_notification(
            user_id=user_id,
            reminder_type=reminder_type,
            reminder_content=reminder_content,
            due_date=due_date
        )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add reminder notification", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add reminder notification")


@app.post("/notifications/{user_id}/achievement")
async def add_achievement_notification(
    user_id: str,
    achievement_data: Dict[str, Any]
):
    """Add achievement notification"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        achievement_type = achievement_data.get("achievement_type", "milestone")
        achievement_description = achievement_data.get("achievement_description", "")
        milestone_data = achievement_data.get("milestone_data", {})
        
        if not achievement_description:
            raise HTTPException(status_code=400, detail="achievement_description is required")
        
        notification = await queue_manager.add_achievement_notification(
            user_id=user_id,
            achievement_type=achievement_type,
            achievement_description=achievement_description,
            milestone_data=milestone_data
        )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add achievement notification", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add achievement notification")


@app.post("/notifications/{user_id}/alert")
async def add_alert_notification(
    user_id: str,
    alert_data: Dict[str, Any]
):
    """Add alert notification"""
    if not queue_manager:
        raise HTTPException(status_code=500, detail="Queue manager not initialized")
    
    try:
        alert_type = alert_data.get("alert_type", "general")
        alert_message = alert_data.get("alert_message", "")
        severity = alert_data.get("severity", "medium")
        
        if not alert_message:
            raise HTTPException(status_code=400, detail="alert_message is required")
        
        notification = await queue_manager.add_alert_notification(
            user_id=user_id,
            alert_type=alert_type,
            alert_message=alert_message,
            severity=severity
        )
        
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add alert notification", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add alert notification")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
