"""
Notification Queue Manager for user messaging system
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import redis
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

from shared.models import NotificationMessage, NotificationQueue
from shared.constants import NOTIFICATION_PRIORITY, CACHE_KEYS
from shared.utils import generate_id, utc_now


class NotificationQueueManager:
    """Manages user notification queues and message delivery"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.logger = structlog.get_logger("notification_queue")
        self.redis_client = redis_client or redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
    
    async def add_notification(
        self,
        user_id: str,
        message_type: str,
        title: str,
        content: str,
        priority: str = "medium",
        expires_at: Optional[datetime] = None,
        actions: Optional[List[Dict[str, str]]] = None
    ) -> NotificationMessage:
        """Add a new notification to user's queue"""
        
        # Validate priority
        if priority not in NOTIFICATION_PRIORITY.values():
            priority = NOTIFICATION_PRIORITY["MEDIUM"]
        
        # Set default expiration (7 days for non-urgent messages)
        if expires_at is None:
            if priority == NOTIFICATION_PRIORITY["URGENT"]:
                expires_at = utc_now() + timedelta(hours=24)
            elif priority == NOTIFICATION_PRIORITY["HIGH"]:
                expires_at = utc_now() + timedelta(days=3)
            else:
                expires_at = utc_now() + timedelta(days=7)
        
        notification = NotificationMessage(
            id=generate_id(),
            user_id=user_id,
            message_type=message_type,
            title=title,
            content=content,
            priority=priority,
            created_at=utc_now(),
            expires_at=expires_at,
            read=False,
            actions=actions
        )
        
        # Add to user's notification queue
        queue_key = CACHE_KEYS["USER_NOTIFICATIONS"].format(user_id=user_id)
        
        # Get current queue
        current_queue = await self._get_user_queue(user_id)
        
        # Add new notification (priority ordering)
        current_queue.messages.append(notification)
        current_queue.messages.sort(
            key=lambda n: (
                self._priority_weight(n.priority),
                n.created_at
            ),
            reverse=True
        )
        
        # Keep only last 50 notifications
        current_queue.messages = current_queue.messages[:50]
        
        # Update queue metadata
        current_queue.last_updated = utc_now()
        current_queue.unread_count = len([n for n in current_queue.messages if not n.read])
        
        # Save updated queue
        await self._save_user_queue(user_id, current_queue)
        
        self.logger.info(
            "Notification added",
            user_id=user_id,
            message_type=message_type,
            priority=priority,
            notification_id=notification.id
        )
        
        return notification
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 20,
        unread_only: bool = False,
        message_types: Optional[List[str]] = None
    ) -> List[NotificationMessage]:
        """Get user's notifications with filtering"""
        
        queue = await self._get_user_queue(user_id)
        
        # Filter messages
        messages = queue.messages
        
        if unread_only:
            messages = [m for m in messages if not m.read]
        
        if message_types:
            messages = [m for m in messages if m.message_type in message_types]
        
        # Remove expired messages
        current_time = utc_now()
        messages = [m for m in messages if m.expires_at > current_time]
        
        return messages[:limit]
    
    async def mark_notification_read(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """Mark a notification as read"""
        
        queue = await self._get_user_queue(user_id)
        
        # Find and mark notification as read
        for notification in queue.messages:
            if notification.id == notification_id:
                notification.read = True
                
                # Update queue metadata
                queue.last_updated = utc_now()
                queue.unread_count = len([n for n in queue.messages if not n.read])
                
                # Save updated queue
                await self._save_user_queue(user_id, queue)
                
                self.logger.info(
                    "Notification marked as read",
                    user_id=user_id,
                    notification_id=notification_id
                )
                return True
        
        return False
    
    async def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user"""
        
        queue = await self._get_user_queue(user_id)
        
        # Mark all as read
        marked_count = 0
        for notification in queue.messages:
            if not notification.read:
                notification.read = True
                marked_count += 1
        
        if marked_count > 0:
            # Update queue metadata
            queue.last_updated = utc_now()
            queue.unread_count = 0
            
            # Save updated queue
            await self._save_user_queue(user_id, queue)
        
        self.logger.info(
            "All notifications marked as read",
            user_id=user_id,
            marked_count=marked_count
        )
        
        return marked_count
    
    async def delete_notification(
        self,
        user_id: str,
        notification_id: str
    ) -> bool:
        """Delete a specific notification"""
        
        queue = await self._get_user_queue(user_id)
        
        # Find and remove notification
        original_count = len(queue.messages)
        queue.messages = [n for n in queue.messages if n.id != notification_id]
        
        if len(queue.messages) < original_count:
            # Update queue metadata
            queue.last_updated = utc_now()
            queue.unread_count = len([n for n in queue.messages if not n.read])
            
            # Save updated queue
            await self._save_user_queue(user_id, queue)
            
            self.logger.info(
                "Notification deleted",
                user_id=user_id,
                notification_id=notification_id
            )
            return True
        
        return False
    
    async def cleanup_expired_notifications(self, user_id: str) -> int:
        """Remove expired notifications from user's queue"""
        
        queue = await self._get_user_queue(user_id)
        current_time = utc_now()
        
        # Filter out expired notifications
        original_count = len(queue.messages)
        queue.messages = [
            n for n in queue.messages 
            if n.expires_at > current_time
        ]
        
        removed_count = original_count - len(queue.messages)
        
        if removed_count > 0:
            # Update queue metadata
            queue.last_updated = utc_now()
            queue.unread_count = len([n for n in queue.messages if not n.read])
            
            # Save updated queue
            await self._save_user_queue(user_id, queue)
            
            self.logger.info(
                "Expired notifications cleaned up",
                user_id=user_id,
                removed_count=removed_count
            )
        
        return removed_count
    
    async def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications"""
        
        queue = await self._get_user_queue(user_id)
        
        # Clean up expired first
        await self.cleanup_expired_notifications(user_id)
        
        return queue.unread_count
    
    async def add_insight_notification(
        self,
        user_id: str,
        insight_type: str,
        insight_content: str,
        confidence_score: float,
        priority: str = "medium"
    ) -> NotificationMessage:
        """Add notification for new AI insight"""
        
        title = f"New {insight_type.replace('_', ' ').title()} Insight"
        
        # Truncate long content for notification
        content = insight_content[:200] + "..." if len(insight_content) > 200 else insight_content
        
        # Add confidence context
        content += f" (Confidence: {confidence_score:.0%})"
        
        actions = [
            {"type": "view", "label": "View Details"},
            {"type": "feedback", "label": "Provide Feedback"}
        ]
        
        return await self.add_notification(
            user_id=user_id,
            message_type="insight",
            title=title,
            content=content,
            priority=priority,
            actions=actions
        )
    
    async def add_reminder_notification(
        self,
        user_id: str,
        reminder_type: str,
        reminder_content: str,
        due_date: Optional[datetime] = None
    ) -> NotificationMessage:
        """Add reminder notification"""
        
        title = f"Reminder: {reminder_type.replace('_', ' ').title()}"
        
        if due_date:
            content = f"{reminder_content} (Due: {due_date.strftime('%Y-%m-%d %H:%M')})"
            expires_at = due_date + timedelta(hours=12)  # Keep for 12 hours after due
        else:
            content = reminder_content
            expires_at = None
        
        actions = [
            {"type": "complete", "label": "Mark Complete"},
            {"type": "snooze", "label": "Remind Later"}
        ]
        
        return await self.add_notification(
            user_id=user_id,
            message_type="reminder",
            title=title,
            content=content,
            priority="medium",
            expires_at=expires_at,
            actions=actions
        )
    
    async def add_achievement_notification(
        self,
        user_id: str,
        achievement_type: str,
        achievement_description: str,
        milestone_data: Dict[str, Any]
    ) -> NotificationMessage:
        """Add achievement/milestone notification"""
        
        title = f"Achievement Unlocked: {achievement_type.replace('_', ' ').title()}"
        
        content = achievement_description
        if milestone_data.get('progress'):
            content += f" Progress: {milestone_data['progress']}"
        
        actions = [
            {"type": "celebrate", "label": "Celebrate!"},
            {"type": "share", "label": "Share Achievement"}
        ]
        
        return await self.add_notification(
            user_id=user_id,
            message_type="achievement",
            title=title,
            content=content,
            priority="high",  # Achievements deserve attention
            actions=actions
        )
    
    async def add_alert_notification(
        self,
        user_id: str,
        alert_type: str,
        alert_message: str,
        severity: str = "medium"
    ) -> NotificationMessage:
        """Add alert notification for important events"""
        
        title = f"Alert: {alert_type.replace('_', ' ').title()}"
        
        # Map severity to priority
        priority_map = {
            "low": NOTIFICATION_PRIORITY["LOW"],
            "medium": NOTIFICATION_PRIORITY["MEDIUM"],
            "high": NOTIFICATION_PRIORITY["HIGH"],
            "critical": NOTIFICATION_PRIORITY["URGENT"]
        }
        
        priority = priority_map.get(severity, NOTIFICATION_PRIORITY["MEDIUM"])
        
        actions = [
            {"type": "acknowledge", "label": "Acknowledge"},
            {"type": "learn_more", "label": "Learn More"}
        ]
        
        return await self.add_notification(
            user_id=user_id,
            message_type="alert",
            title=title,
            content=alert_message,
            priority=priority,
            actions=actions
        )
    
    async def _get_user_queue(self, user_id: str) -> NotificationQueue:
        """Get user's notification queue from Redis"""
        
        queue_key = CACHE_KEYS["USER_NOTIFICATIONS"].format(user_id=user_id)
        
        try:
            queue_data = self.redis_client.get(queue_key)
            if queue_data:
                queue_dict = json.loads(queue_data)
                
                # Parse datetime strings back to datetime objects
                if queue_dict.get('last_updated'):
                    queue_dict['last_updated'] = datetime.fromisoformat(queue_dict['last_updated'])
                
                # Parse messages
                messages = []
                for msg_data in queue_dict.get('messages', []):
                    msg_data['created_at'] = datetime.fromisoformat(msg_data['created_at'])
                    if msg_data.get('expires_at'):
                        msg_data['expires_at'] = datetime.fromisoformat(msg_data['expires_at'])
                    messages.append(NotificationMessage(**msg_data))
                
                queue_dict['messages'] = messages
                return NotificationQueue(**queue_dict)
        except Exception as e:
            self.logger.error("Failed to get user queue", user_id=user_id, error=str(e))
        
        # Return empty queue if not found or error
        return NotificationQueue(
            user_id=user_id,
            messages=[],
            last_updated=utc_now(),
            unread_count=0
        )
    
    async def _save_user_queue(self, user_id: str, queue: NotificationQueue):
        """Save user's notification queue to Redis"""
        
        queue_key = CACHE_KEYS["USER_NOTIFICATIONS"].format(user_id=user_id)
        
        try:
            # Convert to dict for JSON serialization
            queue_data = queue.model_dump()
            
            # Convert datetime objects to ISO strings
            queue_data['last_updated'] = queue_data['last_updated'].isoformat()
            
            for msg_data in queue_data['messages']:
                msg_data['created_at'] = msg_data['created_at'].isoformat()
                if msg_data.get('expires_at'):
                    msg_data['expires_at'] = msg_data['expires_at'].isoformat()
            
            # Save with 30 day expiration
            self.redis_client.setex(
                queue_key,
                2592000,  # 30 days
                json.dumps(queue_data)
            )
            
        except Exception as e:
            self.logger.error("Failed to save user queue", user_id=user_id, error=str(e))
    
    def _priority_weight(self, priority: str) -> int:
        """Convert priority to numerical weight for sorting"""
        
        weights = {
            NOTIFICATION_PRIORITY["URGENT"]: 4,
            NOTIFICATION_PRIORITY["HIGH"]: 3,
            NOTIFICATION_PRIORITY["MEDIUM"]: 2,
            NOTIFICATION_PRIORITY["LOW"]: 1
        }
        
        return weights.get(priority, 2)
