"""
Database manager for EvoHuman.AI Gateway
"""
import os
from typing import Optional, List
import asyncpg
import structlog
from datetime import datetime

from shared.models import User
from shared.utils import utc_now


logger = structlog.get_logger("database")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://evohuman:evohuman@localhost:5432/evohuman")


class DatabaseManager:
    """Handle database operations"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
            await self.create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")
    
    async def health_check(self):
        """Check database health"""
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    
    async def create_tables(self):
        """Create database tables"""
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(36) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    full_name VARCHAR(255),
                    role VARCHAR(20) DEFAULT 'user',
                    is_active BOOLEAN DEFAULT true,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Bio-twin snapshots table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bio_twin_snapshots (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    state VARCHAR(20) NOT NULL,
                    metrics JSONB,
                    insights JSONB,
                    recommendations JSONB,
                    confidence_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Evolution goals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_goals (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    category VARCHAR(50) NOT NULL,
                    target_metric VARCHAR(100) NOT NULL,
                    current_value FLOAT NOT NULL,
                    target_value FLOAT NOT NULL,
                    timeline_days INTEGER NOT NULL,
                    priority INTEGER DEFAULT 5,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # AI service requests table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_service_requests (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    service_type VARCHAR(50) NOT NULL,
                    input_data JSONB,
                    parameters JSONB,
                    status VARCHAR(20) DEFAULT 'pending',
                    result JSONB,
                    error TEXT,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Feedback loops table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_loops (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) REFERENCES users(id),
                    loop_type VARCHAR(50) NOT NULL,
                    input_data JSONB,
                    human_feedback JSONB,
                    ai_response JSONB,
                    improvement_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Database tables created/verified")
    
    async def create_user(self, user: User, password_hash: str):
        """Create a new user"""
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (id, email, username, full_name, role, is_active, password_hash, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, user.id, user.email, user.username, user.full_name, user.role, 
                user.is_active, password_hash, user.created_at, user.updated_at)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        if not self.pool:
            return None
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, username, full_name, role, is_active, created_at, updated_at
                FROM users WHERE email = $1
            """, email)
            
            if row:
                return User(**dict(row))
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        if not self.pool:
            return None
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, username, full_name, role, is_active, created_at, updated_at
                FROM users WHERE id = $1
            """, user_id)
            
            if row:
                return User(**dict(row))
            return None
    
    async def get_user_password_hash(self, user_id: str) -> Optional[str]:
        """Get user's password hash"""
        if not self.pool:
            return None
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT password_hash FROM users WHERE id = $1
            """, user_id)
    
    async def update_user(self, user_id: str, **updates):
        """Update user information"""
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        if not updates:
            return
        
        # Build dynamic update query
        set_clauses = []
        values = []
        param_count = 1
        
        for field, value in updates.items():
            set_clauses.append(f"{field} = ${param_count}")
            values.append(value)
            param_count += 1
        
        values.append(utc_now())  # updated_at
        values.append(user_id)    # WHERE condition
        
        query = f"""
            UPDATE users 
            SET {', '.join(set_clauses)}, updated_at = ${param_count}
            WHERE id = ${param_count + 1}
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(query, *values)
