"""
Authentication manager for EvoHuman.AI Gateway
"""
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import structlog

from shared.models import User, UserCreate
from shared.utils import generate_id, utc_now
from .database import DatabaseManager


logger = structlog.get_logger("auth")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthManager:
    """Handle user authentication and authorization"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return await self.db.get_user_by_email(email)
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await self.db.get_user_by_id(user_id)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Hash password
        hashed_password = self.get_password_hash(user_data.password)
        
        # Create user
        user = User(
            id=generate_id(),
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            role=user_data.role,
            is_active=True,
            created_at=utc_now(),
            updated_at=utc_now()
        )
        
        # Save to database
        await self.db.create_user(user, hashed_password)
        
        logger.info("User created", user_id=user.id, email=user.email)
        return user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = await self.get_user_by_email(email)
        if not user:
            return None
        
        # Get hashed password from database
        hashed_password = await self.db.get_user_password_hash(user.id)
        if not hashed_password:
            return None
        
        if not self.verify_password(password, hashed_password):
            return None
        
        return user
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
        except jwt.PyJWTError:
            return None
        
        user = await self.get_user_by_id(user_id)
        return user
    
    async def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an access token"""
        user = await self.get_user_from_token(token)
        if not user:
            return None
        
        return self.create_access_token(user.id)
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)"""
        # TODO: Implement token blacklist in Redis
        return True
    
    def is_admin(self, user: User) -> bool:
        """Check if user is admin"""
        return user.role == "admin"
    
    def is_researcher(self, user: User) -> bool:
        """Check if user is researcher or admin"""
        return user.role in ["researcher", "admin"]
