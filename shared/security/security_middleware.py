"""
Security Middleware for EvoHuman.AI Services
Comprehensive security enhancements including authentication, authorization, and input validation
"""
import jwt
import bcrypt
import secrets
import hashlib
import re
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog
from pydantic import BaseModel, validator
import aioredis
import time
from functools import wraps
import ipaddress


logger = structlog.get_logger("security")


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_origins: List[str] = None
    require_https: bool = True
    session_timeout: int = 3600  # 1 hour


class User(BaseModel):
    """User model for authentication"""
    id: str
    email: str
    name: str
    roles: List[str] = []
    permissions: List[str] = []
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthToken(BaseModel):
    """Authentication token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    roles: List[str] = []


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""
    
    def __init__(self, app, config: SecurityConfig, redis_client: aioredis.Redis):
        super().__init__(app)
        self.config = config
        self.redis_client = redis_client
        self.security = HTTPBearer()
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # Rate limiting storage
        self.rate_limit_storage: Dict[str, List[float]] = {}
        
        # Blocked IPs (in production, this would be in Redis/database)
        self.blocked_ips: Set[str] = set()
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """Main security middleware dispatch"""
        start_time = time.time()
        
        try:
            # Security checks
            await self._check_request_size(request)
            await self._check_rate_limiting(request)
            await self._check_ip_blocking(request)
            await self._validate_request_headers(request)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log security event
            processing_time = time.time() - start_time
            await self._log_security_event(request, response, processing_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Security middleware error", error=str(e))
            raise HTTPException(status_code=500, detail="Internal security error")
    
    async def _check_request_size(self, request: Request):
        """Check request size limits"""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            logger.warning("Request size exceeded", 
                         size=content_length, 
                         max_size=self.config.max_request_size,
                         client_ip=request.client.host)
            raise HTTPException(status_code=413, detail="Request entity too large")
    
    async def _check_rate_limiting(self, request: Request):
        """Check rate limiting"""
        client_ip = request.client.host
        current_time = time.time()
        
        # Get rate limit key
        rate_key = f"rate_limit:{client_ip}"
        
        try:
            # Get current request count from Redis
            current_count = await self.redis_client.get(rate_key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= self.config.rate_limit_requests:
                logger.warning("Rate limit exceeded", 
                             client_ip=client_ip,
                             current_count=current_count,
                             limit=self.config.rate_limit_requests)
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_key)
            pipe.expire(rate_key, self.config.rate_limit_window)
            await pipe.execute()
            
        except aioredis.RedisError:
            # Fallback to in-memory rate limiting
            if client_ip not in self.rate_limit_storage:
                self.rate_limit_storage[client_ip] = []
            
            # Clean old requests
            window_start = current_time - self.config.rate_limit_window
            self.rate_limit_storage[client_ip] = [
                req_time for req_time in self.rate_limit_storage[client_ip]
                if req_time > window_start
            ]
            
            # Check rate limit
            if len(self.rate_limit_storage[client_ip]) >= self.config.rate_limit_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Add current request
            self.rate_limit_storage[client_ip].append(current_time)
    
    async def _check_ip_blocking(self, request: Request):
        """Check if IP is blocked"""
        client_ip = request.client.host
        
        if client_ip in self.blocked_ips:
            logger.warning("Blocked IP attempted access", client_ip=client_ip)
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check Redis for blocked IPs
        try:
            is_blocked = await self.redis_client.sismember("blocked_ips", client_ip)
            if is_blocked:
                self.blocked_ips.add(client_ip)  # Cache locally
                raise HTTPException(status_code=403, detail="Access denied")
        except aioredis.RedisError:
            pass  # Continue if Redis is unavailable
    
    async def _validate_request_headers(self, request: Request):
        """Validate request headers for security"""
        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-originating-ip"
        ]
        
        for header in suspicious_headers:
            value = request.headers.get(header)
            if value and not self._is_valid_ip_list(value):
                logger.warning("Suspicious header detected", 
                             header=header, 
                             value=value,
                             client_ip=request.client.host)
        
        # Validate User-Agent
        user_agent = request.headers.get("user-agent", "")
        if self._is_suspicious_user_agent(user_agent):
            logger.warning("Suspicious user agent", 
                         user_agent=user_agent,
                         client_ip=request.client.host)
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    async def _log_security_event(self, request: Request, response: Response, processing_time: float):
        """Log security-related events"""
        event_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "status_code": response.status_code,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in Redis for security analysis
        try:
            key = f"security_log:{int(time.time())}"
            await self.redis_client.setex(key, 86400, str(event_data))  # 24 hours
        except aioredis.RedisError:
            pass  # Continue if Redis is unavailable
        
        # Log suspicious activities
        if response.status_code >= 400:
            logger.warning("Security event", **event_data)
    
    def _is_valid_ip_list(self, ip_string: str) -> bool:
        """Validate IP address list"""
        try:
            ips = [ip.strip() for ip in ip_string.split(",")]
            for ip in ips:
                ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agents"""
        suspicious_patterns = [
            r"bot",
            r"crawler",
            r"spider",
            r"scraper",
            r"curl",
            r"wget",
            r"python",
            r"java",
            r"go-http-client"
        ]
        
        user_agent_lower = user_agent.lower()
        return any(re.search(pattern, user_agent_lower) for pattern in suspicious_patterns)


class AuthenticationService:
    """Authentication and authorization service"""
    
    def __init__(self, config: SecurityConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis_client = redis_client
        self.security = HTTPBearer()
    
    async def create_user(self, email: str, password: str, name: str, roles: List[str] = None) -> User:
        """Create new user"""
        # Validate email format
        if not self._is_valid_email(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Check if user exists
        if await self._user_exists(email):
            raise HTTPException(status_code=409, detail="User already exists")
        
        # Validate password strength
        if not self._is_strong_password(password):
            raise HTTPException(status_code=400, detail="Password does not meet requirements")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=self.config.bcrypt_rounds))
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        user = User(
            id=user_id,
            email=email,
            name=name,
            roles=roles or ["user"],
            created_at=datetime.utcnow()
        )
        
        # Store user data
        await self._store_user(user, password_hash)
        
        logger.info("User created", user_id=user_id, email=email)
        return user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        try:
            # Get user data
            user_data = await self.redis_client.hget("users", email)
            if not user_data:
                return None
            
            user_info = eval(user_data)  # In production, use proper JSON serialization
            stored_hash = user_info["password_hash"]
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return None
            
            # Create user object
            user = User(**user_info["user_data"])
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self._update_user_login(user)
            
            logger.info("User authenticated", user_id=user.id, email=email)
            return user
            
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return None
    
    async def create_token(self, user: User) -> AuthToken:
        """Create JWT token for user"""
        payload = {
            "user_id": user.id,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for token revocation
        }
        
        token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        
        # Store token for revocation checking
        await self._store_token(payload["jti"], user.id, self.config.jwt_expiration_hours * 3600)
        
        return AuthToken(
            access_token=token,
            expires_in=self.config.jwt_expiration_hours * 3600,
            user_id=user.id,
            roles=user.roles
        )
    
    async def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user"""
        try:
            # Decode token
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and await self._is_token_revoked(jti):
                return None
            
            # Get user
            user_data = await self.redis_client.hget("users", payload["email"])
            if not user_data:
                return None
            
            user_info = eval(user_data)
            user = User(**user_info["user_data"])
            
            return user if user.is_active else None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token used")
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None
    
    async def revoke_token(self, token: str):
        """Revoke JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            jti = payload.get("jti")
            if jti:
                await self.redis_client.setex(f"revoked_token:{jti}", 86400, "1")  # 24 hours
                logger.info("Token revoked", jti=jti)
        except Exception as e:
            logger.error("Token revocation error", error=str(e))
    
    def require_auth(self, required_roles: List[str] = None, required_permissions: List[str] = None):
        """Decorator to require authentication"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    raise HTTPException(status_code=500, detail="Request object not found")
                
                # Get token from header
                auth_header = request.headers.get("authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                token = auth_header.split(" ")[1]
                user = await self.verify_token(token)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid or expired token")
                
                # Check roles
                if required_roles and not any(role in user.roles for role in required_roles):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Check permissions
                if required_permissions and not any(perm in user.permissions for perm in required_permissions):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Add user to request state
                request.state.user = user
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def _user_exists(self, email: str) -> bool:
        """Check if user exists"""
        return await self.redis_client.hexists("users", email)
    
    async def _store_user(self, user: User, password_hash: bytes):
        """Store user data"""
        user_data = {
            "user_data": user.dict(),
            "password_hash": password_hash
        }
        await self.redis_client.hset("users", user.email, str(user_data))
    
    async def _update_user_login(self, user: User):
        """Update user last login time"""
        user_data = await self.redis_client.hget("users", user.email)
        if user_data:
            user_info = eval(user_data)
            user_info["user_data"]["last_login"] = user.last_login.isoformat()
            await self.redis_client.hset("users", user.email, str(user_info))
    
    async def _store_token(self, jti: str, user_id: str, expiration: int):
        """Store token for revocation checking"""
        await self.redis_client.setex(f"token:{jti}", expiration, user_id)
    
    async def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        return await self.redis_client.exists(f"revoked_token:{jti}")
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_strong_password(self, password: str) -> bool:
        """Check password strength"""
        if len(password) < 8:
            return False
        
        # Check for uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special


class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_protein_sequence(sequence: str) -> bool:
        """Validate protein sequence"""
        if not sequence:
            return False
        
        # Valid amino acid codes
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        return all(aa.upper() in valid_aa for aa in sequence)
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate numeric value is within range"""
        return min_val <= value <= max_val
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = re.sub(r'\.\.', '', sanitized)  # Remove parent directory references
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')
        
        return sanitized
