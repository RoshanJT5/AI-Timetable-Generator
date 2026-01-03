import os
import jwt
import datetime
import uuid
from functools import wraps
from flask import request, jsonify, current_app, g, redirect, url_for
import redis

# Initialize Redis for Token Blocklist
# Using DB 2 for Auth to keep it separate from Cache (DB 1) and Celery (DB 0)
redis_url = os.getenv('REDIS_AUTH_URL', 'redis://localhost:6379/2')
redis_client = None
redis_available = False

# Check Redis availability for token revocation
try:
    _temp_client = redis.from_url(redis_url, socket_connect_timeout=1)
    _temp_client.ping()
    redis_client = _temp_client
    redis_available = True
    print("[Auth] Redis connected - Token revocation enabled")
except Exception as e:
    print(f"[Auth] Redis not available: {e}")
    print("[Auth] Running with stateless JWT (no revocation)")
    redis_available = False

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_tokens(user_id, role):
    """Generate Access and Refresh tokens"""
    access_id = str(uuid.uuid4())
    refresh_id = str(uuid.uuid4())
    
    access_payload = {
        'sub': str(user_id),
        'role': role,
        'type': 'access',
        'jti': access_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        'iat': datetime.datetime.utcnow()
    }
    
    refresh_payload = {
        'sub': str(user_id),
        'role': role,
        'type': 'refresh',
        'jti': refresh_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        'iat': datetime.datetime.utcnow()
    }
    
    secret = current_app.config['SECRET_KEY']
    access_token = jwt.encode(access_payload, secret, algorithm='HS256')
    refresh_token = jwt.encode(refresh_payload, secret, algorithm='HS256')
    
    return access_token, refresh_token

def decode_token(token):
    """Decode and verify token"""
    try:
        secret = current_app.config['SECRET_KEY']
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        
        # Check if revoked (only if Redis is available)
        if redis_available and is_token_revoked(payload['jti']):
            return None
            
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def revoke_token(jti, expires_in):
    """Add token JTI to blocklist (only if Redis is available)"""
    if not redis_available:
        return  # Silently skip if Redis is not available
    try:
        redis_client.setex(f"revoked:{jti}", expires_in, 'true')
    except Exception as e:
        print(f"[Auth] Token revocation failed: {e}")

def is_token_revoked(jti):
    """Check if token is in blocklist (only if Redis is available)"""
    if not redis_available:
        return False  # Cannot revoke without Redis, so assume not revoked
    try:
        return redis_client.exists(f"revoked:{jti}")
    except Exception as e:
        print(f"[Auth] Token revocation check failed: {e}")
        return False  # On error, assume not revoked to allow access

def jwt_required(f):
    """Decorator to protect routes with JWT (Cookie based)"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('access_token')
        
        if not token:
            # If no access token, try refresh token flow automatically?
            # For now, just redirect to login or return 401
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'message': 'Missing access token'}), 401
            return redirect(url_for('login'))
        
        payload = decode_token(token)
        
        if not payload or payload['type'] != 'access':
            # Token invalid or expired
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'message': 'Invalid or expired token'}), 401
            return redirect(url_for('login'))
            
        # Store user info in g
        g.user_id = payload['sub']
        g.user_role = payload['role']
        
        return f(*args, **kwargs)
    return decorated

def admin_required_jwt(f):
    """Decorator for admin only routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('access_token')
        if not token:
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'message': 'Missing token'}), 401
            return redirect(url_for('login'))
            
        payload = decode_token(token)
        if not payload or payload['role'] != 'admin':
            if request.headers.get('Accept') == 'application/json':
                return jsonify({'message': 'Admin privilege required'}), 403
            return redirect(url_for('login'))
            
        g.user_id = payload['sub']
        g.user_role = payload['role']
        
        return f(*args, **kwargs)
    return decorated
