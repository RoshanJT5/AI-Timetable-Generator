import os
import json
import hashlib
from functools import wraps
from flask import request, jsonify, Response
import redis

# Initialize Redis
# Use DB 1 for cache (0 is typically used for Celery)
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
redis_client = None
redis_available = False

# Check Redis availability
try:
    _temp_client = redis.from_url(redis_url, socket_connect_timeout=1)
    _temp_client.ping()
    redis_client = _temp_client
    redis_available = True
    print("[Cache] Redis connected successfully")
except Exception as e:
    print(f"[Cache] Redis not available: {e}")
    print("[Cache] Running in no-cache mode")
    redis_available = False

def get_cache_version(prefix):
    """Get the current version for a cache prefix."""
    if not redis_available:
        return "1"
    try:
        v = redis_client.get(f"version:{prefix}")
        if not v:
            return "1"
        return v.decode('utf-8')
    except Exception:
        return "1"

def generate_cache_key(prefix, *args, **kwargs):
    """Generate a consistent cache key based on request path, args, user, and version."""
    version = get_cache_version(prefix)
    key_parts = [prefix, version, request.path]
    
    # Add query parameters
    if request.args:
        key_parts.append(json.dumps(dict(request.args), sort_keys=True))
    
    # Add function arguments
    for arg in args:
        key_parts.append(str(arg))
    if kwargs:
        key_parts.append(json.dumps(kwargs, sort_keys=True))
        
    # Create hash
    key_str = "|".join(key_parts)
    return f"cache:{hashlib.sha256(key_str.encode()).hexdigest()}"


def invalidate_cache(prefix):
    """Invalidate all cache keys with a specific prefix by incrementing version."""
    if not redis_available:
        return  # Silently skip if Redis is not available
    try:
        redis_client.incr(f"version:{prefix}")
        print(f"[Cache] Invalidated prefix: {prefix}")
    except Exception as e:
        print(f"[Cache] Invalidation failed: {e}")

def cache_response(ttl=300, prefix='view'):
    """
    Decorator to cache Flask endpoints.
    Supports JSON responses and rendered templates (strings).
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip caching if Redis is not available
            if not redis_available:
                return f(*args, **kwargs)
            
            # Skip caching for non-GET methods usually, but user might want it
            if request.method != 'GET':
                return f(*args, **kwargs)

            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    content = data['content']
                    content_type = data['content_type']
                    
                    if content_type == 'application/json':
                        return jsonify(content)
                    else:
                        return Response(content, mimetype=content_type)
            except Exception as e:
                # If Redis fails, just run the function
                print(f"Cache read error: {e}")

            # Execute function
            response = f(*args, **kwargs)
            
            # Extract content to cache
            try:
                if hasattr(response, 'get_json') and response.get_json():
                    content = response.get_json()
                    content_type = 'application/json'
                elif hasattr(response, 'data'):
                    content = response.data.decode('utf-8')
                    content_type = response.mimetype
                elif isinstance(response, (dict, list)):
                    content = response
                    content_type = 'application/json'
                    response = jsonify(content) # Ensure it's a response object
                elif isinstance(response, str):
                    content = response
                    content_type = 'text/html'
                else:
                    # Can't easily cache other types
                    return response

                # Store in Redis
                cache_payload = {
                    'content': content,
                    'content_type': content_type
                }
                redis_client.setex(cache_key, ttl, json.dumps(cache_payload))
            except Exception as e:
                print(f"Cache write error: {e}")
                
            return response
        return decorated_function
    return decorator

