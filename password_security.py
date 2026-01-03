"""
Secure Password Hashing Module using bcrypt
Provides thread-safe password hashing with configurable work factor
"""
import bcrypt
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import threading

# Thread pool for offloading heavy bcrypt operations
# This prevents blocking the main thread during password hashing
_hash_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bcrypt_hash")

# Bcrypt configuration
BCRYPT_ROUNDS = 12  # Recommended: 12-14 rounds (2^12 = 4096 iterations)
# Each additional round doubles the time
# 12 rounds ≈ 250ms on modern hardware (good balance of security vs UX)


def hash_password(plaintext: str, rounds: int = BCRYPT_ROUNDS) -> str:
    """
    Hash a plaintext password using bcrypt with configurable work factor.
    
    Args:
        plaintext: The plaintext password to hash
        rounds: Number of bcrypt rounds (default: 12)
                Higher = more secure but slower
                Recommended range: 10-14
    
    Returns:
        str: The bcrypt hash as a UTF-8 string
        
    Example:
        >>> hashed = hash_password("my_secure_password")
        >>> print(hashed)
        '$2b$12$...'
    
    Security Notes:
        - Uses bcrypt's built-in salt generation
        - Salt is automatically included in the hash
        - Each hash is unique even for the same password
        - Resistant to rainbow table attacks
    """
    if not plaintext:
        raise ValueError("Password cannot be empty")
    
    # Generate salt with specified rounds
    salt = bcrypt.gensalt(rounds=rounds)
    
    # Hash the password
    hashed = bcrypt.hashpw(plaintext.encode('utf-8'), salt)
    
    # Return as UTF-8 string for MongoDB storage
    return hashed.decode('utf-8')


def verify_password(plaintext: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.
    
    Args:
        plaintext: The plaintext password to verify
        hashed: The bcrypt hash to check against
    
    Returns:
        bool: True if password matches, False otherwise
        
    Example:
        >>> hashed = hash_password("my_password")
        >>> verify_password("my_password", hashed)
        True
        >>> verify_password("wrong_password", hashed)
        False
    
    Security Notes:
        - Constant-time comparison (resistant to timing attacks)
        - Automatically extracts salt and rounds from hash
        - Works with hashes created with different round counts
    """
    if not plaintext or not hashed:
        return False
    
    try:
        # bcrypt.checkpw handles the comparison securely
        return bcrypt.checkpw(
            plaintext.encode('utf-8'),
            hashed.encode('utf-8')
        )
    except (ValueError, TypeError):
        # Invalid hash format or encoding issue
        return False


def hash_password_async(plaintext: str, rounds: int = BCRYPT_ROUNDS):
    """
    Hash password asynchronously in a thread pool.
    Useful for async frameworks or to avoid blocking the main thread.
    
    Args:
        plaintext: The plaintext password to hash
        rounds: Number of bcrypt rounds (default: 12)
    
    Returns:
        concurrent.futures.Future: Future object containing the hash
        
    Example:
        >>> future = hash_password_async("my_password")
        >>> hashed = future.result()  # Blocks until complete
    """
    return _hash_executor.submit(hash_password, plaintext, rounds)


def verify_password_async(plaintext: str, hashed: str):
    """
    Verify password asynchronously in a thread pool.
    
    Args:
        plaintext: The plaintext password to verify
        hashed: The bcrypt hash to check against
    
    Returns:
        concurrent.futures.Future: Future object containing bool result
        
    Example:
        >>> future = verify_password_async("my_password", hashed)
        >>> is_valid = future.result()  # Blocks until complete
    """
    return _hash_executor.submit(verify_password, plaintext, hashed)


def threaded_hash(func):
    """
    Decorator to automatically offload password hashing to thread pool.
    Use this for Flask routes that hash passwords to avoid blocking.
    
    Example:
        @app.route('/register', methods=['POST'])
        @threaded_hash
        def register():
            password = request.json['password']
            hashed = hash_password(password)  # Runs in thread pool
            # ... rest of registration logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're already in a worker thread
        if threading.current_thread().name.startswith('bcrypt_hash'):
            # Already in thread pool, execute directly
            return func(*args, **kwargs)
        else:
            # Submit to thread pool
            future = _hash_executor.submit(func, *args, **kwargs)
            return future.result()
    return wrapper


def get_hash_info(hashed: str) -> dict:
    """
    Extract information from a bcrypt hash.
    Useful for debugging or migration purposes.
    
    Args:
        hashed: The bcrypt hash string
    
    Returns:
        dict: Information about the hash including algorithm, rounds, salt
        
    Example:
        >>> info = get_hash_info('$2b$12$...')
        >>> print(info)
        {'algorithm': '2b', 'rounds': 12, 'salt': '...', 'hash': '...'}
    """
    try:
        parts = hashed.split('$')
        if len(parts) >= 4:
            return {
                'algorithm': parts[1],  # Usually '2b' for bcrypt
                'rounds': int(parts[2]),
                'salt': parts[3][:22] if len(parts[3]) >= 22 else parts[3],
                'hash': parts[3][22:] if len(parts[3]) > 22 else '',
                'full_hash': hashed
            }
    except (IndexError, ValueError):
        pass
    
    return {'error': 'Invalid hash format'}


def needs_rehash(hashed: str, target_rounds: int = BCRYPT_ROUNDS) -> bool:
    """
    Check if a password hash needs to be rehashed with more rounds.
    Useful for gradually increasing security over time.
    
    Args:
        hashed: The bcrypt hash to check
        target_rounds: The desired number of rounds
    
    Returns:
        bool: True if hash should be regenerated with more rounds
        
    Example:
        >>> old_hash = hash_password("password", rounds=10)
        >>> needs_rehash(old_hash, target_rounds=12)
        True
    """
    info = get_hash_info(hashed)
    if 'rounds' in info:
        return info['rounds'] < target_rounds
    return True  # If we can't parse it, assume it needs rehashing


# Cleanup function (call on application shutdown)
def shutdown_hash_executor():
    """
    Gracefully shutdown the thread pool executor.
    Call this when your application is shutting down.
    """
    _hash_executor.shutdown(wait=True)
