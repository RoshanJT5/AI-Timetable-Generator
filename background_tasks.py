import threading
import queue
import time
import logging
from password_security import hash_password
from models import get_next_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task queue for user account creation
_user_creation_queue = queue.Queue()

def queue_user_creation_task(user_data_list, default_password, user_type='faculty'):
    """
    Queue a list of users for background account creation (hashing).
    
    Args:
        user_data_list: List of dicts with {'username', 'name', 'email', 'entity_id'}
        default_password: Password to hash for all users
        user_type: 'faculty' or 'student'
    """
    task = {
        'user_data_list': user_data_list,
        'default_password': default_password,
        'user_type': user_type
    }
    _user_creation_queue.put(task)
    logger.info(f"[Background Tasks] Queued {len(user_data_list)} {user_type} accounts for creation")

def _worker_loop(get_db_session_func):
    """
    Background worker thread that processes the user creation queue.
    """
    logger.info("[Background Worker] Starting...")
    while True:
        try:
            # Wait for a task
            task = _user_creation_queue.get(timeout=5)
            if task is None: # Shutdown signal
                break
                
            user_data_list = task['user_data_list']
            default_password = task['default_password']
            user_type = task['user_type']
            collection_name = 'faculty' if user_type == 'faculty' else 'student'
            
            # Get a fresh DB session/object
            # In our case, db.session handles the batching if we use it, 
            # but we can also use db._db directly for speed.
            session = get_db_session_func()
            db_raw = session._db # Access the underlying pymongo db object
            
            logger.info(f"[Background Worker] Processing {len(user_data_list)} {user_type} accounts...")
            
            processed_count = 0
            for user_info in user_data_list:
                username = user_info.get('username')
                name = user_info.get('name')
                email = user_info.get('email')
                entity_id = user_info.get('entity_id') # The ID of the faculty/student record
                
                if not username:
                    continue
                    
                # 1. Check if user already exists
                existing_user = db_raw['user'].find_one({'username': username})
                if existing_user:
                    user_id = existing_user['id']
                    # Update role if needed (normalize faculty -> teacher)
                    role = 'teacher' if user_type == 'faculty' else 'student'
                    db_raw['user'].update_one({'id': user_id}, {'$set': {'role': role, 'name': name}})
                else:
                    # 2. Hash password and create new user
                    # Note: We use the helper to get next ID
                    user_id = get_next_id(db_raw, 'user')
                    password_hash = hash_password(default_password)
                    
                    user_doc = {
                        'id': user_id,
                        'username': username,
                        'email': email,
                        'role': 'teacher' if user_type == 'faculty' else 'student',
                        'name': name,
                        'password_hash': password_hash
                    }
                    db_raw['user'].insert_one(user_doc)
                
                # 3. Link user_id back to entity (faculty/student)
                if entity_id:
                    db_raw[collection_name].update_one({'id': entity_id}, {'$set': {'user_id': user_id}})
                
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"[Background Worker] Progress: {processed_count}/{len(user_data_list)}")
            
            logger.info(f"[Background Worker] Finished processing {processed_count} accounts")
            _user_creation_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"[Background Worker] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Don't mark task as done if it failed catastrophically? 
            # Or mark it done to prevent infinite loop.
            try:
                _user_creation_queue.task_done()
            except:
                pass

def start_background_worker(get_db_session_func):
    """
    Initialize and start the background worker thread.
    """
    worker_thread = threading.Thread(
        target=_worker_loop, 
        args=(get_db_session_func,),
        name="BackgroundUserCreator",
        daemon=True # Exit when main thread exits
    )
    worker_thread.start()
    return worker_thread
