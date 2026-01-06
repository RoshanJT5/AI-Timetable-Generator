from pymongo import MongoClient, ASCENDING, ReplaceOne, DeleteOne
from bson.objectid import ObjectId
from typing import Any, Dict, List
from flask import abort

# Import secure bcrypt-based password hashing
from password_security import hash_password, verify_password


class _Session:
    def __init__(self, db):
        self._db = db
        self._added = []
        self._deleted = []

    def add(self, obj):
        self._added.append(obj)

    def delete(self, obj):
        self._deleted.append(obj)

    def flush(self):
        ops = {}  # {collection_name: [operations]}

        # Process deletions
        for obj in list(self._deleted):
            try:
                coll_name = _get_collection_name(obj.__class__)
                if coll_name not in ops:
                    ops[coll_name] = []
                
                obj_id = getattr(obj, 'id', None)
                if obj_id is not None:
                    ops[coll_name].append(DeleteOne({'id': obj_id}))
                elif hasattr(obj, '_id'):
                    ops[coll_name].append(DeleteOne({'_id': obj._id}))
                else:
                    # Fallback for dict-based delete (cannot be bulked easily if filter is complex)
                    # Execute immediately
                    self._db[coll_name].delete_many(obj.to_dict())
            except Exception:
                pass

        # Process additions/updates
        for obj in list(self._added):
            coll_name = _get_collection_name(obj.__class__)
            if coll_name not in ops:
                ops[coll_name] = []
            
            # Ensure integer id sequence
            if getattr(obj, 'id', None) is None:
                obj.id = get_next_id(self._db, coll_name)
            
            data = obj.to_dict()
            # Remove _id from data to prevent WriteError (immutable field)
            data.pop('_id', None)
            ops[coll_name].append(ReplaceOne({'id': obj.id}, data, upsert=True))

        # Execute bulk writes
        for coll_name, operations in ops.items():
            if operations:
                try:
                    # ordered=False continues processing even if one fails
                    self._db[coll_name].bulk_write(operations, ordered=False)
                except Exception as e:
                    print(f"[MongoDB] Bulk write error in {coll_name}: {e}")

    def commit(self):
        # for simplicity, flush does the persistence
        self.flush()
        self._added.clear()
        self._deleted.clear()

    def rollback(self):
        # MongoDB doesn't support transactions in the same way as SQL
        # Just clear the pending operations
        self._added.clear()
        self._deleted.clear()


class _DB:
    def __init__(self):
        self.client: MongoClient | None = None
        self._db = None
        self.session = None
        self.engine = None

    def init_app(self, app):
        uri = app.config.get('MONGO_URI', 'mongodb://localhost:27017')
        dbname = app.config.get('MONGO_DBNAME', 'timetable')
        
        print(f"[MongoDB] Attempting to connect to database: {dbname}")
        print(f"[MongoDB] Connection URI: {uri[:20]}...") # Show only first part for security
        
        # Determine if we're using MongoDB Atlas (cloud) or local
        is_atlas = 'mongodb+srv://' in uri or 'mongodb.net' in uri
        is_local_dev = app.config.get('ENV') == 'development' or app.debug
        
        try:
            # Configure connection options with increased timeouts for VPN environments
            connection_options = {
                'serverSelectionTimeoutMS': 45000,  # 45 seconds for VPN latency
                'connectTimeoutMS': 45000,           # 45 seconds for initial connection
                'socketTimeoutMS': 45000,            # 45 seconds for socket operations
            }
            
            # Add TLS configuration for Atlas or if explicitly in URI
            if is_atlas:
                print("[MongoDB] Detected MongoDB Atlas connection")
                connection_options['tls'] = True
                
                # Always allow invalid certificates to handle VPN tunneling issues
                print("[MongoDB] Enabling TLS certificate bypass for VPN compatibility")
                connection_options['tlsAllowInvalidCertificates'] = True
            
            # Attempt connection
            self.client = MongoClient(uri, **connection_options)
            
            # Force connection test - this will raise an exception if connection fails
            print("[MongoDB] Testing connection with ping...")
            self.client.admin.command('ping')
            
            print(f"[MongoDB] Successfully connected to {dbname}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[MongoDB] Connection failed: {error_msg}")
            
            # Provide helpful error messages based on error type
            if 'ServerSelectionTimeoutError' in str(type(e)):
                print("\n" + "="*80)
                print("ERROR: Cannot connect to MongoDB")
                print("="*80)
                print("\nPossible causes:")
                print("1. MongoDB Atlas IP whitelist - Add your IP address in Atlas dashboard")
                print("2. Incorrect MONGO_URI in .env file")
                print("3. Network/firewall blocking connection")
                print("4. MongoDB Atlas cluster is paused or unavailable")
                print("\nCurrent URI (partial):", uri[:50] + "...")
                print("\nPlease check:")
                print("- .env file has correct MONGO_URI")
                print("- MongoDB Atlas Network Access allows your IP")
                print("- Internet connection is working")
                print("="*80 + "\n")
            elif 'Authentication failed' in error_msg:
                print("\n" + "="*80)
                print("ERROR: MongoDB Authentication Failed")
                print("="*80)
                print("\nThe username or password in MONGO_URI is incorrect.")
                print("Please check your .env file and ensure credentials are URL-encoded.")
                print("="*80 + "\n")
            elif 'SSL' in error_msg or 'TLS' in error_msg or 'handshake' in error_msg.lower():
                print("\n" + "="*80)
                print("ERROR: TLS/SSL Handshake Issue")
                print("="*80)
                print("\nTrying to reconnect with TLS certificate bypass...")
                print("="*80 + "\n")
                
                # Retry with TLS bypass for development
                try:
                    connection_options['tls'] = True
                    connection_options['tlsAllowInvalidCertificates'] = True
                    self.client = MongoClient(uri, **connection_options)
                    self.client.admin.command('ping')
                    print("[MongoDB] ✓ Connected with TLS bypass")
                except Exception as retry_error:
                    print(f"[MongoDB] ✗ Retry failed: {retry_error}")
                    print("\n" + "="*80)
                    print("CRITICAL: Cannot establish MongoDB connection")
                    print("="*80)
                    print("\nThe application cannot start without a database connection.")
                    print("Please check your network connection and MongoDB Atlas settings.")
                    print("="*80 + "\n")
                    raise SystemExit("MongoDB connection failed - application cannot start")
            else:
                print(f"\n[MongoDB] Unexpected error: {error_msg}")
                raise
        
        # Only set db and session if connection was successful
        if self.client is None:
            raise SystemExit("MongoDB client not initialized - application cannot start")
            
        self._db = self.client[dbname]
        self.session = _Session(self._db)
        self.engine = None

    def create_all(self):
        # Create indexes
        if self._db is not None:
            try:
                self._db['user'].create_index('username', unique=True)
                self._db['timetableentry'].create_index([('time_slot_id', 1)])
                self._db['timetableentry'].create_index([('faculty_id', 1)])
                self._db['timetableentry'].create_index([('room_id', 1)])
                self._db['timetableentry'].create_index([('student_group', 1)])
                self._db['timeslot'].create_index([('day', 1), ('period', 1)])
                print("[MongoDB] Indexes created successfully.")
            except Exception as e:
                print(f"[MongoDB] Index creation failed: {e}")

    def drop_all(self):
        # No-op
        pass


db = _DB()


def _get_collection_name(cls):
    return cls.__name__.lower()


def get_next_id(db, name: str) -> int:
    counters = db['__counters__']
    res = counters.find_one_and_update({'_id': name}, {'$inc': {'seq': 1}}, upsert=True, return_document=True)
    return int(res['seq'])


def get_next_id_batch(db, collection_name: str, count: int = 1000) -> int:
    """
    Get a starting ID for batch operations.
    Returns the starting ID; caller should increment in memory.
    
    Args:
        db: MongoDB database instance
        collection_name: Name of the collection
        count: Number of IDs to reserve (for logging purposes)
    
    Returns:
        Starting ID for the batch
    """
    # Find the maximum ID in the collection
    current_max = db[collection_name].find_one(
        sort=[('id', -1)],
        projection={'id': 1}
    )
    
    # If collection is empty, start at 1
    if current_max is None or current_max.get('id') is None:
        starting_id = 1
    else:
        starting_id = int(current_max['id']) + 1
    
    # Also update the counter to keep it in sync
    counters = db['__counters__']
    counters.find_one_and_update(
        {'_id': collection_name},
        {'$set': {'seq': starting_id + count - 1}},
        upsert=True
    )
    
    return starting_id


class ColumnRef:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def desc(self):
        # Return a special object or tuple that order_by can recognize
        return (self.name, -1)  # -1 is DESC in pymongo

    def asc(self):
        return (self.name, 1)   # 1 is ASC in pymongo


class ModelMeta(type):
    def __getattr__(cls, item):
        # Provide class-level helpers:
        # - `Model.query` should return a Query(model) so callers can do
        #    Model.query.count(), Model.query.first(), etc.
        # - other unknown attributes are treated as column references
        if item == 'query':
            return Query(cls)
        return ColumnRef(item)


class Query:
    def __init__(self, model_cls):
        self.model_cls = model_cls
        self._filter = {}
        self._sort = None
        self._projection = None

    def filter(self, *args):
        """
        Mock filter method to prevent crashes when SQLAlchemy-style filter() is called.
        NOTE: This custom ORM does not support complex SQL-like expressions (like db.func.lower).
        Using this method will fallback to returning all documents, allowing Python-side filtering.
        """
        # We cannot easily translate SQL expressions to MongoDB syntax here without a full expression parser.
        # So we simply return self (no-op) and let the caller handle filtering in Python if needed,
        # OR we log a warning.
        # Ideally, the caller should use filter_by for simple equality or do manual filtering.
        print("[WARNING] Query.filter() called but not fully supported in this lightweight MongoDB wrapper. returning unfiltered query.")
        return self

    def filter_by(self, **kwargs):
        self._filter.update(kwargs)
        return self
    
    def filter_by_departments(self, departments):
        """
        Filter by departments field (array field).
        Supports filtering by a single department or multiple departments.
        
        Args:
            departments: str, list, or None
                - If str: Find documents where departments array contains this value
                - If list: Find documents where departments array contains ANY of these values ($in)
                - If None: Find documents where departments is empty or None
        
        Usage:
            Faculty.query.filter_by_departments('Computer Science').all()
            Faculty.query.filter_by_departments(['CS', 'IT']).all()
        """
        if departments is None:
            # Find documents with empty or null departments
            self._filter['departments'] = {'$in': [[], None]}
        elif isinstance(departments, str):
            # Single department - find documents containing this value
            self._filter['departments'] = departments
        elif isinstance(departments, list):
            # Multiple departments - find documents containing ANY of these values
            if len(departments) == 0:
                self._filter['departments'] = {'$in': [[], None]}
            elif len(departments) == 1:
                self._filter['departments'] = departments[0]
            else:
                # Use $in operator for multiple values
                self._filter['departments'] = {'$in': departments}
        return self
    
    def options(self, projection):
        """
        Specify fields to include/exclude.
        Usage: Model.query.options({'field1': 1, 'field2': 1})
        """
        self._projection = projection
        return self

    def order_by(self, *attrs):
        # Support calling order_by(Model.field, Model.other) or order_by('field')
        sorts = []
        for attr in attrs:
            if isinstance(attr, tuple):
                # Handle (name, direction) from .desc()/.asc()
                sorts.append(attr)
            elif isinstance(attr, str):
                sorts.append((attr, ASCENDING))
            elif hasattr(attr, 'name'):
                sorts.append((attr.name, ASCENDING))
            else:
                # fallback: try to use str()
                sorts.append((str(attr), ASCENDING))
        self._sort = sorts if sorts else None
        return self

    def all(self):
        coll = db._db[_get_collection_name(self.model_cls)]
        cursor = coll.find(self._filter, self._projection)
        if self._sort:
            cursor = cursor.sort(self._sort)
        return [self.model_cls(**doc) for doc in cursor]

    def first(self):
        coll = db._db[_get_collection_name(self.model_cls)]
        doc = coll.find_one(self._filter, self._projection)
        if not doc:
            return None
        return self.model_cls(**doc)

    def count(self):
        coll = db._db[_get_collection_name(self.model_cls)]
        return coll.count_documents(self._filter)

    def delete(self, *args, **kwargs):
        # Accept extra SQLAlchemy-specific args (e.g., synchronize_session)
        coll = db._db[_get_collection_name(self.model_cls)]
        return coll.delete_many(self._filter)

    def get(self, id_value):
        coll = db._db[_get_collection_name(self.model_cls)]
        doc = coll.find_one({'id': id_value})
        if not doc:
            return None
        return self.model_cls(**doc)

    def get_or_404(self, id_value):
        obj = self.get(id_value)
        if obj is None:
            abort(404)
        return obj


class BaseModel(metaclass=ModelMeta):
    def __init__(self, **kwargs):
        # support both Mongo _id and integer id
        # Set all provided keys as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    # `query` is provided at the class level by `ModelMeta.__getattr__` so
    # callers can use `SomeModel.query.count()` or `SomeModel.query.first()`.

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        # Convert MongoDB ObjectId to string for JSON serialization
        if '_id' in d and d['_id'] is not None:
            d['_id'] = str(d['_id'])
        return d

    def _save(self, mongo_db):
        coll = mongo_db[_get_collection_name(self.__class__)]
        # ensure integer id sequence
        if getattr(self, 'id', None) is None:
            self.id = get_next_id(mongo_db, _get_collection_name(self.__class__))
        data = self.to_dict()
        # Remove _id from data to prevent WriteError (immutable field)
        # MongoDB's _id cannot be changed after document creation
        data.pop('_id', None)
        coll.replace_one({'id': self.id}, data, upsert=True)

    def save(self):
        """
        Save the current instance to the database.
        """
        # Use the global db instance
        self._save(db._db)


# --- Model definitions ---


class User(BaseModel):
    def set_password(self, password):
        """
        Hash and store password using bcrypt with 12 rounds.
        
        Args:
            password: Plaintext password to hash
            
        Security:
            - Uses bcrypt with 12 rounds (2^12 = 4096 iterations)
            - Automatically generates unique salt per password
            - Resistant to rainbow table and brute force attacks
        """
        self.password_hash = hash_password(password)

    def check_password(self, password):
        """
        Verify password against stored hash (supports both bcrypt and legacy Werkzeug).
        
        Args:
            password: Plaintext password to verify
            
        Returns:
            bool: True if password matches, False otherwise
            
        Security:
            - Supports bcrypt (new) and Werkzeug (legacy) hashes
            - Automatically migrates old hashes to bcrypt on successful login
            - Constant-time comparison (resistant to timing attacks)
            - Handles invalid hashes gracefully
        """
        stored_hash = getattr(self, 'password_hash', '')
        if not stored_hash:
            return False
        
        # Check if it's a bcrypt hash (starts with $2b$ or $2a$ or $2y$)
        if stored_hash.startswith('$2'):
            # New bcrypt hash
            is_valid = verify_password(password, stored_hash)
            return is_valid
        else:
            # Legacy Werkzeug hash - import here to avoid circular dependency
            from werkzeug.security import check_password_hash
            
            # Verify with Werkzeug
            is_valid = check_password_hash(stored_hash, password)
            
            if is_valid:
                # Migrate to bcrypt automatically on successful login
                print(f"[Security] Migrating password hash for user {getattr(self, 'username', 'unknown')} to bcrypt")
                self.set_password(password)
                # Note: Caller should commit this change to database
            
            return is_valid

    def __repr__(self):
        return f'<User {getattr(self, "username", None)} ({getattr(self, "role", None)})>'


class Course(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure default values for new fields if not present
        if not hasattr(self, 'program'): self.program = None
        if not hasattr(self, 'semester'): self.semester = None
        if not hasattr(self, 'branch'): self.branch = None
        if not hasattr(self, 'subject_type'): self.subject_type = None

    def to_dict(self):
        d = super().to_dict()
        d['program'] = getattr(self, 'program', None)
        d['semester'] = getattr(self, 'semester', None)
        d['branch'] = getattr(self, 'branch', None)
        d['subject_type'] = getattr(self, 'subject_type', None)
        return d

    def __repr__(self):
        return f'<Course {getattr(self, "code", None)} {getattr(self, "program", "")} Sem-{getattr(self, "semester", "")}>'


class Faculty(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure departments is always a list
        if not hasattr(self, 'departments'):
            self.departments = []
        elif isinstance(self.departments, str):
            # Handle case where departments might be stored as comma-separated string
            self.departments = [d.strip() for d in self.departments.split(',') if d.strip()]
        elif self.departments is None:
            self.departments = []
        elif not isinstance(self.departments, list):
            self.departments = [self.departments]
    
    def to_dict(self):
        d = super().to_dict()
        # Ensure departments is always stored as a list in the database
        d['departments'] = getattr(self, 'departments', [])
        return d
    
    def __repr__(self):
        dept_str = ', '.join(getattr(self, 'departments', [])) if getattr(self, 'departments', []) else 'No Dept'
        return f'<Faculty {getattr(self, "name", None)} ({dept_str})>'


class Room(BaseModel):
    def __repr__(self):
        return f'<Room {getattr(self, "name", None)}>'


class Student(BaseModel):
    def __init__(self, **kwargs):
        # Map current_semester to semester if provided
        if 'current_semester' in kwargs:
            kwargs['semester'] = kwargs.pop('current_semester')
        super().__init__(**kwargs)
        # Ensure default values for student selection context
        if not hasattr(self, 'program'): self.program = None
        if not hasattr(self, 'branch'): self.branch = None
        if not hasattr(self, 'semester'): self.semester = None
        if not hasattr(self, 'enrolled_courses'): self.enrolled_courses = []

    @property
    def current_semester(self):
        return getattr(self, 'semester', None)

    @current_semester.setter
    def current_semester(self, value):
        self.semester = value

    def to_dict(self):
        d = super().to_dict()
        d['program'] = getattr(self, 'program', None)
        d['branch'] = getattr(self, 'branch', None)
        d['semester'] = getattr(self, 'semester', None)
        d['current_semester'] = getattr(self, 'semester', None)
        d['enrolled_courses'] = getattr(self, 'enrolled_courses', [])
        return d

    def __repr__(self):
        return f'<Student {getattr(self, "student_id", None)}>'


class StudentGroup(BaseModel):
    def __init__(self, **kwargs):
        # Map current_semester to semester if provided in constructor
        if 'current_semester' in kwargs:
            kwargs['semester'] = kwargs.pop('current_semester')
        super().__init__(**kwargs)
        if not hasattr(self, 'name'): self.name = ""
        if not hasattr(self, 'description'): self.description = ""
        if not hasattr(self, 'program'): self.program = None
        if not hasattr(self, 'branch'): self.branch = None
        if not hasattr(self, 'semester'): self.semester = None
        if not hasattr(self, 'total_students'): self.total_students = 0
        if not hasattr(self, 'batches'): self.batches = None

    @property
    def current_semester(self):
        """Alias for semester for backward compatibility"""
        return getattr(self, 'semester', None)

    @current_semester.setter
    def current_semester(self, value):
        self.semester = value

    def to_dict(self):
        d = super().to_dict()
        d['program'] = getattr(self, 'program', None)
        d['branch'] = getattr(self, 'branch', None)
        d['semester'] = getattr(self, 'semester', None)
        # Include alias for frontend compatibility if needed
        d['current_semester'] = getattr(self, 'semester', None)
        return d

    def __repr__(self):
        return f'<StudentGroup {getattr(self, "name", None)} {getattr(self, "program", "")}-{getattr(self, "branch", "")} Sem-{getattr(self, "semester", "")}>'


class Branch(BaseModel):
    """Represents an academic branch/specialization (e.g., Computer Science in B.Tech)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'program'): self.program = None
        if not hasattr(self, 'name'): self.name = None
        if not hasattr(self, 'code'): self.code = None
        if not hasattr(self, 'hod_name'): self.hod_name = None
        if not hasattr(self, 'duration_years'): self.duration_years = 4
        if not hasattr(self, 'total_semesters'): self.total_semesters = 8

    def to_dict(self):
        d = super().to_dict()
        d['program'] = getattr(self, 'program', None)
        d['name'] = getattr(self, 'name', None)
        d['code'] = getattr(self, 'code', None)
        d['hod_name'] = getattr(self, 'hod_name', None)
        d['duration_years'] = getattr(self, 'duration_years', 4)
        d['total_semesters'] = getattr(self, 'total_semesters', 8)
        return d

    def __repr__(self):
        return f'<Branch {getattr(self, "name", None)} ({getattr(self, "program", "")})>'


class PeriodConfig(BaseModel):
    def __repr__(self):
        return f'<PeriodConfig {getattr(self, "periods_per_day", None)} periods, {getattr(self, "period_duration_minutes", None)} min>'


class BreakConfig(BaseModel):
    def __repr__(self):
        return f'<BreakConfig {getattr(self, "break_name", None)}>'


class TimeSlot(BaseModel):
    def __repr__(self):
        return f'<TimeSlot {getattr(self, "day", None)} P{getattr(self, "period", None)}>'


class TimetableEntry(BaseModel):
    def __repr__(self):
        return f'<TimetableEntry {getattr(self, "course_id", None)}-{getattr(self, "faculty_id", None)}-{getattr(self, "room_id", None)}-{getattr(self, "student_group", None)}>'
    pass


class TimetableHistory(BaseModel):
    """
    Stores snapshots of generated timetables for historical viewing.
    
    Fields:
        - name: User-friendly name for this timetable (e.g., "CSE Sem-3 Spring 2024")
        - generated_at: Timestamp when this timetable was generated
        - generated_by: User ID who generated this timetable
        - filters: JSON string of filters used (program, branch, semester)
        - entries: JSON string containing the complete timetable data
        - stats: JSON string with generation statistics (total entries, warnings, etc.)
        - is_active: Boolean indicating if this is the currently active timetable
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'is_active'):
            self.is_active = False
        if not hasattr(self, 'generated_at'):
            from datetime import datetime
            self.generated_at = datetime.utcnow().isoformat()
    
    def __repr__(self):
        return f'<TimetableHistory {getattr(self, "name", None)} ({getattr(self, "generated_at", None)})'

