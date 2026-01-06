# Load environment variables FIRST before any other imports
import os
from dotenv import load_dotenv
load_dotenv()

# Now import everything else
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, session, flash, abort, g, make_response
from celery import Celery
from cache import cache_response, invalidate_cache
from auth_jwt import create_tokens, decode_token, revoke_token, is_token_revoked
from models import db, Course, Faculty, Room, Student, TimeSlot, TimetableEntry, User, PeriodConfig, BreakConfig, StudentGroup, Branch, TimetableHistory, get_next_id
from scheduler import TimetableGenerator
from functools import wraps
import time
try:
    from pyinstrument import Profiler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    Profiler = None
import csv
import io
from datetime import datetime
import json
import secrets
import math

# Optional pandas for Excel support
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from csv_processor import process_upload_stream, get_missing_columns, process_departments_field, normalize_string
from pymongo.errors import DuplicateKeyError as IntegrityError
import warnings

def time_to_minutes(time_str):
    """Convert time string (HH:MM) to minutes since midnight"""
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

def minutes_to_time(minutes):
    """Convert minutes since midnight to time string (HH:MM)"""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def hydrate_default_faculty_values():
    """
    One-time migration to set default values for faculty fields.
    Only runs if Faculty collection is empty (first startup).
    For existing databases, defaults are handled in the Faculty model __init__.
    """
    # Only run if database is empty (first-time setup)
    faculty_count = Faculty.query.count()
    if faculty_count == 0:
        # No faculty yet, nothing to hydrate
        return
    
    # Check if any faculty needs hydration (legacy data)
    # Only check first faculty to avoid full table scan
    sample_faculty = Faculty.query.first()
    if sample_faculty and (
        sample_faculty.min_hours_per_week is not None and
        sample_faculty.max_hours_per_week is not None and
        sample_faculty.availability
    ):
        # Sample faculty has all defaults, assume rest are fine
        return
    
    # Legacy data detected, hydrate all faculty
    updated = False
    for faculty in Faculty.query.all():
        if faculty.min_hours_per_week is None:
            faculty.min_hours_per_week = 4
            updated = True
        if faculty.max_hours_per_week is None:
            faculty.max_hours_per_week = 16
            updated = True
        if not faculty.availability:
            faculty.availability = "{}"
            updated = True
    
    if updated:
        db.session.commit()
        print(f"[MIGRATION] Hydrated default values for {faculty_count} faculty members")

def validate_faculty_availability(availability_data):
    """
    Validates that faculty is available for at least 70% of total periods.
    Returns (is_valid, error_message, availability_percentage)
    """
    # Get period configuration
    period_config = PeriodConfig.query.first()
    if not period_config:
        # Use default values if no config exists
        periods_per_day = 8
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    else:
        periods_per_day = period_config.periods_per_day
        days_of_week = [d.strip() for d in period_config.days_of_week.split(',')]
    
    total_periods = periods_per_day * len(days_of_week)
    min_required_periods = math.ceil(total_periods * 0.70)  # 70% threshold
    
    # Parse availability data
    if isinstance(availability_data, str):
        try:
            availability_data = json.loads(availability_data)
        except:
            availability_data = {}
    
    # Count available periods
    available_periods = 0
    for day in days_of_week:
        if day in availability_data:
            available_periods += len(availability_data[day])
    
    availability_percentage = (available_periods / total_periods * 100) if total_periods > 0 else 0
    
    if available_periods < min_required_periods:
        error_msg = f"Faculty must be available for at least 70% of periods. Currently available for {available_periods}/{total_periods} periods ({availability_percentage:.1f}%). Minimum required: {min_required_periods} periods."
        return False, error_msg, availability_percentage
    
    return True, None, availability_percentage

def create_faculty_profile(payload):
    username = payload.get('username', '').strip() or None
    raw_password = payload.get('password', '').strip()
    generated_password = None

    user = None
    if username:
        existing_faculty = Faculty.query.filter_by(username=username).first()
        if existing_faculty:
            raise ValueError('Username already assigned to another faculty profile.')
        user = User.query.filter_by(username=username).first()
        if user:
            user.role = 'teacher'
            user.name = payload['name']
            if raw_password:
                user.set_password(raw_password)
        else:
            password_to_use = raw_password or secrets.token_urlsafe(8)
            generated_password = None if raw_password else password_to_use
            email = payload.get('email') or f'{username}@faculty.local'
            email = email.strip()
            existing_email_user = User.query.filter_by(email=email).first()
            if existing_email_user:
                email = f'{username}+{secrets.token_hex(3)}@faculty.local'
            user = User(username=username, email=email, role='teacher', name=payload['name'])
            user.set_password(password_to_use)
            db.session.add(user)
            db.session.flush()

    availability_payload = payload.get('availability', '{}')
    # Sanitize availability: allow dict/list -> JSON; else ensure string JSON object
    if isinstance(availability_payload, (dict, list)):
        availability_payload = json.dumps(availability_payload)
    elif not isinstance(availability_payload, str):
        availability_payload = '{}'

    expertise_payload = normalize_comma_list(payload.get('expertise', []))
    
    # Handle departments field using csv_processor helper for consistency
    departments_payload = payload.get('departments', [])
    departments_list = process_departments_field(departments_payload)

    faculty = Faculty(
        name=payload['name'],
        email=payload.get('email', ''),
        expertise=','.join(expertise_payload),
        availability=availability_payload,
        username=username,
        min_hours_per_week=int(payload.get('min_hours_per_week', 4)),
        max_hours_per_week=int(payload.get('max_hours_per_week', 16)),
        user_id=user.id if user else None,
        departments=departments_list
    )
    db.session.add(faculty)
    return faculty, generated_password

def parse_int(value, default=0):
    try:
        return int(value) if value not in (None, '', 'nan') else default
    except (TypeError, ValueError):
        return default

def normalize_comma_list(value):
    if not value or value == 'nan':
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(',') if item.strip()]


# Navigation flow for guided setup
def get_next_page(current_page):
    """Get the next page URL in the navigation flow for admin guided setup"""
    navigation_map = {
        'courses': '/faculty',
        'faculty': '/rooms',
        'rooms': '/students',
        'students': '/student-groups',
        'student-groups': '/settings',
        'settings': '/timetable',
        'timetable': None  # Last step
    }
    return navigation_map.get(current_page)

def get_progress_steps(current_page):
    """Get list of all steps with current step marked"""
    steps = [
        {'name': 'courses', 'title': 'Courses', 'icon': 'book'},
        {'name': 'faculty', 'title': 'Faculty', 'icon': 'person-badge'},
        {'name': 'rooms', 'title': 'Rooms', 'icon': 'building'},
        {'name': 'students', 'title': 'Students', 'icon': 'people'},
        {'name': 'student-groups', 'title': 'Groups', 'icon': 'people-fill'},
        {'name': 'settings', 'title': 'Settings', 'icon': 'gear'},
        {'name': 'timetable', 'title': 'Timetable', 'icon': 'calendar-week'}
    ]
    
    current_index = next((i for i, s in enumerate(steps) if s['name'] == current_page), -1)
    
    for i, step in enumerate(steps):
        if i < current_index:
            step['status'] = 'completed'
        elif i == current_index:
            step['status'] = 'active'
        else:
            step['status'] = 'pending'
    
    return steps





def generate_time_slots():
    """Generate time slots based on PeriodConfig and BreakConfig"""
    # Clear existing time slots efficiently
    TimeSlot.query.delete()
    
    # Get period configuration
    period_config = PeriodConfig.query.first()
    if not period_config:
        # Use defaults if no config exists
        period_config = PeriodConfig(
            periods_per_day=8,
            period_duration_minutes=60,
            day_start_time='09:00',
            days_of_week='Monday,Tuesday,Wednesday,Thursday,Friday'
        )
        db.session.add(period_config)
        db.session.commit()
    
    # Get break configurations, ordered by after_period
    breaks = BreakConfig.query.order_by(BreakConfig.after_period).all()
    break_map = {br.after_period: br for br in breaks}
    
    days = [d.strip() for d in period_config.days_of_week.split(',')]
    start_minutes = time_to_minutes(period_config.day_start_time)
    period_duration = period_config.period_duration_minutes
    
    # Prepare list for bulk insert
    slots_data = []
    
    for day in days:
        current_time = start_minutes
        for period_num in range(1, period_config.periods_per_day + 1):
            # Calculate period start and end
            period_start = current_time
            period_end = period_start + period_duration
            
            # Create time slot dict
            slots_data.append({
                'day': day,
                'period': period_num,
                'start_time': minutes_to_time(period_start),
                'end_time': minutes_to_time(period_end)
            })
            
            # Move to next period start (after this period ends)
            current_time = period_end
            
            # Check if there's a break after this period
            if period_num in break_map:
                break_config = break_map[period_num]
                current_time += break_config.duration_minutes
    
    if slots_data:
        # Bulk allocate IDs
        count = len(slots_data)
        counters = db._db['__counters__']
        res = counters.find_one_and_update(
            {'_id': 'timeslot'}, 
            {'$inc': {'seq': count}}, 
            upsert=True, 
            return_document=True
        )
        end_seq = int(res['seq'])
        start_seq = end_seq - count + 1
        
        # Assign IDs
        for i, slot in enumerate(slots_data):
            slot['id'] = start_seq + i
            
        # Bulk insert
        db._db['timeslot'].insert_many(slots_data)
        print(f"[Performance] Bulk inserted {count} time slots.")


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        broker=app.config.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)

# Load configuration from environment variables
mongo_uri = os.getenv('MONGO_URI')
mongo_dbname = os.getenv('MONGO_DBNAME', 'timetable')
secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key')

# Detect environment
is_development = os.getenv('FLASK_ENV') == 'development' or os.getenv('ENV') == 'development' or app.debug

# For local development, add TLS parameters if not already present
if mongo_uri and is_development:
    if 'mongodb+srv://' in mongo_uri or 'mongodb.net' in mongo_uri:
        # Add TLS parameters for Atlas connections in development
        if 'tls=' not in mongo_uri.lower() and 'ssl=' not in mongo_uri.lower():
            separator = '&' if '?' in mongo_uri else '?'
            mongo_uri += f'{separator}tls=true&tlsAllowInvalidCertificates=true'
            print("[Config] Added TLS parameters for local development")

app.config['MONGO_URI'] = mongo_uri
app.config['MONGO_DBNAME'] = mongo_dbname
app.config['SECRET_KEY'] = secret_key
app.config['ENV'] = 'development' if is_development else 'production'

# Profiling Middleware
@app.before_request
def before_request():
    request._start_time = time.time()
    
    if 'profile' in request.args and PROFILER_AVAILABLE:
        g.profiler = Profiler()
        g.profiler.start()

@app.after_request
def after_request(response):
    # Timing Log
    if hasattr(request, '_start_time'):
        elapsed = time.time() - request._start_time
        # Log to console/file
        app.logger.info(f"[{request.remote_addr}] {request.method} {request.path} {elapsed:.3f}s")
        
        # Add header
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

    # Profiler Report
    if hasattr(g, 'profiler'):
        g.profiler.stop()
        output_html = g.profiler.output_html()
        return make_response(output_html)
        
    return response

# Celery Configuration
app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = make_celery(app)

def _generate_timetable_core(filters=None, progress_callback=None):
    """
    Core timetable generation logic (can be called synchronously or from Celery).
    
    Args:
        filters: Optional dict with program, branch, semester filters
        progress_callback: Optional function to call with progress updates
    
    Returns:
        dict with success status and generation results
    """
    try:
        if progress_callback:
            progress_callback('PROGRESS', {'status': 'Initializing generation...'})
        
        # Log the filters being used for this generation
        print(f"[GENERATE_CORE] Starting generation with filters: {filters}")
        
        # Clear existing timetable (selectively for target groups only)
        if filters:
            # Build target groups query based on same filters used for generation
            # This ensures we only delete what we are about to replace
            target_groups_query = StudentGroup.query
            
            # Apply filters to target only specific program/branch/semester
            if filters.get('program'):
                target_groups_query = target_groups_query.filter_by(program=filters['program'])
                print(f"[GENERATE_CORE] Filter: program = {filters['program']}")
            if filters.get('branch'):
                target_groups_query = target_groups_query.filter_by(branch=filters['branch'])
                print(f"[GENERATE_CORE] Filter: branch = {filters['branch']}")
            if filters.get('semester'):
                try:
                    sem = int(filters['semester'])
                    target_groups_query = target_groups_query.filter_by(semester=sem)
                    print(f"[GENERATE_CORE] Filter: semester = {sem}")
                except (ValueError, TypeError): 
                    print(f"[GENERATE_CORE] Warning: Invalid semester value: {filters.get('semester')}")
            
            groups_to_clear = target_groups_query.all()
            
            if groups_to_clear:
                group_names = [g.name for g in groups_to_clear]
                print(f"[GENERATE_CORE] Found {len(group_names)} student groups to clear: {group_names}")
                
                # TARGETED DELETION: Only remove entries for these exact student groups
                # This prevents clearing the whole department/semester if only one branch is selected
                deleted_result = db._db['timetableentry'].delete_many({
                    'student_group': {'$in': group_names}
                })
                print(f"[GENERATE_CORE] ✓ Targeted Deletion Complete: Removed {deleted_result.deleted_count} timetable entries")
                print(f"[GENERATE_CORE] ✓ Affected student groups: {group_names}")
                print(f"[GENERATE_CORE] ✓ Other semesters/branches remain untouched")
            else:
                print(f"[GENERATE_CORE] ⚠ Warning: No student groups found matching filters {filters}")
                print(f"[GENERATE_CORE] ⚠ No timetable entries will be deleted")
                print(f"[GENERATE_CORE] ⚠ Generation will proceed but may not produce results")
        else:
            # Dangerous full deletion only if no filters provided
            print("[GENERATE_CORE] ⚠ CRITICAL: Full timetable clear triggered (no filters)")
            print("[GENERATE_CORE] ⚠ This will delete ALL timetable entries")
            TimetableEntry.query.delete()
        
        db.session.commit()
        
        if progress_callback:
            progress_callback('PROGRESS', {'status': 'Running algorithm...'})
        
        # Pre-filter courses and student groups based on filters
        target_courses = None
        target_groups = None
        
        if filters:
            # Build course query
            course_query = Course.query
            if filters.get('program'):
                course_query = course_query.filter_by(program=filters['program'])
            if filters.get('branch'):
                course_query = course_query.filter_by(branch=filters['branch'])
            if filters.get('semester'):
                course_query = course_query.filter_by(semester=filters['semester'])
            target_courses = course_query.all()
            
            # Build student group query
            group_query = StudentGroup.query
            if filters.get('program'):
                group_query = group_query.filter_by(program=filters['program'])
            if filters.get('branch'):
                group_query = group_query.filter_by(branch=filters['branch'])
            if filters.get('semester'):
                group_query = group_query.filter_by(semester=filters['semester'])
            target_groups = group_query.all()
        
        # Generate new timetable
        print(f"[GENERATE_CORE] Using {len(target_courses) if target_courses else 0} courses and {len(target_groups) if target_groups else 0} groups")
        if target_groups:
            print(f"[GENERATE_CORE] Group names: {[g.name for g in target_groups]}")
        generator = TimetableGenerator(db, courses=target_courses, groups=target_groups, config={
            'verbose': True,  # Enable for performance logging
            'ultra_fast': True,  # CRITICAL: Enable greedy-first strategy
            'skip_faculty_schedules': True,  # Skip faculty schedules generation for speed
            'skip_overwork_check': False,  # Keep overwork check but make it fast
            'greedy_success_threshold': 0.7  # Accept greedy if >=70% placement rate
        })
        result = generator.generate(filters or {})
        
        # Save to history if generation was successful
        if result.get('success'):
            try:
                from datetime import datetime
                
                # Fetch all current timetable entries to save in history
                all_entries = TimetableEntry.query.all()
                entries_data = []
                
                for entry in all_entries:
                    entries_data.append({
                        'time_slot_id': getattr(entry, 'time_slot_id', None),
                        'course_id': getattr(entry, 'course_id', None),
                        'faculty_id': getattr(entry, 'faculty_id', None),
                        'room_id': getattr(entry, 'room_id', None),
                        'student_group': getattr(entry, 'student_group', None)
                    })
                
                # Build a descriptive name for this timetable
                name_parts = []
                if filters:
                    if filters.get('program'):
                        name_parts.append(filters['program'])
                    if filters.get('branch'):
                        name_parts.append(filters['branch'])
                    if filters.get('semester'):
                        name_parts.append(f"Sem-{filters['semester']}")
                
                timetable_name = " ".join(name_parts) if name_parts else "Full Timetable"
                timetable_name += f" - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # Create history record
                history = TimetableHistory(
                    name=timetable_name,
                    generated_at=datetime.utcnow().isoformat(),
                    generated_by=filters.get('user_id') if filters else None,
                    filters=json.dumps(filters or {}),
                    entries=json.dumps(entries_data),
                    stats=json.dumps({
                        'total_entries': len(entries_data),
                        'total_courses': len(target_courses) if target_courses else 0,
                        'total_groups': len(target_groups) if target_groups else 0,
                        'warnings': result.get('warnings', []),
                        'generation_time': result.get('generation_time', 0)
                    }),
                    is_active=True
                )
                
                # Mark all other histories as inactive
                db._db['timetablehistory'].update_many(
                    {'is_active': True},
                    {'$set': {'is_active': False}}
                )
                
                db.session.add(history)
                db.session.commit()
                
                print(f"[GENERATE_CORE] ✓ Saved timetable to history: {timetable_name}")
                print(f"[GENERATE_CORE] ✓ History ID: {history.id}")
            except Exception as e:
                print(f"[GENERATE_CORE] ⚠ Warning: Failed to save history: {str(e)}")
                # Don't fail the entire generation if history save fails
        
        return result
    except Exception as e:
        print(f"[GENERATE_CORE] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


@celery.task(bind=True)
def generate_timetable_task(self, filters=None):
    """Background task wrapper for timetable generation"""
    def progress_callback(state, meta):
        """Update Celery task state"""
        self.update_state(state=state, meta=meta)
    
    return _generate_timetable_core(filters=filters, progress_callback=progress_callback)


# Initialize our MongoDB-backed db compatibility layer
print("\n" + "="*80)
print("INITIALIZING DATABASE CONNECTION")
print("="*80)

try:
    db.init_app(app)
    print("[App] Database initialization successful")
    print("="*80 + "\n")
except Exception as e:
    print("\n" + "="*80)
    print("CRITICAL ERROR: Database Initialization Failed")
    print("="*80)
    print(f"\nError: {str(e)}")
    print("\nWARNING: The application cannot start without a database connection.")
    print("\nQuick Fix Steps:")
    print("1. Check your .env file exists in the project root")
    print("2. Verify MONGO_URI is set correctly")
    print("3. For MongoDB Atlas:")
    print("   - Go to Network Access in Atlas dashboard")
    print("   - Add your current IP address to the whitelist")
    print("   - Or add 0.0.0.0/0 for development (not recommended for production)")
    print("4. Ensure your MongoDB cluster is running")
    print("\nExample .env file:")
    print("MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/")
    print("MONGO_DBNAME=timetable")
    print("="*80 + "\n")
    raise SystemExit("Database connection failed. Please fix the issues above and restart.")

# Centralized metadata helper function
def get_metadata_context():
    """
    Centralized helper to fetch unique programs, branches, and semesters.
    This ensures 100% synchronization across all forms and pages.
    
    CRITICAL: Dropdowns MUST NEVER expose raw database strings!
    
    Why Normalization is Essential:
    1. Prevents Duplicates: Raw DB may have "Btech", "btech", "B.Tech"
       → Normalized output: ["btech"] (single value)
    
    2. Ensures Consistency: User selections will match database values
       → User selects "btech" → Query finds program="btech" ✓
    
    3. Data Integrity: Dropdowns reflect canonical format
       → Users can only select valid, normalized values
    
    4. Prevents Mismatches: No case/format differences between UI and DB
       → "Btech" != "btech" (would fail) vs "btech" == "btech" (works)
    
    IMPORTANT: Programs and Branches come ONLY from the Branch collection.
    This makes the Courses section the MASTER SOURCE.
    Student Groups can only use programs/branches created in Courses.
    
    Returns:
        dict: Contains 'programs', 'branches', and 'semesters' lists
              All values are normalized (lowercase, no dots, no spaces)
    """
    try:
        from normalization import normalize_key
        
        # SINGLE SOURCE OF TRUTH: Branch collection only (for programs and branches)
        # This is populated when you create courses/branches in the Courses section
        branch_programs = db._db['branch'].distinct('program')
        branch_names = db._db['branch'].distinct('name')  # Branch names (e.g., "Computer Science")
        
        # Semesters come from Course collection (course-specific data)
        course_semesters = db._db['course'].distinct('semester')
        
        # Normalize programs using canonical format (lowercase, no dots, no spaces)
        all_programs = set()
        for p in branch_programs:
            if p and str(p).strip():
                normalized = normalize_key(str(p))
                if normalized:
                    all_programs.add(normalized)
        
        # Normalize branches using canonical format
        all_branches = set()
        for b in branch_names:
            if b and str(b).strip():
                normalized = normalize_key(str(b))
                if normalized:
                    all_branches.add(normalized)
        
        # Process semesters (integers)
        all_semesters = set()
        for s in course_semesters:
            if s is not None:
                try:
                    all_semesters.add(int(s))
                except (ValueError, TypeError):
                    pass
        
        # If no semesters found, provide default range
        if not all_semesters:
            all_semesters = set(range(1, 9))  # Default: 1-8
        
        return {
            'programs': sorted(list(all_programs)),
            'branches': sorted(list(all_branches)),
            'semesters': sorted(list(all_semesters))
        }
    except Exception as e:
        # Return empty lists on error to prevent template crashes
        print(f"[Metadata] Error fetching metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'programs': [],
            'branches': [],
            'semesters': list(range(1, 9))  # Default semesters
        }


# Inject metadata and next_page into all templates
@app.context_processor
def inject_template_globals():
    """
    Inject global variables into all templates.
    This ensures metadata is always available without manual passing.
    """
    try:
        path = request.path or '/'
    except RuntimeError:
        # No request context
        path = '/'
    
    # Calculate next_page for navigation
    navigation_order = ['/', '/courses', '/faculty', '/rooms', '/students', '/student-groups', '/settings', '/timetable']
    next_page = None
    
    # Exact match
    if path in navigation_order:
        idx = navigation_order.index(path)
        if idx < len(navigation_order) - 1:
            next_page = navigation_order[idx + 1]
    else:
        # Handle subpaths like /courses/add or /faculty/123 by matching prefix
        for i, p in enumerate(navigation_order):
            if p != '/' and path.startswith(p + '/'):
                if i < len(navigation_order) - 1:
                    next_page = navigation_order[i + 1]
                break
    
    # Get metadata for all templates
    metadata = get_metadata_context()
    
    return {
        'next_page': next_page,
        'metadata': metadata,
        # Also expose individual lists for convenience
        'available_programs': metadata['programs'],
        'available_branches': metadata['branches'],
        'available_semesters': metadata['semesters']
    }


# Initialize database with default data
with app.app_context():
    # MongoDB doesn't require schema migrations - collections are created automatically
    # We only need to ensure default data exists
    
    # Retry loop for VPN connection stability
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Init] Attempt {attempt}/{max_retries}: Checking database connection...")
            
            # Test connection by attempting a simple query
            # Create default period config if it doesn't exist (singleton enforced)
            if PeriodConfig.query.count() == 0:
                print("[Init] Creating default period configuration...")
                period_config = PeriodConfig(
                    id=1,
                    periods_per_day=8,
                    period_duration_minutes=60,
                    day_start_time='09:00',
                    days_of_week='Monday,Tuesday,Wednesday,Thursday,Friday'
                )
                db.session.add(period_config)
                db.session.commit()
                print("[Init] ✓ Period configuration created")
            
            # Create default break configs if they don't exist
            if BreakConfig.query.count() == 0:
                print("[Init] Creating default break configurations...")
                breaks = [
                    BreakConfig(break_name='Short Break', after_period=2, duration_minutes=15, order=1),
                    BreakConfig(break_name='Lunch Break', after_period=4, duration_minutes=60, order=2),
                    BreakConfig(break_name='Short Break', after_period=6, duration_minutes=15, order=3)
                ]
                for br in breaks:
                    db.session.add(br)
                db.session.commit()
                print("[Init] ✓ Break configurations created")
            
            # Generate time slots based on config if they don't exist
            if TimeSlot.query.count() == 0:
                print("[Init] Generating time slots...")
                generate_time_slots()
                print("[Init] ✓ Time slots generated")
            
            # Create default admin user if it doesn't exist
            if User.query.filter_by(username='admin').first() is None:
                print("[Init] Creating default admin user...")
                admin = User(username='admin', email='admin@college.edu', role='admin', name='Administrator')
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("[Init] ✓ Admin user created (username: admin, password: admin123)")

            # One-time migration for legacy faculty data (optimized to skip if not needed)
            hydrate_default_faculty_values()
            
            # Fix any branches with invalid or empty codes (self-healing)
            invalid_branches = Branch.query.all()
            found_broken = False
            for b in invalid_branches:
                needs_save = False
                if not getattr(b, 'code', None) or str(b.code).strip() == "":
                    new_code = (b.name[:10].upper().replace(" ", "") if getattr(b, 'name', None) else f"BR{b.id}")
                    print(f"[Init] Fixing branch {b.id} ('{getattr(b, 'name', 'Unknown')}') missing code. New code: {new_code}")
                    b.code = new_code
                    needs_save = True
                
                # Normalize existing codes (trim, uppercase, no spaces)
                old_code = str(b.code)
                normalized = old_code.strip().upper().replace(" ", "")
                if old_code != normalized:
                    print(f"[Init] Normalizing branch code: '{old_code}' -> '{normalized}'")
                    b.code = normalized
                    needs_save = True
                
                if needs_save:
                    b.save()
                    found_broken = True
            
            if found_broken:
                db.session.commit()
                print("[Init] Branch cleanup complete")
            
            # If we got here, initialization succeeded
            print(f"[Init] Database initialization successful on attempt {attempt}")
            break  # Exit retry loop on success
            
        except Exception as e:
            print(f"[Init] Attempt {attempt}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries:
                print(f"[Init] Retrying in {retry_delay} seconds...")
                import time as time_module
                time_module.sleep(retry_delay)
            else:
                print("[Init] ✗ All retry attempts exhausted")
                print("="*80)
                print("CRITICAL: Database initialization failed after 3 attempts")
                print("="*80)
                print("\nPossible causes:")
                print("1. VPN connection is unstable or disconnected")
                print("2. MongoDB Atlas is unreachable")
                print("3. Network firewall is blocking the connection")
                print("4. IP address not whitelisted in MongoDB Atlas")
                print("\nPlease check:")
                print("- VPN connection is active and stable")
                print("- MongoDB Atlas Network Access whitelist includes your IP")
                print("- Internet connection is working")
                print("="*80 + "\n")
                raise SystemExit("Database initialization failed - please check VPN and network connection")



def get_current_user():
    """Get the current user from session or JWT"""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    if hasattr(g, 'user_id'):
        return User.query.get(g.user_id)
    return None

def safe_get_request_data():
    """
    Safely extract data from request, supporting JSON, form data, and query parameters.
    Returns empty dict if no data is available or parsing fails.
    
    This prevents JSONDecodeError when request body is empty or malformed.
    """
    # Try JSON first (most API calls)
    try:
        data = request.get_json(force=True, silent=True)
        if data is not None:
            return data
    except Exception as e:
        app.logger.debug(f"[Request] JSON parsing failed: {e}")
    
    # Try form data (HTML forms)
    try:
        if request.form:
            return request.form.to_dict()
    except Exception as e:
        app.logger.debug(f"[Request] Form parsing failed: {e}")
    
    # Try query parameters (GET requests with filters)
    try:
        if request.args:
            return request.args.to_dict()
    except Exception as e:
        app.logger.debug(f"[Request] Args parsing failed: {e}")
    
    # Return empty dict as fallback
    return {}

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Improved API detection
        is_api_request = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
            request.is_json or
            'application/json' in request.headers.get('Accept', '') or
            request.path.startswith('/api/')
        )

        if 'user_id' not in session:
            # Check JWT
            token = request.cookies.get('access_token')
            if token:
                payload = decode_token(token)
                if payload and payload['type'] == 'access':
                    g.user_id = int(payload['sub'])
                    g.user_role = payload['role']
                    return f(*args, **kwargs)
            
            if is_api_request:
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Improved API detection
        is_api_request = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
            request.is_json or
            'application/json' in request.headers.get('Accept', '') or
            request.path.startswith('/api/')
        )

        # Check Session
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            if not user or user.role != 'admin':
                if is_api_request:
                    return jsonify({'success': False, 'error': 'Access denied. Admin privileges required.'}), 403
                flash('Access denied. Admin privileges required.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)

        # Check JWT
        token = request.cookies.get('access_token')
        if token:
            payload = decode_token(token)
            if payload and payload['type'] == 'access':
                if payload['role'] == 'admin':
                    g.user_id = int(payload['sub'])
                    g.user_role = payload['role']
                    return f(*args, **kwargs)
        
        if is_api_request:
            return jsonify({'success': False, 'error': 'Access denied. Admin privileges required.'}), 403
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login'))
    return decorated_function


@app.route('/api/metadata')
@login_required
def get_metadata():
    """
    API endpoint to provide unique lists of programs and branches for dropdowns.
    
    IMPORTANT: This endpoint ONLY fetches from the Branch collection.
    The Branch collection is the MASTER SOURCE populated when courses are created.
    This ensures Student Groups can ONLY select programs/branches that exist in Courses.
    
    Data Flow:
    1. Admin creates Branch in Courses section → Branch collection
    2. Student Groups fetches from /api/metadata → Branch collection ONLY
    3. This prevents orphaned data and ensures consistency
    """
    try:
        # SINGLE SOURCE OF TRUTH: Branch collection only
        # This is populated when you create courses/branches in the Courses section
        branch_programs = db._db['branch'].distinct('program')
        branch_names = db._db['branch'].distinct('name')  # Branch names (e.g., "Computer Science")
        
        from normalization import normalize_key
        # Normalize to canonical format (lowercase, no dots/spaces)
        all_programs = set()
        for p in branch_programs:
            normalized = normalize_key(str(p))
            if normalized: all_programs.add(normalized)
        
        all_branches = set()
        for b in branch_names:
            normalized = normalize_key(str(b))
            if normalized: all_branches.add(normalized)
        
        programs = sorted(list(all_programs))
        branches = sorted(list(all_branches))
        
        # If no data found, provide helpful message
        if not programs and not branches:
            return jsonify({
                'success': True,
                'programs': [],
                'branches': [],
                'semesters': [1, 2, 3, 4, 5, 6, 7, 8],
                'message': 'No programs or branches found. Please create them in the Courses section first.'
            })
        
        return jsonify({
            'success': True,
            'programs': programs,
            'branches': branches,
            'semesters': [1, 2, 3, 4, 5, 6, 7, 8]
        })
    except Exception as e:
        print(f"[METADATA] Error fetching metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'programs': [],
            'branches': [],
            'semesters': [1, 2, 3, 4, 5, 6, 7, 8]
        }), 500

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        # ============================================
        # BACKGROUND TASK PROTECTION
        # ============================================
        # Check if user account is still being created by background worker
        if not user:
            # Check if faculty/student exists but user account not yet created
            faculty = Faculty.query.filter_by(username=username).first()
            student = Student.query.filter_by(username=username).first()
            
            if faculty or student:
                # Account exists in Faculty/Student but not in User yet
                # Background worker is still processing
                if request.headers.get('Content-Type') == 'application/json' or request.is_json:
                    return jsonify({
                        'status': 'processing',
                        'message': 'Account is being finalized. Please wait 10 seconds and try again.'
                    }), 202  # 202 Accepted
                else:
                    flash('⏳ Your account is being set up. Please wait 10 seconds and try again.', 'info')
                    return render_template('login.html')
        
        if user and user.check_password(password):
            # Save user in case password was migrated from Werkzeug to bcrypt
            user.save()
            
            # Create JWTs
            access_token, refresh_token = create_tokens(user.id, user.role)
            
            # Set Session (Legacy support)
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['name'] = user.name
            
            flash(f'Welcome back, {user.name}!', 'success')
            
            resp = make_response(redirect(url_for('index')))
            
            # Set Cookies (HttpOnly, Secure if HTTPS)
            is_secure = request.scheme == 'https'
            resp.set_cookie('access_token', access_token, httponly=True, secure=is_secure, samesite='Lax', max_age=15*60)
            resp.set_cookie('refresh_token', refresh_token, httponly=True, secure=is_secure, samesite='Lax', max_age=7*24*60*60)
            
            return resp
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        name = request.form.get('name')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return render_template('register.html')
        
        user = User(username=username, email=email, role=role, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/refresh', methods=['POST'])
def refresh():
    refresh_token = request.cookies.get('refresh_token')
    if not refresh_token:
        return jsonify({'message': 'Missing refresh token'}), 401
        
    payload = decode_token(refresh_token)
    if not payload or payload['type'] != 'refresh':
        return jsonify({'message': 'Invalid refresh token'}), 401
        
    # Rotate tokens: Revoke old refresh token
    revoke_token(payload['jti'], 7*24*60*60)
    
    new_access, new_refresh = create_tokens(payload['sub'], payload['role'])
    
    resp = make_response(jsonify({'message': 'Token refreshed'}))
    is_secure = request.scheme == 'https'
    resp.set_cookie('access_token', new_access, httponly=True, secure=is_secure, samesite='Lax', max_age=15*60)
    resp.set_cookie('refresh_token', new_refresh, httponly=True, secure=is_secure, samesite='Lax', max_age=7*24*60*60)
    
    return resp

@app.route('/logout')
def logout():
    try:
        # Revoke tokens if present
        access_token = request.cookies.get('access_token')
        refresh_token = request.cookies.get('refresh_token')
        
        if access_token:
            try:
                payload = decode_token(access_token)
                if payload:
                    revoke_token(payload['jti'], 15*60)
            except Exception:
                pass
                
        if refresh_token:
            try:
                payload = decode_token(refresh_token)
                if payload:
                    revoke_token(payload['jti'], 7*24*60*60)
            except Exception:
                pass

        session.clear()
        flash('You have been logged out', 'info')
        resp = make_response(redirect(url_for('login')))
        resp.delete_cookie('access_token')
        resp.delete_cookie('refresh_token')
        return resp
    except Exception as e:
        print(f"Logout error: {e}")
        session.clear()
        return redirect(url_for('login'))


@app.route('/download-template/<entity>')
@admin_required
def download_template(entity):
    """Generate a CSV or Excel template for courses, faculty, rooms, students, or student-groups and send as attachment.
    Usage: /download-template/courses?format=csv or ?format=xlsx
    """
    fmt = (request.args.get('format') or 'csv').lower()
    if entity not in ('courses', 'faculty', 'rooms', 'students', 'student-groups'):
        abort(404)

    if entity == 'courses':
        # Check if custom courses_template.xlsx exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses_template.xlsx')
        if os.path.exists(template_path):
            if fmt in ('xls', 'xlsx'):
                return send_file(template_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='courses_template.xlsx')
            elif fmt == 'csv' and PANDAS_AVAILABLE:
                try:
                    # Convert custom Excel template to CSV
                    df = pd.read_excel(template_path)
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    mem = io.BytesIO()
                    mem.write(output.getvalue().encode('utf-8'))
                    mem.seek(0)
                    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='courses_template.csv')
                except Exception as e:
                    print(f"Error converting custom courses template to CSV: {e}")
                    # Fall through to default generation

        columns = ['code', 'name', 'credits', 'hours_per_week', 'course_type', 'subject_type', 'program', 'branch', 'semester', 'required_room_tags']
        example = ['CS101', 'Introduction to Programming', '4', '5', 'Theory', 'Major', 'B.Tech', 'Computer Science', '1', 'projector,computer']
        filename_base = 'courses_template'
    elif entity == 'faculty':
        # Check if custom faculty_template.xlsx exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faculty_template.xlsx')
        if os.path.exists(template_path):
            if fmt in ('xls', 'xlsx'):
                return send_file(template_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='faculty_template.xlsx')
            elif fmt == 'csv' and PANDAS_AVAILABLE:
                try:
                    # Convert custom Excel template to CSV
                    df = pd.read_excel(template_path)
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    mem = io.BytesIO()
                    mem.write(output.getvalue().encode('utf-8'))
                    mem.seek(0)
                    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='faculty_template.csv')
                except Exception as e:
                    print(f"Error converting custom faculty template to CSV: {e}")
                    # Fall through to default generation
        
        columns = ['full name', 'username', 'email', 'departments', 'expertise', 'min_hours_per_week', 'max_hours_per_week', 'availability (optional)']
        example = ['Dr. John Smith', 'john.smith', 'john.smith@university.edu', 'Computer Science, IT', 'cs101,cs102,math101', '8', '16', '{"Monday": [1,2,3,4,5], "Tuesday": [1,2,3,4,5]}']
        filename_base = 'faculty_template'
    elif entity == 'rooms':
        # Check if custom rooms_template.xlsx exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rooms_template.xlsx')
        if os.path.exists(template_path):
            if fmt in ('xls', 'xlsx'):
                return send_file(template_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='rooms_template.xlsx')
            elif fmt == 'csv' and PANDAS_AVAILABLE:
                try:
                    df = pd.read_excel(template_path)
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    mem = io.BytesIO()
                    mem.write(output.getvalue().encode('utf-8'))
                    mem.seek(0)
                    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='rooms_template.csv')
                except Exception as e:
                    print(f"Error converting custom rooms template to CSV: {e}")

        columns = ['name', 'capacity', 'room_type', 'equipment', 'tags']
        example = ['Lab-101', '30', 'lab', 'Computers,Projector', 'computer,projector,lab']
        filename_base = 'rooms_template'
    elif entity == 'students':
        # Check if custom students_template.xlsx exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'students_template.xlsx')
        if os.path.exists(template_path):
            if fmt in ('xls', 'xlsx'):
                return send_file(template_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='students_template.xlsx')
            elif fmt == 'csv' and PANDAS_AVAILABLE:
                try:
                    df = pd.read_excel(template_path)
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    mem = io.BytesIO()
                    mem.write(output.getvalue().encode('utf-8'))
                    mem.seek(0)
                    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='students_template.csv')
                except Exception as e:
                    print(f"Error converting custom students template to CSV: {e}")

        columns = ['student_id', 'name', 'username', 'password', 'enrolled_courses']
        example = ['2023001', 'Alice Johnson', 'alice.johnson', 'password123', 'cs101,math101,phy101']
        filename_base = 'students_template'
    elif entity == 'student-groups':
        # Check if custom student_groups_template.xlsx exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_groups_template.xlsx')
        if os.path.exists(template_path):
            if fmt in ('xls', 'xlsx'):
                return send_file(template_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='student_groups_template.xlsx')
            elif fmt == 'csv' and PANDAS_AVAILABLE:
                try:
                    df = pd.read_excel(template_path)
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    mem = io.BytesIO()
                    mem.write(output.getvalue().encode('utf-8'))
                    mem.seek(0)
                    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='student_groups_template.csv')
                except Exception as e:
                    print(f"Error converting custom student groups template to CSV: {e}")

        columns = ['name', 'description', 'program', 'branch', 'semester', 'total_students', 'batches', 'batches_students']
        example = ['CS-A', 'Computer Science Section A', 'B.Tech', 'Computer Science', '1', '60', '2', '30,30']
        filename_base = 'student_groups_template'

    if fmt == 'csv':
        # Create CSV in-memory
        output = io.StringIO()
        import csv as _csv
        writer = _csv.writer(output)
        writer.writerow(columns)
        writer.writerow(example)  # Add example data row
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        return send_file(mem, mimetype='text/csv', as_attachment=True, download_name=f"{filename_base}.csv")

    elif fmt in ('xls', 'xlsx'):
        # Use pandas to create an Excel file in-memory. Try available engines.
        if not PANDAS_AVAILABLE:
            # Fallback to CSV if pandas not available
            output = io.StringIO()
            import csv as _csv
            writer = _csv.writer(output)
            writer.writerow(columns)
            writer.writerow(example)  # Add example data row
            mem2 = io.BytesIO()
            mem2.write(output.getvalue().encode('utf-8'))
            mem2.seek(0)
            return send_file(mem2, mimetype='text/csv', as_attachment=True, download_name=f"{filename_base}.csv")
        
        df = pd.DataFrame([example], columns=columns)  # Create DataFrame with example data
        mem = io.BytesIO()
        engines_to_try = ['xlsxwriter', 'openpyxl']
        writer_used = None
        for eng in engines_to_try:
            try:
                with pd.ExcelWriter(mem, engine=eng) as writer:
                    df.to_excel(writer, index=False, sheet_name='Template')
                writer_used = eng
                break
            except ModuleNotFoundError:
                # try next engine
                mem.seek(0)
                mem.truncate(0)
                continue

        if not writer_used:
            # Fallback: return CSV if no excel engine is available
            output = io.StringIO()
            import csv as _csv
            writer = _csv.writer(output)
            writer.writerow(columns)
            writer.writerow(example)  # Add example data row
            mem2 = io.BytesIO()
            mem2.write(output.getvalue().encode('utf-8'))
            mem2.seek(0)
            return send_file(mem2, mimetype='text/csv', as_attachment=True, download_name=f"{filename_base}.csv")

        mem.seek(0)
        return send_file(mem, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name=f"{filename_base}.xlsx")

    else:
        # Unsupported format
        return jsonify({'success': False, 'error': 'Unsupported format'}), 400

# Health Check Endpoint (for load balancers, Docker, monitoring)
@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns 200 OK if application is healthy.
    """
    try:
        # Check database connectivity
        db._db.command('ping')
        
        return jsonify({
            'status': 'healthy',
            'service': 'AI Timetable Generator',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'AI Timetable Generator',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/')
@login_required
def index():
    user = get_current_user()
    
    # Redirect students to their dashboard
    if user and user.role == 'student':
        return redirect('/student/dashboard')
    
    stats = {
        'courses': Course.query.count(),
        'faculty': Faculty.query.count(),
        'rooms': Room.query.count(),
        'students': Student.query.count(),
        'timetable_entries': TimetableEntry.query.count()
    }
    return render_template('index.html', stats=stats, user=user)

# Course Management
@app.route('/courses')
@login_required
def courses():
    """Main courses page - now shows branches with subjects organized by semester"""
    user = get_current_user()
    
    # Get all branches
    branches = Branch.query.all()
    
    # Build structure: branch -> semesters -> subjects
    # Get all subjects once to reduce database queries (Optimization)
    all_subjects = Course.query.all()

    # Build structure: branch -> semesters -> subjects
    branch_structure = {}
    for branch in branches:
        # Filter subjects for this branch in Python (case-insensitive)
        subjects = [
            s for s in all_subjects 
            if str(s.program).strip().lower() == str(branch.program).strip().lower() 
            and str(s.branch).strip().lower() == str(branch.name).strip().lower()
        ]
        
        # Organize by semester
        subjects_by_semester = {}
        for semester in range(1, branch.total_semesters + 1):
            semester_subjects = [
                s for s in subjects 
                if getattr(s, 'semester', None) == semester
            ]
            subjects_by_semester[semester] = [
                {
                    'id': str(s.id) if hasattr(s.id, '__str__') else s.id,  # Convert ObjectId to string
                    'code': s.code,
                    'name': s.name,
                    'credits': getattr(s, 'credits', 0),
                    'course_type': getattr(s, 'course_type', 'theory'),
                    'subject_type': getattr(s, 'subject_type', None),
                    'hours_per_week': getattr(s, 'hours_per_week', 0)
                }
                for s in semester_subjects
            ]
        
        # Convert branch data to dict and ensure ObjectId is converted
        branch_dict = branch.to_dict()
        if '_id' in branch_dict:
            branch_dict['_id'] = str(branch_dict['_id'])
        if 'id' in branch_dict:
            branch_dict['id'] = str(branch_dict['id'])
        
        branch_structure[branch.code] = {
            'branch': branch_dict,
            'subjects_by_semester': subjects_by_semester
        }
    
    return render_template(
        'courses.html',
        branches=branches,
        branch_structure=branch_structure,
        user=user
    )

@app.route('/courses/add', methods=['POST'])
@admin_required
def add_course():
    """
    DEPRECATED: This route is kept for backward compatibility only.
    Use /branches/<code>/subjects/add instead for the new branch-based system.
    """
    return jsonify({
        'success': False,
        'error': 'This endpoint is deprecated. Please use the new branch-based system: Create a branch first, then add subjects to it.',
        'redirect': '/courses'
    }), 400

@app.route('/courses/<int:course_id>/delete', methods=['POST'])
@admin_required
def delete_course(course_id):
    course = Course.query.get_or_404(course_id)
    # Remove timetable entries referencing this course first to avoid
    # NOT NULL / FK constraint failures when the course is deleted.
    TimetableEntry.query.filter_by(course_id=course.id).delete(synchronize_session=False)
    db.session.delete(course)
    db.session.commit()
    invalidate_cache('timetable_view')
    return jsonify({'success': True})

@app.route('/courses/import', methods=['POST'])
@admin_required
def import_courses():
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    try:
        # Validate file type and get streaming processor
        chunks_generator = process_upload_stream(upload, chunk_size=1000)
        
        # Validate required columns from first chunk
        first_chunk = None
        required_columns = {'code', 'name', 'credits', 'hours_per_week'}
        
        # Pre-fetch existing courses to avoid N+1 queries
        existing_courses = {c.code: c for c in Course.query.all()}
        
        created, updated = 0, 0
        
        for chunk_idx, chunk in enumerate(chunks_generator):
            # Validate columns on first chunk
            if chunk_idx == 0 and chunk:
                available_columns = set(chunk[0].keys())
                missing = get_missing_columns(available_columns, required_columns)
                if missing:
                    return jsonify({
                        'success': False,
                        'error': f'Missing columns: {", ".join(sorted(missing))}'
                    }), 400
            
            # Process chunk
            for row in chunk:
                code = str(row.get('code', '')).strip()
                if not code:
                    continue
                
                course = existing_courses.get(code)
                course_type = str(row.get('course_type', row.get('type', 'theory'))).lower()
                course_type = 'practical' if 'prac' in course_type else 'theory'
                tags_raw = row.get('required_room_tags') or row.get('room_tags') or ''
                tags = ','.join(tag.strip() for tag in str(tags_raw).split(',') if tag.strip())

                # Extract and normalize subject_type
                subject_type_raw = str(row.get('subject_type', '')).strip().lower()
                subject_type = subject_type_raw if subject_type_raw in ['major', 'minor', 'md', 'ae', 'se', 'va'] else None

                # NORMALIZE program and branch to canonical format (lowercase, no dots/spaces)
                # This ensures consistent matching with student groups and branches
                from normalization import normalize_key
                program_raw = str(row.get('program', '')).strip()
                program = normalize_key(program_raw) if program_raw else None
                
                branch_raw = str(row.get('branch', '')).strip()
                branch = normalize_key(branch_raw) if branch_raw else None

                payload = {
                    'code': code,
                    'name': str(row.get('name', code)).strip(),
                    'credits': parse_int(row.get('credits'), 0),
                    'course_type': course_type,
                    'subject_type': subject_type,
                    'hours_per_week': parse_int(row.get('hours_per_week'), 1),
                    'program': program,
                    'branch': branch,
                    'semester': parse_int(row.get('semester'), 0),
                    'required_room_tags': tags
                }

                if course:
                    course.name = payload['name']
                    course.credits = payload['credits']
                    course.course_type = payload['course_type']
                    course.subject_type = payload['subject_type']
                    course.hours_per_week = payload['hours_per_week']
                    course.program = payload['program']
                    course.branch = payload['branch']
                    course.semester = payload['semester']
                    course.required_room_tags = payload['required_room_tags']
                    updated += 1
                    db.session.add(course)
                else:
                    course = Course(
                        code=payload['code'],
                        name=payload['name'],
                        credits=payload['credits'],
                        course_type=payload['course_type'],
                        subject_type=payload['subject_type'],
                        hours_per_week=payload['hours_per_week'],
                        program=payload['program'],
                        branch=payload['branch'],
                        semester=payload['semester'],
                        required_room_tags=payload['required_room_tags']
                    )
                    existing_courses[code] = course
                    db.session.add(course)
                    created += 1
            
            # Commit after each chunk for better memory management
            db.session.commit()
        
        # Auto-create missing branches for the imported courses
        # This ensures they show up in the UI which is grouped by branch
        print("[Import] Verifying branches for imported courses...")
        all_imported_courses = Course.query.all()
        unique_combinations = set()
        
        # Collect unique (program, branch) combinations (already normalized in storage)
        for c in all_imported_courses:
            if c.program and c.branch:
                unique_combinations.add((c.program, c.branch))
        
        for prog_normalized, br_name_normalized in unique_combinations:
            # Check if branch exists (exact match since both are normalized)
            all_branches = Branch.query.all()
            existing = None
            for b in all_branches:
                if (str(b.program) == prog_normalized and 
                    str(b.name) == br_name_normalized):
                    existing = b
                    break
            
            if not existing:
                # Generate a code
                base_code = f"{prog_normalized[:3]}{br_name_normalized[:3]}".upper()
                # Ensure uniqueness
                counter = 1
                final_code = base_code
                while Branch.query.filter_by(code=final_code).first():
                    final_code = f"{base_code}{counter}"
                    counter += 1
                
                print(f"[Import] Auto-creating missing branch: {prog_normalized} - {br_name_normalized} (Code: {final_code})")
                new_branch = Branch(
                    program=prog_normalized,
                    name=br_name_normalized,
                    code=final_code,
                    duration_years=4, # Default
                    total_semesters=8 # Default
                )
                db.session.add(new_branch)
        
        db.session.commit()
        
        return jsonify({'success': True, 'created': created, 'updated': updated})
    
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500


@app.route('/courses/delete-all', methods=['POST'])
@admin_required
def delete_all_courses():
    """Delete all courses"""
    try:
        # Bulk delete timetable entries first
        TimetableEntry.query.delete(synchronize_session=False)
        
        # Count courses before deletion
        deleted_count = Course.query.count()
        
        # Bulk delete all courses
        Course.query.delete(synchronize_session=False)
        
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================================
# BRANCH MANAGEMENT ROUTES
# ============================================================================

@app.route('/branches', methods=['GET'])
@login_required
def get_branches():
    """Get all branches"""
    branches = Branch.query.all()
    return jsonify({
        'success': True,
        'branches': [b.to_dict() for b in branches]
    })


@app.route('/branches/add', methods=['POST'])
@admin_required
def add_branch():
    """Create a new branch/specialization"""
    try:
        data = request.json
        
        # Validation
        if 'code' not in data or not str(data['code']).strip():
            return jsonify({'success': False, 'error': 'Branch code is required'}), 400
        if 'name' not in data or not str(data['name']).strip():
            return jsonify({'success': False, 'error': 'Branch name is required'}), 400
        if 'program' not in data or not str(data['program']).strip():
            return jsonify({'success': False, 'error': 'Degree program is required'}), 400

        # Normalize code: uppercase, trim, no spaces
        from normalization import normalize_key
        code = str(data['code']).strip().upper().replace(" ", "")
        name = normalize_key(data['name'])
        program = normalize_key(data['program'])
        
        # Check if branch code already exists
        existing = Branch.query.filter_by(code=code).first()
        if existing:
            return jsonify({
                'success': False,
                'error': f"Branch with code '{code}' already exists"
            }), 400
        
        branch = Branch(
            program=program,
            name=name,
            code=code,
            hod_name=data.get('hod_name', '').strip(),
            duration_years=int(data.get('duration_years', 4)),
            total_semesters=int(data.get('total_semesters', 8))
        )
        
        db.session.add(branch)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Branch created successfully',
            'branch': branch.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# DELETE-ALL routes must come BEFORE parameterized routes to avoid matching issues
@app.route('/branches/delete-all', methods=['POST'])
@admin_required
def delete_all_branches():
    """Delete all branches and their associated subjects"""
    try:
        # Count before deletion
        branch_count = Branch.query.count()
        
        # Delete all subjects first (all courses)
        subject_count = Course.query.delete(synchronize_session=False)
        
        # Delete all branches
        Branch.query.delete(synchronize_session=False)
        
        db.session.commit()
        invalidate_cache('courses')
        
        return jsonify({
            'success': True,
            'message': 'All branches deleted'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/branches/<branch_code>', methods=['GET'])
@login_required
def get_branch(branch_code):
    """Get a specific branch with all its subjects organized by semester"""
    branch = Branch.query.filter_by(code=branch_code).first()
    if not branch:
        return jsonify({'success': False, 'error': 'Branch not found'}), 404
    
    # Get all subjects for this branch
    subjects = Course.query.filter_by(branch=branch.name, program=branch.program).all()
    
    # Organize by semester
    subjects_by_semester = {}
    for semester in range(1, branch.total_semesters + 1):
        semester_subjects = [s for s in subjects if getattr(s, 'semester', None) == semester]
        subjects_by_semester[semester] = [
            {
                'id': s.id,
                'code': s.code,
                'name': s.name,
                'credits': getattr(s, 'credits', 0),
                'course_type': getattr(s, 'course_type', 'theory'),
                'subject_type': getattr(s, 'subject_type', None),
                'hours_per_week': getattr(s, 'hours_per_week', 0)
            }
            for s in semester_subjects
        ]
    
    return jsonify({
        'success': True,
        'branch': branch.to_dict(),
        'subjects_by_semester': subjects_by_semester
    })


@app.route('/branches/<branch_code>/delete', methods=['POST'])
@admin_required
def delete_branch(branch_code):
    """Delete a branch and all its subjects"""
    try:
        branch = Branch.query.filter_by(code=branch_code).first()
        if not branch:
            return jsonify({'success': False, 'error': 'Branch not found'}), 404
        
        # Delete all subjects in this branch
        Course.query.filter_by(branch=branch.name, program=branch.program).delete()
        
        # Delete the branch
        Branch.query.filter_by(code=branch_code).delete()
        
        db.session.commit()
        invalidate_cache('courses')
        return jsonify({'success': True, 'message': 'Branch deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# SUBJECT MANAGEMENT ROUTES (within branches)
# ============================================================================

@app.route('/branches/<branch_code>/subjects/add', methods=['POST'])
@admin_required
def add_subject_to_branch(branch_code):
    """Add a subject to a specific semester of a branch"""
    try:
        # Get the branch
        branch = Branch.query.filter_by(code=branch_code).first()
        if not branch:
            return jsonify({'success': False, 'error': 'Branch not found'}), 404
        
        data = request.json
        semester = int(data['semester'])
        
        # Validate semester is within range
        if semester < 1 or semester > branch.total_semesters:
            return jsonify({
                'success': False,
                'error': f'Semester must be between 1 and {branch.total_semesters}'
            }), 400
        
        # Check if subject code already exists in this branch
        existing = Course.query.filter_by(
            code=data['code'],
            branch=branch.name,
            program=branch.program
        ).first()
        if existing:
            return jsonify({
                'success': False,
                'error': f"Subject with code '{data['code']}' already exists in this branch"
            }), 400
        
        # Create the subject (Course)
        subject = Course(
            code=data['code'],
            name=data['name'],
            program=branch.program,
            branch=branch.name,
            semester=semester,
            credits=int(data.get('credits', 3)),
            course_type=data.get('type', 'theory').lower(),
            subject_type=data.get('subject_type'),
            hours_per_week=int(data.get('hours_per_week', 3)),
            required_room_tags=data.get('required_room_tags', '')
        )
        
        db.session.add(subject)
        db.session.commit()
        invalidate_cache('courses')
        
        return jsonify({
            'success': True,
            'message': 'Subject added',
            'subject': {
                'id': subject.id,
                'code': subject.code,
                'name': subject.name,
                'semester': subject.semester,
                'credits': subject.credits,
                'course_type': subject.course_type,
                'subject_type': subject.subject_type
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/branches/<branch_code>/subjects/delete-all', methods=['POST'])
@admin_required
def delete_all_subjects_in_branch(branch_code):
    """Delete all subjects in a specific branch"""
    try:
        branch = Branch.query.filter_by(code=branch_code).first()
        if not branch:
            return jsonify({'success': False, 'error': 'Branch not found'}), 404
        
        # Delete all subjects for this branch
        deleted_count = Course.query.filter_by(
            branch=branch.name, 
            program=branch.program
        ).delete(synchronize_session=False)
        
        db.session.commit()
        invalidate_cache('courses')
        
        return jsonify({
            'success': True,
            'message': 'All subjects deleted'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/branches/<branch_code>/subjects/<int:subject_id>/delete', methods=['POST'])
@admin_required
def delete_subject_from_branch(branch_code, subject_id):
    """Delete a subject from a branch"""
    try:
        branch = Branch.query.filter_by(code=branch_code).first()
        if not branch:
            return jsonify({'success': False, 'error': 'Branch not found'}), 404
        
        subject = Course.query.get(subject_id)
        if not subject:
            return jsonify({'success': False, 'error': 'Subject not found'}), 404
        
        # Verify subject belongs to this branch
        if subject.branch != branch.name or subject.program != branch.program:
            return jsonify({'success': False, 'error': 'Subject does not belong to this branch'}), 400
        
        Course.query.filter_by(id=subject_id).delete(synchronize_session=False)
        db.session.commit()
        invalidate_cache('courses')
        
        return jsonify({'success': True, 'message': 'Subject deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/subject/<int:subject_id>/update', methods=['POST'])
@admin_required
def update_subject(subject_id):
    """Update a subject's details"""
    try:
        subject = Course.query.get(subject_id)
        if not subject:
            return jsonify({'success': False, 'error': 'Subject not found'}), 404
        
        data = request.json
        
        # Update subject fields
        if 'name' in data:
            subject.name = data['name']
        if 'code' in data:
            subject.code = data['code']
        if 'subject_type' in data:
            subject.subject_type = data['subject_type']
        if 'type' in data:
            subject.course_type = data['type']
        if 'credits' in data:
            subject.credits = int(data['credits'])
        if 'hours_per_week' in data:
            subject.hours_per_week = int(data['hours_per_week'])
        
        subject.save()
        invalidate_cache('courses')
        
        return jsonify({'success': True, 'message': 'Subject updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/branches/<branch_code>/subjects/import', methods=['POST'])
@admin_required
def import_subjects_to_branch(branch_code):
    """Import subjects into a specific semester of a branch from CSV or Excel.

    Accepts multipart/form-data with:
    - file: CSV/XLS/XLSX file
    - semester: integer target semester (required if file rows do not include 'semester')

    Supported columns (case-insensitive):
    - code (required)
    - name (required)
    - type / course_type (optional: 'theory' or 'practical')
    - credits (optional)
    - hours_per_week (optional)
    - semester (optional; overrides form semester per row if present)
    - required_room_tags / room_tags (optional, comma-separated)
    """
    try:
        branch = Branch.query.filter_by(code=branch_code).first()
        if not branch:
            return jsonify({'success': False, 'error': 'Branch not found'}), 404

        upload = request.files.get('file')
        if not upload:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        # Semester from form; may be overridden by per-row 'semester'
        form_semester = request.form.get('semester') or request.args.get('semester')
        try:
            default_semester = int(form_semester) if form_semester not in (None, '') else None
        except (TypeError, ValueError):
            default_semester = None

        # If no default semester provided and file doesn't include 'semester', we'll validate after first chunk
        from csv_processor import process_upload_stream, get_missing_columns

        chunks_generator = process_upload_stream(upload, chunk_size=1000)

        created, updated = 0, 0

        # Index existing subjects by code for this branch+program for quick upserts
        existing = {c.code: c for c in Course.query.filter_by(program=branch.program, branch=branch.name).all()}

        first_chunk_checked = False
        for chunk in chunks_generator:
            if not first_chunk_checked:
                first_chunk_checked = True
                if chunk:
                    available_columns = set(chunk[0].keys())
                    required_cols = {'code', 'name'}
                    missing = get_missing_columns(available_columns, required_cols)
                    if missing:
                        return jsonify({'success': False, 'error': f"Missing columns: {', '.join(sorted(missing))}"}), 400

                    # If no semester provided via form and column not present -> error
                    if default_semester is None and 'semester' not in available_columns:
                        return jsonify({'success': False, 'error': 'Semester is required. Provide a semester field in the file or pass form field `semester`.'}), 400

            for row in chunk:
                code = str(row.get('code', '')).strip()
                name = str(row.get('name', '')).strip()
                if not code or not name:
                    continue

                # Resolve semester: row overrides form default if present and valid
                sem_val = row.get('semester', '')
                if sem_val not in (None, ''):
                    try:
                        semester = int(float(sem_val))  # handle excel numeric
                    except (TypeError, ValueError):
                        semester = default_semester
                else:
                    semester = default_semester

                if semester is None:
                    return jsonify({'success': False, 'error': f"Row for code '{code}' missing semester and no default provided."}), 400

                # Validate semester range
                if semester < 1 or semester > branch.total_semesters:
                    return jsonify({'success': False, 'error': f"Semester must be between 1 and {branch.total_semesters}. Invalid for code '{code}'."}), 400

                course_type_raw = str(row.get('course_type', row.get('type', 'theory'))).lower().strip()
                course_type = 'practical' if 'prac' in course_type_raw else 'theory'
                credits = parse_int(row.get('credits'), 0)
                hpw = parse_int(row.get('hours_per_week'), 3)
                tags_raw = row.get('required_room_tags') or row.get('room_tags') or ''
                tags = ','.join(t.strip() for t in str(tags_raw).split(',') if t and t.strip())

                existing_course = existing.get(code)
                if existing_course:
                    # Update in-place
                    existing_course.name = name
                    existing_course.course_type = course_type
                    existing_course.credits = credits
                    existing_course.hours_per_week = hpw
                    existing_course.program = branch.program
                    existing_course.branch = branch.name
                    existing_course.semester = semester
                    existing_course.required_room_tags = tags
                    db.session.add(existing_course)
                    updated += 1
                else:
                    new_course = Course(
                        code=code,
                        name=name,
                        course_type=course_type,
                        credits=credits,
                        hours_per_week=hpw,
                        program=branch.program,
                        branch=branch.name,
                        semester=semester,
                        required_room_tags=tags
                    )
                    db.session.add(new_course)
                    existing[code] = new_course
                    created += 1

            # Commit per chunk
            db.session.commit()

        return jsonify({'success': True, 'created': created, 'updated': updated})
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500


# Faculty Management
@app.route('/faculty')
@login_required
def faculty():
    user = get_current_user()
    faculty_list = Faculty.query.all()
    courses_list = Course.query.all()
    return render_template('faculty.html', faculty=faculty_list, courses=courses_list, user=user)

@app.route('/faculty/add', methods=['POST'])
@admin_required
def add_faculty():
    data = request.json
    try:
        # Admin adds faculty: do not enforce 70% availability validation here.
        # Admin-provided availability (if any) will be stored as-is; missing
        # availability defaults to an empty JSON object '{}' which scheduler
        # treats as 100% available.
        faculty, generated_password = create_faculty_profile(data)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    db.session.commit()
    response = {'success': True, 'id': faculty.id}
    if generated_password:
        response['generated_password'] = generated_password
    return jsonify(response)

@app.route('/api/faculty/<int:faculty_id>/update', methods=['POST'])
@admin_required
def update_faculty(faculty_id):
    """Update faculty details"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty:
            return jsonify({'success': False, 'error': 'Faculty not found'}), 404
        
        data = request.json
        
        if 'name' in data:
            faculty.name = data['name']
        if 'email' in data:
            faculty.email = data['email']
        if 'username' in data:
            faculty.username = data['username']
        if 'expertise' in data:
            faculty.expertise = data['expertise']
        if 'min_hours_per_week' in data:
            faculty.min_hours_per_week = int(data['min_hours_per_week'])
        if 'max_hours_per_week' in data:
            faculty.max_hours_per_week = int(data['max_hours_per_week'])
        
        faculty.save()
        invalidate_cache('faculty')
        
        return jsonify({'success': True, 'message': 'Faculty updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/faculty/<int:faculty_id>/delete', methods=['POST'])
@admin_required
def delete_faculty(faculty_id):
    faculty = Faculty.query.get_or_404(faculty_id)
    linked_user = User.query.get(faculty.user_id) if faculty.user_id else None
    # Remove timetable entries referencing this faculty to avoid FK issues
    TimetableEntry.query.filter_by(faculty_id=faculty.id).delete(synchronize_session=False)
    db.session.delete(faculty)
    if linked_user and linked_user.role == 'teacher':
        db.session.delete(linked_user)
    db.session.commit()
    invalidate_cache('faculty')
    return jsonify({'success': True})

@app.route('/faculty/import', methods=['POST'])
@admin_required
def import_faculty():
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    try:
        import time
        from background_tasks import queue_user_creation_task
        from models import get_next_id_batch
        from pymongo import UpdateOne
        
        start_time = time.time()
        chunks_generator = process_upload_stream(upload, chunk_size=1000)
        
        DEFAULT_TEMP_PASSWORD = "ChangeMe@123"
        
        # ============================================
        # PHASE 1: IN-MEMORY VALIDATION - Pre-fetch metadata
        # ============================================
        print("[Faculty Import] Pre-fetching metadata for validation...")
        metadata_start = time.time()
        
        # Fetch valid departments from Branch collection
        valid_departments = {
            doc['name'].lower().strip() 
            for doc in db._db['branch'].find({}, {'name': 1})
            if doc.get('name')
        }
        
        # Fetch valid programs and branches for validation
        valid_programs = {
            doc['program'].lower().strip() 
            for doc in db._db['course'].find({}, {'program': 1})
            if doc.get('program')
        }
        
        valid_branches = {
            doc['branch'].lower().strip() 
            for doc in db._db['course'].find({}, {'branch': 1})
            if doc.get('branch')
        }
        
        # Build program-to-branches mapping for validation
        program_branches_map = {}
        for doc in db._db['course'].find({}, {'program': 1, 'branch': 1}):
            program = doc.get('program', '').lower().strip()
            branch = doc.get('branch', '').lower().strip()
            if program and branch:
                if program not in program_branches_map:
                    program_branches_map[program] = set()
                program_branches_map[program].add(branch)
        
        print(f"[Faculty Import] Metadata loaded in {time.time() - metadata_start:.2f}s")
        
        # ============================================
        # PHASE 2: ATOMIC BULK OPERATIONS - Pre-fetch existing data
        # ============================================
        print("[Faculty Import] Pre-fetching existing faculty...")
        prefetch_start = time.time()
        
        # Load into sets for O(1) lookups (NO model instantiation!)
        existing_usernames = {
            doc['username'] 
            for doc in db._db['faculty'].find({}, {'username': 1})
            if doc.get('username')
        }
        
        existing_emails = {
            doc['email'] 
            for doc in db._db['faculty'].find({'email': {'$exists': True}}, {'email': 1})
            if doc.get('email')
        }
        
        print(f"[Faculty Import] Pre-fetch completed in {time.time() - prefetch_start:.2f}s")
        print(f"[Faculty Import] Found {len(existing_usernames)} existing faculty")
        
        created = 0
        updated = 0
        skipped = 0
        validation_errors = []
        pending_user_creation = []
        
        # Lists for bulk operations (RAW DICTIONARIES!)
        faculty_to_insert = []
        faculty_to_update = []
        
        # Get starting ID for batch
        starting_id = get_next_id_batch(db._db, 'faculty', 1000)
        current_id = starting_id
        
        total_rows_processed = 0
        
        for chunk_idx, chunk in enumerate(chunks_generator):
            # Validate columns on first chunk
            if chunk_idx == 0 and chunk:
                available_columns = set(chunk[0].keys())
                has_name = 'name' in available_columns or 'full name' in available_columns
                if not has_name or 'username' not in available_columns:
                    missing = []
                    if not has_name: missing.append('name/full name')
                    if 'username' not in available_columns: missing.append('username')
                    return jsonify({
                        'success': False,
                        'error': f'Missing columns: {", ".join(missing)}'
                    }), 400
            
            # ============================================
            # BUILD RAW DICTIONARIES (NO BaseModel!)
            # ============================================
            for row_idx, row in enumerate(chunk):
                total_rows_processed += 1
                name = str(row.get('full name', row.get('name', ''))).strip()
                if not name:
                    skipped += 1
                    continue
                    
                username = str(row.get('username', '')).strip()
                email = str(row.get('email', '')).strip()
                expertise = normalize_comma_list(row.get('expertise', ''))
                
                departments_value = row.get('departments', [])
                departments_list = process_departments_field(departments_value)
                
                # In-memory validation of departments (only if provided and not empty)
                # IMPORTANT: Make validation OPTIONAL - warn but don't skip if validation data is missing
                if departments_list and valid_departments:  # Only validate if BOTH are provided
                    invalid_departments = []
                    for dept in departments_list:
                        dept_lower = dept.lower().strip()
                        if dept_lower and dept_lower not in valid_departments:
                            invalid_departments.append(dept)
                    
                    if invalid_departments:
                        # WARN but don't skip - validation is informational only
                        print(f"[Faculty Import] WARNING Row {row_idx + 1}: Departments not in database: {', '.join(invalid_departments)}")
                        # Don't skip - allow import to continue
                elif departments_list and not valid_departments:
                    # No departments in database - skip validation
                    print(f"[Faculty Import] INFO: Skipping department validation (no departments in database)")
                
                # Validate program (only if provided and validation data exists)
                program = str(row.get('program', '')).strip().lower()
                if program and valid_programs:  # Only validate if BOTH are provided
                    if program not in valid_programs:
                        # WARN but don't skip
                        print(f"[Faculty Import] WARNING Row {row_idx + 1}: Program '{program}' not in database")
                elif program and not valid_programs:
                    print(f"[Faculty Import] INFO: Skipping program validation (no programs in database)")
                
                # Validate branch (only if provided and validation data exists)
                branch = str(row.get('branch', '')).strip().lower()
                if branch and valid_branches:  # Only validate if BOTH are provided
                    if branch not in valid_branches:
                        # WARN but don't skip
                        print(f"[Faculty Import] WARNING Row {row_idx + 1}: Branch '{branch}' not in database")
                elif branch and not valid_branches:
                    print(f"[Faculty Import] INFO: Skipping branch validation (no branches in database)")
                
                # Validate program-branch combination (only if both provided and validation data exists)
                if program and branch and program_branches_map:
                    valid_branches_for_program = program_branches_map.get(program, set())
                    if valid_branches_for_program and branch not in valid_branches_for_program:
                        # WARN but don't skip
                        print(f"[Faculty Import] WARNING Row {row_idx + 1}: Branch '{branch}' may not be valid for program '{program}'")
                
                min_hours = parse_int(row.get('min_hours', row.get('min_hours_per_week')), 4)
                max_hours = parse_int(row.get('max_hours', row.get('max_hours_per_week')), 16)

                raw_availability = row.get('availability (optional)', row.get('availability', '{}'))
                if isinstance(raw_availability, (dict, list)):
                    raw_availability = json.dumps(raw_availability)
                elif not isinstance(raw_availability, str) or not raw_availability.strip():
                    raw_availability = '{}'
                
                # O(1) lookup in set!
                if username in existing_usernames:
                    # Prepare for bulk update
                    faculty_to_update.append({
                        'username': username,
                        'name': name,
                        'email': email,
                        'expertise': ','.join(expertise),
                        'departments': departments_list,
                        'min_hours_per_week': min_hours,
                        'max_hours_per_week': max_hours,
                        'availability': raw_availability
                    })
                    updated += 1
                else:
                    # ============================================
                    # RAW DICTIONARY (NO BaseModel!)
                    # ============================================
                    faculty_to_insert.append({
                        'id': current_id,  # Increment in memory!
                        'name': name,
                        'email': email,
                        'expertise': ','.join(expertise),
                        'availability': raw_availability,
                        'username': username if username else None,  # Allow empty username
                        'min_hours_per_week': min_hours,
                        'max_hours_per_week': max_hours,
                        'user_id': None,  # Will be set by background task
                        'departments': departments_list
                    })
                    
                    if username:  # Only add to set if username exists
                        existing_usernames.add(username)  # Prevent duplicates
                    created += 1
                    
                    # Queue for background processing ONLY if username exists
                    if username:
                        pending_user_creation.append({
                            'username': username,
                            'name': name,
                            'email': email or f'{username}@faculty.local',
                            'entity_id': current_id
                        })
                    
                    current_id += 1  # Increment in memory!
        
        # ============================================
        # PHASE 2: ATOMIC BULK INSERTION (ONE database call!)
        # ============================================
        print(f"[Faculty Import] ========== BULK OPERATIONS DEBUG ==========")
        print(f"[Faculty Import] Total rows processed: {total_rows_processed}")
        print(f"[Faculty Import] Created counter: {created}")
        print(f"[Faculty Import] Updated counter: {updated}")
        print(f"[Faculty Import] Skipped counter: {skipped}")
        print(f"[Faculty Import] faculty_to_insert list size: {len(faculty_to_insert)}")
        print(f"[Faculty Import] faculty_to_update list size: {len(faculty_to_update)}")
        
        # Show sample of first faculty to insert (if any)
        if faculty_to_insert:
            print(f"[Faculty Import] Sample faculty to insert: {faculty_to_insert[0]}")
        else:
            print(f"[Faculty Import] WARNING: faculty_to_insert is EMPTY!")
        
        print(f"[Faculty Import] Performing bulk operations...")
        bulk_start = time.time()
        
        # Bulk insert (ONE call, NO BaseModel overhead!)
        if faculty_to_insert:
            try:
                result = db._db['faculty'].insert_many(faculty_to_insert, ordered=False)
                print(f"[Faculty Import] ✓ Successfully inserted {len(result.inserted_ids)} faculty in {time.time() - bulk_start:.2f}s")
            except Exception as e:
                print(f"[Faculty Import] ✗ Error during bulk insert: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"[Faculty Import] ⚠️ Skipping bulk insert (faculty_to_insert is empty)")
        
        # Bulk update (ONE call!)
        if faculty_to_update:
            try:
                bulk_ops = [
                    UpdateOne(
                        {'username': fac['username']},
                        {'$set': {
                            'name': fac['name'],
                            'email': fac['email'],
                            'expertise': fac['expertise'],
                            'departments': fac['departments'],
                            'min_hours_per_week': fac['min_hours_per_week'],
                            'max_hours_per_week': fac['max_hours_per_week'],
                            'availability': fac['availability']
                        }}
                    )
                    for fac in faculty_to_update
                ]
                result = db._db['faculty'].bulk_write(bulk_ops, ordered=False)
                print(f"[Faculty Import] Updated {result.modified_count} faculty")
            except Exception as e:
                print(f"[Faculty Import] Error during bulk update: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"[Faculty Import] No faculty to update (faculty_to_update is empty)")
        
        bulk_time = time.time() - bulk_start
        elapsed_time = time.time() - start_time
        
        # ============================================
        # PHASE 3: BACKGROUND HASHING - Queue user creation
        # ============================================
        if pending_user_creation:
            queue_user_creation_task(
                user_data_list=pending_user_creation,
                default_password=DEFAULT_TEMP_PASSWORD,
                user_type='faculty'
            )
            print(f"[Faculty Import] Queued {len(pending_user_creation)} accounts for background creation")
        
        # Invalidate cache to ensure fresh data is shown
        invalidate_cache('faculty')
        
        print(f"[Faculty Import] TOTAL: {elapsed_time:.2f}s - Created: {created}, Updated: {updated}, Skipped: {skipped}")
        print(f"[Faculty Import] Performance: Metadata={time.time() - metadata_start:.2f}s, Bulk={bulk_time:.2f}s")
        print(f"[Faculty Import] Inserted: {len(faculty_to_insert)}, Updated: {len(faculty_to_update)}")
        print(f"[Faculty Import] Total rows processed: {total_rows_processed}")
        
        # Debug: Check if data was actually inserted
        if faculty_to_insert:
            # Verify insertion by checking one record
            sample_username = faculty_to_insert[0].get('username')
            if sample_username:
                verify = db._db['faculty'].find_one({'username': sample_username})
                if verify:
                    print(f"[Faculty Import] ✓ Verified: Sample record with username '{sample_username}' exists in database")
                else:
                    print(f"[Faculty Import] ✗ WARNING: Sample record with username '{sample_username}' NOT found in database!")
        
        # Build response message
        if created == 0 and updated == 0:
            # No data was imported - this might indicate an issue
            message = f"⚠️ No faculty were imported. "
            if skipped > 0:
                message += f"Skipped: {skipped} rows (validation errors). "
            if total_rows_processed == 0:
                message += "No rows were processed from the file."
            else:
                message += f"Processed {total_rows_processed} rows but none were imported. Check validation errors."
            if validation_errors:
                message += f"\n\nFirst few errors:\n" + "\n".join(validation_errors[:5])
        else:
            message = f"Import successful! Created: {created}, Updated: {updated}"
            if skipped > 0:
                message += f", Skipped: {skipped} (validation errors)"
            if pending_user_creation:
                message += f"\n\n⏳ Creating {len(pending_user_creation)} login accounts in background...\n"
                message += f"Default password: '{DEFAULT_TEMP_PASSWORD}'\n"
                message += f"Accounts will be ready in ~5 seconds."
        
        response = {
            'success': True, 
            'created': created, 
            'updated': updated,
            'skipped': skipped,
            'time_taken': f"{elapsed_time:.2f}s",
            'message': message,
            'background_tasks': len(pending_user_creation),
            'default_password': DEFAULT_TEMP_PASSWORD if pending_user_creation else None,
            'total_rows_processed': total_rows_processed
        }
        
        if validation_errors:
            response['validation_errors'] = validation_errors[:10]  # First 10 errors
            response['total_errors'] = len(validation_errors)
        
        return jsonify(response)
    
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        print(f"[Faculty Import Error] {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500


@app.route('/faculty/delete-all', methods=['POST'])
@admin_required
def delete_all_faculty():
    """Delete all faculty members and their linked user accounts"""
    try:
        # Count faculty before deletion
        deleted_count = Faculty.query.count()
        
        # Get all faculty user IDs for bulk user deletion
        faculty_user_ids = [f.user_id for f in Faculty.query.all() if f.user_id]
        
        # Delete linked teacher users one by one (MongoDB doesn't support filter().in_())
        if faculty_user_ids:
            for user_id in faculty_user_ids:
                user = User.query.filter_by(id=user_id, role='teacher').first()
                if user:
                    db.session.delete(user)
        
        # Bulk delete timetable entries
        TimetableEntry.query.delete(synchronize_session=False)
        
        # Bulk delete all faculty
        Faculty.query.delete(synchronize_session=False)
        
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

# Room Management
@app.route('/rooms')
@login_required
def rooms():
    user = get_current_user()
    rooms_list = Room.query.all()
    return render_template('rooms.html', rooms=rooms_list, user=user)

@app.route('/rooms/add', methods=['POST'])
@admin_required
def add_room():
    data = request.json or {}

    # Validate name
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Room name is required.'}), 400

    # Prevent duplicate room names with a friendly error
    existing = Room.query.filter_by(name=name).first()
    if existing:
        return jsonify({'success': False, 'error': f'A room named "{name}" already exists.'}), 400

    # Parse capacity safely
    try:
        capacity = int(data.get('capacity')) if data.get('capacity') not in (None, '') else 0
    except (TypeError, ValueError):
        capacity = 0

    room = Room(
        name=name,
        capacity=capacity,
        room_type=data.get('type', ''),
        equipment=data.get('equipment', ''),
        tags=','.join(tag.strip() for tag in data.get('tags', '').split(',') if tag.strip())
    )
    db.session.add(room)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'A room named "{name}" already exists.'}), 400

    return jsonify({'success': True, 'id': room.id})

@app.route('/rooms/<int:room_id>/delete', methods=['POST'])
@admin_required
def delete_room(room_id):
    room = Room.query.get_or_404(room_id)
    db.session.delete(room)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/rooms/import', methods=['POST'])
@admin_required
def import_rooms():
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    try:
        chunks_generator = process_upload_stream(upload, chunk_size=1000)
        required_columns = {'name', 'capacity'}
        
        # Pre-fetch existing rooms to avoid N+1 queries
        existing_rooms = {r.name: r for r in Room.query.all()}
        
        created, updated = 0, 0
        
        for chunk_idx, chunk in enumerate(chunks_generator):
            if chunk_idx == 0 and chunk:
                available_columns = set(chunk[0].keys())
                missing = get_missing_columns(available_columns, required_columns)
                if missing:
                    return jsonify({
                        'success': False,
                        'error': f'Missing columns: {", ".join(sorted(missing))}'
                    }), 400
            
            for row in chunk:
                name = str(row.get('name', '')).strip()
                if not name:
                    continue
                
                room = existing_rooms.get(name)
                capacity = parse_int(row.get('capacity'), 0)
                room_type = str(row.get('room_type', 'classroom')).strip()
                equipment = str(row.get('equipment', '')).strip()
                tags = ','.join(tag.strip() for tag in str(row.get('tags', '')).split(',') if tag.strip())

                payload = {
                    'name': name,
                    'capacity': capacity,
                    'room_type': room_type,
                    'equipment': equipment,
                    'tags': tags
                }

                if room:
                    room.name = payload['name']
                    room.capacity = payload['capacity']
                    room.room_type = payload['room_type']
                    room.equipment = payload['equipment']
                    room.tags = payload['tags']
                    updated += 1
                    db.session.add(room)
                else:
                    room = Room(
                        name=payload['name'],
                        capacity=payload['capacity'],
                        room_type=payload['room_type'],
                        equipment=payload['equipment'],
                        tags=payload['tags']
                    )
                    existing_rooms[name] = room
                    db.session.add(room)
                    created += 1
            
            db.session.commit()
        
        return jsonify({'success': True, 'created': created, 'updated': updated})
    
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500

@app.route('/rooms/delete-all', methods=['POST'])
@admin_required
def delete_all_rooms():
    """Delete all rooms"""
    try:
        # Count rooms before deletion
        deleted_count = Room.query.count()
        
        # Bulk delete timetable entries
        TimetableEntry.query.delete(synchronize_session=False)
        
        # Bulk delete all rooms
        Room.query.delete(synchronize_session=False)
        
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

# Student Management
@app.route('/students')
@login_required
def students():
    user = get_current_user()
    students_list = Student.query.all()
    courses_list = Course.query.all()
    return render_template('students.html', students=students_list, courses=courses_list, user=user)

@app.route('/students/add', methods=['POST'])
@admin_required
def add_student():
    data = request.json or {}
    # Basic fields
    name = (data.get('name') or '').strip()
    student_id = (data.get('student_id') or '').strip()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    courses = data.get('courses', []) or []

    if not name or not student_id:
        return jsonify({'success': False, 'error': 'name and student_id are required'}), 400
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'username and password are required'}), 400

    # Create/Link student user account
    user = None
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        if existing_user.role not in ('student', 'teacher', 'admin'):
            # Unknown role, but still allow setting to student
            existing_user.role = 'student'
        elif existing_user.role != 'student':
            return jsonify({'success': False, 'error': 'Username already used by another account'}), 400
        existing_user.name = name
        existing_user.set_password(password)
        user = existing_user
    else:
        # Create new student user
        email = f"{username}@student.local"
        # Ensure email uniqueness
        if User.query.filter_by(email=email).first():
            email = f"{username}+{secrets.token_hex(3)}@student.local"
        user = User(username=username, email=email, role='student', name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()

    student = Student(
        name=name,
        student_id=student_id,
        enrolled_courses=','.join(courses),
        username=username,
        user_id=user.id
    )
    db.session.add(student)
    db.session.commit()
    response = {'success': True, 'id': student.id}
    return jsonify(response)

@app.route('/api/student/<int:student_id>/update', methods=['POST'])
@admin_required
def update_student(student_id):
    """Update student details"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'success': False, 'error': 'Student not found'}), 404
        
        data = request.json
        
        if 'student_id' in data:
            student.student_id = data['student_id']
        if 'name' in data:
            student.name = data['name']
        
        student.save()
        invalidate_cache('students')
        
        return jsonify({'success': True, 'message': 'Student updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/students/<int:student_id>/delete', methods=['POST'])
@admin_required
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    # Remove linked user account if it exists and is a student
    if getattr(student, 'user_id', None):
        u = User.query.get(student.user_id)
        if u and u.role == 'student':
            db.session.delete(u)
    db.session.delete(student)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/students/import', methods=['POST'])
@admin_required
def import_students():
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    try:
        import time
        from background_tasks import queue_user_creation_task
        from models import get_next_id_batch
        from pymongo import UpdateOne
        
        start_time = time.time()
        chunks_generator = process_upload_stream(upload, chunk_size=1000)
        required = {'student_id', 'name'}
        
        DEFAULT_TEMP_PASSWORD = "Student@123"
        
        # ============================================
        # PHASE 1: IN-MEMORY VALIDATION - Pre-fetch existing data
        # ============================================
        print("[Student Import] Pre-fetching existing data...")
        prefetch_start = time.time()
        
        # Load into sets and mappings for O(1) lookups (NO model instantiation!)
        existing_student_ids = set()
        student_id_to_db_id = {}  # Map student_id to database id for linking users
        existing_student_usernames = set()
        
        for doc in db._db['student'].find({}, {'student_id': 1, 'id': 1, 'username': 1}):
            if doc.get('student_id'):
                existing_student_ids.add(doc['student_id'])
                student_id_to_db_id[doc['student_id']] = doc.get('id')
            if doc.get('username'):
                existing_student_usernames.add(doc['username'])
        
        existing_user_usernames = {
            doc['username'] 
            for doc in db._db['user'].find({}, {'username': 1})
            if doc.get('username')
        }
        
        # Map username to user_id for existing users
        username_to_user_id = {
            doc['username']: doc.get('id')
            for doc in db._db['user'].find({}, {'username': 1, 'id': 1})
            if doc.get('username')
        }
        
        print(f"[Student Import] Pre-fetch completed in {time.time() - prefetch_start:.2f}s")
        print(f"[Student Import] Found {len(existing_student_ids)} existing students")
        
        created = 0
        updated = 0
        pending_user_creation = []
        
        # Lists for bulk operations (RAW DICTIONARIES!)
        students_to_insert = []
        students_to_update = []
        users_to_update = []  # For existing users that need role updates
        
        # Get starting IDs for batch
        student_starting_id = get_next_id_batch(db._db, 'student', 1000)
        current_student_id = student_starting_id
        
        for chunk_idx, chunk in enumerate(chunks_generator):
            if chunk_idx == 0 and chunk:
                available_columns = set(chunk[0].keys())
                missing = get_missing_columns(available_columns, required)
                if missing:
                    return jsonify({
                        'success': False,
                        'error': f'Missing columns: {", ".join(sorted(missing))}'
                    }), 400
            
            # ============================================
            # BUILD RAW DICTIONARIES (NO BaseModel!)
            # ============================================
            for row in chunk:
                student_id = str(row.get('student_id', '')).strip()
                if not student_id:
                    continue
                
                name = str(row.get('name', '')).strip()
                enrolled_courses = str(row.get('enrolled_courses', '')).strip()
                username = str(row.get('username', '')).strip()
                program = normalize_string(row.get('program', ''))
                branch = normalize_string(row.get('branch', ''))
                semester = parse_int(row.get('semester'), 0)

                
                # O(1) lookup in set!
                if student_id in existing_student_ids:
                    # Prepare for bulk update
                    student_db_id = student_id_to_db_id.get(student_id)
                    update_data = {
                        'student_id': student_id,
                        'name': name,
                        'enrolled_courses': enrolled_courses,
                        'program': program,
                        'branch': branch,
                        'semester': semester
                    }
                    
                    # Handle username/user_id if provided
                    if username:
                        update_data['username'] = username
                        # If user exists, link it immediately
                        if username in existing_user_usernames:
                            user_id = username_to_user_id.get(username)
                            if user_id:
                                update_data['user_id'] = user_id
                            # Update user role if needed
                            users_to_update.append({
                                'username': username,
                                'name': name,
                                'role': 'student'
                            })
                        # If user doesn't exist, queue for background creation
                        elif username not in existing_student_usernames:
                            pending_user_creation.append({
                                'username': username,
                                'name': name,
                                'email': f'{username}@student.local',
                                'entity_id': student_db_id  # Link to existing student
                            })
                    
                    students_to_update.append(update_data)
                    updated += 1
                else:
                    # ============================================
                    # RAW DICTIONARY (NO BaseModel!)
                    # ============================================
                    student_dict = {
                        'id': current_student_id,
                        'student_id': student_id,
                        'name': name,
                        'enrolled_courses': enrolled_courses,
                        'username': username or None,
                        'user_id': None,  # Will be set by background task
                        'program': program,
                        'branch': branch,
                        'semester': semester
                    }
                    
                    students_to_insert.append(student_dict)
                    existing_student_ids.add(student_id)  # Prevent duplicates
                    student_id_to_db_id[student_id] = current_student_id  # Track for potential updates
                    created += 1
                    
                    # Queue user account creation for background processing
                    if username and username not in existing_user_usernames:
                        pending_user_creation.append({
                            'username': username,
                            'name': name,
                            'email': f'{username}@student.local',
                            'entity_id': current_student_id
                        })
                    
                    current_student_id += 1  # Increment in memory!
        
        # ============================================
        # PHASE 2: ATOMIC BULK INSERTION (ONE database call!)
        # ============================================
        print(f"[Student Import] Performing bulk operations...")
        bulk_start = time.time()
        
        # Bulk insert (ONE call, NO BaseModel overhead!)
        if students_to_insert:
            db._db['student'].insert_many(students_to_insert, ordered=False)
            print(f"[Student Import] Inserted {len(students_to_insert)} students in {time.time() - bulk_start:.2f}s")
        
        # Bulk update (ONE call!)
        if students_to_update:
            bulk_ops = []
            for student_update in students_to_update:
                student_id_val = student_update.pop('student_id')
                bulk_ops.append(
                    UpdateOne(
                        {'student_id': student_id_val},
                        {'$set': student_update}
                    )
                )
            db._db['student'].bulk_write(bulk_ops, ordered=False)
            print(f"[Student Import] Updated {len(students_to_update)} students")
        
        # Update existing users (if any)
        if users_to_update:
            for user_update in users_to_update:
                db._db['user'].update_one(
                    {'username': user_update['username']},
                    {'$set': {
                        'name': user_update['name'],
                        'role': user_update['role']
                    }}
                )
            print(f"[Student Import] Updated {len(users_to_update)} existing users")
        
        bulk_time = time.time() - bulk_start
        elapsed_time = time.time() - start_time
        
        # ============================================
        # PHASE 3: BACKGROUND HASHING - Queue user creation
        # ============================================
        if pending_user_creation:
            queue_user_creation_task(
                user_data_list=pending_user_creation,
                default_password=DEFAULT_TEMP_PASSWORD,
                user_type='student'
            )
            print(f"[Student Import] Queued {len(pending_user_creation)} accounts for background creation")
        
        print(f"[Student Import] TOTAL: {elapsed_time:.2f}s - Created: {created}, Updated: {updated}")
        print(f"[Student Import] Performance: Pre-fetch={time.time() - prefetch_start:.2f}s, Bulk={bulk_time:.2f}s")
        
        # Build response message
        message = f"Import successful! Created: {created}, Updated: {updated}"
        if pending_user_creation:
            message += f"\n\n⏳ Creating {len(pending_user_creation)} login accounts in background...\n"
            message += f"Default password: '{DEFAULT_TEMP_PASSWORD}'\n"
            message += f"Accounts will be ready in ~5 seconds."
        
        return jsonify({
            'success': True, 
            'created': created, 
            'updated': updated,
            'time_taken': f"{elapsed_time:.2f}s",
            'message': message,
            'background_tasks': len(pending_user_creation),
            'default_password': DEFAULT_TEMP_PASSWORD if pending_user_creation else None
        })

    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        print(f"[Student Import Error] {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500

@app.route('/students/delete-all', methods=['POST'])
@admin_required
def delete_all_students():
    """Delete all students"""
    try:
        print("[DEBUG] Starting delete_all_students")
        
        # Count students before deletion
        print("[DEBUG] Counting students...")
        deleted_count = Student.query.count()
        print(f"[DEBUG] Found {deleted_count} students to delete")
        
        # Direct MongoDB deletion (bypasses ORM)
        print("[DEBUG] Deleting from student collection...")
        db._db['student'].delete_many({})
        print("[DEBUG] Student collection cleared")
        
        # Also delete student users
        print("[DEBUG] Deleting student users...")
        db._db['user'].delete_many({'role': 'student'})
        print("[DEBUG] Student users deleted")
        
        print(f"[DEBUG] Successfully deleted {deleted_count} students")
        return jsonify({'success': True, 'deleted': deleted_count, 'message': f'Successfully deleted {deleted_count} students'})
    except Exception as e:
        print(f"[ERROR] Delete all students failed at line {e.__traceback__.tb_lineno}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Student Course Selection API
@app.route('/student/dashboard')
@login_required
def student_dashboard():
    """Student dashboard for course selection"""
    user = get_current_user()
    
    # Check if user exists
    if not user:
        return redirect('/login')
    
    # Only students can access this page
    if user.role != 'student':
        return redirect('/')
    
    # Find the student record
    student = Student.query.filter_by(user_id=user.id).first()
    if not student:
        # Create a student record if it doesn't exist
        student = Student(
            student_id=user.username,
            name=user.username,
            user_id=user.id
        )
        student.save()
    
    import json
    student_json = json.dumps(student.to_dict(), default=str)
    
    return render_template('student_dashboard.html', student=student, student_json=student_json, user=user)

@app.route('/api/branches/programs')
@login_required
def get_programs():
    """Get all unique programs"""
    try:
        branches = Branch.query.all()
        programs = list(set(b.program for b in branches if b.program))
        programs.sort()
        return jsonify({'success': True, 'programs': programs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/branches/by-program/<program>')
@login_required
def get_branches_by_program(program):
    """Get all branches for a specific program"""
    try:
        branches = Branch.query.filter_by(program=program).all()
        branches_data = [{
            'code': b.code,
            'name': b.name,
            'program': b.program,
            'total_semesters': getattr(b, 'total_semesters', 8)
        } for b in branches]
        return jsonify({'success': True, 'branches': branches_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/student/update-profile', methods=['POST'])
@login_required
def update_student_profile():
    """Update student's program, branch, and semester"""
    try:
        user = get_current_user()
        if user.role != 'student':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        student = Student.query.filter_by(user_id=user.id).first()
        if not student:
            return jsonify({'success': False, 'error': 'Student record not found'}), 404
        
        data = request.json
        student.program = data.get('program')
        student.branch = data.get('branch')
        student.semester = int(data.get('semester', 1))
        student.save()
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/student/courses')
@login_required
def get_student_courses():
    """Get available courses for student's program, branch, and semester"""
    try:
        program = request.args.get('program')
        branch = request.args.get('branch')
        semester = int(request.args.get('semester', 1))
        
        if not program or not branch:
            return jsonify({'success': False, 'error': 'Program and branch are required'}), 400
        
        courses = Course.query.filter_by(
            program=program,
            branch=branch,
            semester=semester
        ).all()
        
        courses_data = [{
            'id': str(c.id),
            'code': c.code,
            'name': c.name,
            'credits': getattr(c, 'credits', 0),
            'course_type': getattr(c, 'course_type', 'theory'),
            'subject_type': getattr(c, 'subject_type', None),
            'hours_per_week': getattr(c, 'hours_per_week', 0),
            'is_mandatory': (semester in [1, 2] and getattr(c, 'subject_type', None) in ['major', 'minor'])
        } for c in courses]
        
        # For semester 1 and 2, get mandatory major/minor course codes
        mandatory_courses = []
        if semester in [1, 2]:
            mandatory_courses = [c['code'] for c in courses_data if c['is_mandatory']]
        
        return jsonify({
            'success': True, 
            'courses': courses_data,
            'mandatory_courses': mandatory_courses
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/student/enroll-courses', methods=['POST'])
@login_required
def enroll_student_courses():
    """Save student's enrolled courses"""
    try:
        user = get_current_user()
        if user.role != 'student':
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        student = Student.query.filter_by(user_id=user.id).first()
        if not student:
            return jsonify({'success': False, 'error': 'Student record not found'}), 404
        
        data = request.json
        courses = data.get('courses', [])
        
        # For semester 1 and 2, automatically add major and minor courses
        if student.semester in [1, 2] and student.program and student.branch:
            all_courses = Course.query.filter_by(
                program=student.program,
                branch=student.branch,
                semester=student.semester
            ).all()
            
            # Filter for major and minor courses
            mandatory_codes = [c.code for c in all_courses if getattr(c, 'subject_type', None) in ['major', 'minor']]
            
            # Merge mandatory courses with selected courses (remove duplicates)
            courses = list(set(mandatory_codes + courses))
        
        student.enrolled_courses = courses
        student.save()
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled in {len(courses)} courses',
            'enrolled_courses': courses
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Student Group Management
@app.route('/student-groups')
@admin_required
def student_groups():
    """
    Student Groups management page.
    
    Fetches available programs and branches from the Branch collection (single source of truth)
    and passes them to the template for server-side dropdown rendering.
    """
    user = get_current_user()
    
    # Fetch all student groups
    raw_groups = StudentGroup.query.all()
    groups = []
    for g in raw_groups:
        # Safely obtain batches list; some legacy records may have a mis-typed 'batche' field
        batches_raw = getattr(g, 'batches', None)
        if batches_raw is None:
            batches_raw = getattr(g, 'batche', None)
        batches = []
        if batches_raw:
            try:
                parsed = json.loads(batches_raw) if isinstance(batches_raw, str) else batches_raw
                if isinstance(parsed, list):
                    batches = parsed
            except Exception:
                batches = []
        groups.append({
            'id': getattr(g, 'id', None),
            'name': getattr(g, 'name', ''),
            'description': getattr(g, 'description', ''),
            'program': getattr(g, 'program', ''),
            'branch': getattr(g, 'branch', ''),
            'semester': getattr(g, 'semester', 0),
            'total_students': getattr(g, 'total_students', 0),
            'batches': batches
        })
    
    # ============================================
    # CRITICAL: Fetch from Course collection (single source of truth)
    # ============================================
    # This ensures student groups can only be created for programs/branches that exist in Courses
    try:
        # Use MongoDB distinct() to get unique programs and branches from Course collection
        available_programs = db._db['course'].distinct('program')
        available_branches = db._db['course'].distinct('branch')
        
        # Normalize to lowercase for consistency
        available_programs = sorted([str(p).lower().strip() for p in available_programs if p and str(p).strip()])
        available_branches = sorted([str(b).lower().strip() for b in available_branches if b and str(b).strip()])
        
        # Build program-to-branches mapping for dynamic filtering
        # This prevents creating "Mechanical" group for "Computer Science" program
        program_branches_map = {}
        for program in available_programs:
            # Get all branches for this program
            branches = db._db['course'].distinct('branch', {'program': program})
            program_branches_map[program] = sorted([str(b).lower().strip() for b in branches if b and str(b).strip()])
        
        print(f"[STUDENT_GROUPS] Programs: {available_programs}")
        print(f"[STUDENT_GROUPS] Branches: {available_branches}")
        print(f"[STUDENT_GROUPS] Program-Branch mapping: {program_branches_map}")
        
    except Exception as e:
        print(f"[STUDENT_GROUPS] Error fetching metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        available_programs = []
        available_branches = []
        program_branches_map = {}
    
    return render_template(
        'student_groups.html',
        groups=groups,
        user=user,
        available_programs=available_programs,
        available_branches=available_branches,
        program_branches_map=program_branches_map  # Pass mapping to frontend
    )

@app.route('/student-groups/add', methods=['POST'])
@admin_required
def add_student_group():
    try:
        data = request.json
        # Validate name
        name = (data.get('name') or '').strip()
        if not name:
            return jsonify({'success': False, 'error': 'Class name is required.'}), 400

        # Prevent duplicate names
        existing = StudentGroup.query.filter_by(name=name).first()
        if existing:
            return jsonify({'success': False, 'error': f'A class named "{name}" already exists.'}), 400
        
        batches = data.get('batches')
        if isinstance(batches, (list, dict)):
            batches_json = json.dumps(batches)
        else:
            batches_json = batches or None

        total_students = None
        try:
            total_students = int(data.get('total_students')) if data.get('total_students') not in (None, '') else None
        except (TypeError, ValueError):
            total_students = None

        from normalization import normalize_key
        program = normalize_key(data.get('program', ''))
        branch = normalize_key(data.get('branch', ''))
        semester = parse_int(data.get('semester') or data.get('current_semester'), 0)

        group = StudentGroup(
            name=name,
            description=data.get('description', ''),
            program=program,
            branch=branch,
            semester=semester,
            total_students=total_students,
            batches=batches_json
        )
        
        print(f"[DEBUG] Saving StudentGroup: {name} (Sem: {semester})")
        db.session.add(group)
        db.session.commit()
        invalidate_cache('student_groups')
        
        # Safe dictionary for response
        res_group = {
            'id': getattr(group, 'id', None),
            'name': getattr(group, 'name', ''),
            'description': getattr(group, 'description', ''),
            'program': getattr(group, 'program', ''),
            'branch': getattr(group, 'branch', ''),
            'semester': getattr(group, 'semester', 0),
            'total_students': getattr(group, 'total_students', 0)
        }
        
        return jsonify({
            'success': True,
            'message': 'Student group created successfully',
            'group': res_group
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] add_student_group failed: {str(e)}")
        traceback.print_exc()
        db.session.rollback()
        return jsonify({'success': False, 'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/student-group/<int:group_id>/update', methods=['POST'])
@admin_required
def update_student_group(group_id):
    """Update student group details"""
    try:
        group = StudentGroup.query.get(group_id)
        if not group:
            return jsonify({'success': False, 'error': 'Student group not found'}), 404
        
        data = request.json
        
        if 'name' in data:
            group.name = data['name']
        if 'description' in data:
            group.description = data['description']
        if 'total_students' in data:
            group.total_students = int(data['total_students'])
        
        group.save()
        invalidate_cache('student_groups')
        
        return jsonify({'success': True, 'message': 'Student group updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/student-groups/<int:group_id>/delete', methods=['POST'])
@admin_required
def delete_student_group(group_id):
    group = StudentGroup.query.get_or_404(group_id)
    db.session.delete(group)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/student-groups/import', methods=['POST'])
@admin_required
def import_student_groups():
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    try:
        chunks_generator = process_upload_stream(upload, chunk_size=1000)
        required = {'name'}
        
        existing_groups = {g.name: g for g in StudentGroup.query.all()}
        
        created, updated = 0, 0
        
        for chunk_idx, chunk in enumerate(chunks_generator):
            if chunk_idx == 0 and chunk:
                available_columns = set(chunk[0].keys())
                missing = get_missing_columns(available_columns, required)
                if missing:
                    return jsonify({
                        'success': False,
                        'error': f'Missing columns: {", ".join(sorted(missing))}'
                    }), 400
            
            for row in chunk:
                name = str(row.get('name', '')).strip()
                if not name: continue
                
                if not name: continue
                
                description = str(row.get('description', '')).strip()
                
                # NORMALIZE program and branch to canonical format (same as course imports)
                from normalization import normalize_key
                program_raw = str(row.get('program', '')).strip()
                program = normalize_key(program_raw) if program_raw else None
                
                branch_raw = str(row.get('branch', '')).strip()
                branch = normalize_key(branch_raw) if branch_raw else None
                
                semester = parse_int(row.get('semester'), 0)
                total_students = parse_int(row.get('total_students'), None)
                
                # Parse batches
                batches = []
                batches_col = row.get('batches', '')
                batches_students_col = row.get('batches_students', '')
                if batches_col or batches_students_col:
                    batch_names = [b.strip() for b in str(batches_col).split(',') if b.strip()]
                    batch_students = [s.strip() for s in str(batches_students_col).split(',') if s.strip()]
                    for i, batch_name in enumerate(batch_names):
                        students = batch_students[i] if i < len(batch_students) else ''
                        batches.append({'batch_name': batch_name, 'students': students})
                batches_json = json.dumps(batches) if batches else None
                
                group = existing_groups.get(name)
                if group:
                    group.name = name
                    group.description = description
                    group.program = program
                    group.branch = branch
                    group.semester = semester
                    group.total_students = total_students
                    if batches_json:
                        group.batches = batches_json
                    updated += 1
                    db.session.add(group)
                else:
                    group = StudentGroup(
                        name=name,
                        description=description,
                        program=program,
                        branch=branch,
                        semester=semester,
                        total_students=total_students,
                        batches=batches_json
                    )
                    existing_groups[name] = group
                    db.session.add(group)
                    created += 1
            
            db.session.commit()
            
        return jsonify({'success': True, 'created': created, 'updated': updated})

    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Import failed: {str(exc)}'}), 500

@app.route('/student-groups/delete-all', methods=['POST'])
@admin_required
def delete_all_student_groups():
    """Delete all student groups"""
    try:
        # Count student groups before deletion
        deleted_count = StudentGroup.query.count()
        
        # Bulk delete all student groups
        StudentGroup.query.delete(synchronize_session=False)
        
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

# Timetable Generation
@app.route('/timetable')
@login_required
def timetable():
    import time as time_module
    view_start = time_module.time()
    
    user = get_current_user()
    entries_query = TimetableEntry.query
    faculty_profile = None
    if user.role == 'teacher':
        faculty_profile = Faculty.query.filter_by(user_id=user.id).first()
        if faculty_profile:
            entries_query = entries_query.filter_by(faculty_id=faculty_profile.id)
        else:
            entries_query = entries_query.filter_by(faculty_id=-1)
    
    # OPTIMIZATION: Load time slots first (usually < 100 records)
    slots = TimeSlot.query.all()
    slots_dict = {s.id: s for s in slots}
    valid_slot_ids = set(slots_dict.keys())
    
    # Fetch all entries (MongoDB Query doesn't support SQLAlchemy-style filter)
    all_entries = entries_query.all()
    
    # Filter entries to only include those with valid time_slot_id
    entries = [e for e in all_entries if e.time_slot_id in valid_slot_ids]
    
    print(f"[TIMETABLE VIEW] Loading timetable for user: {user.username} (role: {user.role})")
    print(f"[TIMETABLE VIEW] Total entries in DB: {len(all_entries)}")
    print(f"[TIMETABLE VIEW] Entries with valid slots: {len(entries)}")
    print(f"[TIMETABLE VIEW] Found {len(slots)} time slots")

    # OPTIMIZATION: Selective Entity Loading (93% memory reduction)
    if entries:
        # Extract unique referenced IDs
        course_ids = sorted(list(set(e.course_id for e in entries if e.course_id)))
        faculty_ids = sorted(list(set(e.faculty_id for e in entries if e.faculty_id)))
        room_ids = sorted(list(set(e.room_id for e in entries if e.room_id)))
        
        # SINGLE BULK FETCH using MongoDB $in operator (one query per collection)
        # We use db._db directly for more control over the query
        courses_data = list(db._db['course'].find({'id': {'$in': course_ids}}))
        faculty_data = list(db._db['faculty'].find({'id': {'$in': faculty_ids}}))
        rooms_data = list(db._db['room'].find({'id': {'$in': room_ids}}))
        
        # Build lookup dictionaries (O(1) access)
        courses_dict = {c['id']: c for c in courses_data}
        faculty_dict = {f['id']: f for f in faculty_data}
        rooms_dict = {r['id']: r for r in rooms_data}
        
        # Convert to objects if templates expect object attribute access (e.g. course.name)
        # Since our models use dictionary-like storage, we can wrap them or ensure templates are compatible
        # For simplicity and speed, we preserve the current behavior but only load what's needed
    else:
        courses_dict = {}
        faculty_dict = {}
        rooms_dict = {}
    
    # Get break configurations
    breaks = BreakConfig.query.order_by(BreakConfig.after_period).all()
    break_map = {br.after_period: br for br in breaks}
    
    # Organize by day and period (one lecture per period per class is enforced by unique constraint)
    timetable_data = {}
    missing_refs = {'courses': set(), 'faculty': set(), 'rooms': set()}
    
    for entry in entries:
        # Skip entries with invalid time_slot_id
        if entry.time_slot_id not in slots_dict:
            print(f"[TIMETABLE VIEW] WARNING: Entry {entry.id} has invalid time_slot_id: {entry.time_slot_id}")
            continue
            
        slot = slots_dict[entry.time_slot_id]
        key = (slot.day, slot.period)
        
        # Check if all references exist
        course = courses_dict.get(entry.course_id)
        faculty = faculty_dict.get(entry.faculty_id)
        room = rooms_dict.get(entry.room_id)
        
        if not course:
            missing_refs['courses'].add(entry.course_id)
            print(f"[TIMETABLE VIEW] WARNING: Entry {entry.id} references missing course_id: {entry.course_id}")
        if not faculty:
            missing_refs['faculty'].add(entry.faculty_id)
            print(f"[TIMETABLE VIEW] WARNING: Entry {entry.id} references missing faculty_id: {entry.faculty_id}")
        if not room:
            missing_refs['rooms'].add(entry.room_id)
            print(f"[TIMETABLE VIEW] WARNING: Entry {entry.id} references missing room_id: {entry.room_id}")
        
        # Only add entry if all required references exist
        if course and faculty and room:
            if key not in timetable_data:
                timetable_data[key] = []
            timetable_data[key].append({
                'course': course,
                'faculty': faculty,
                'room': room,
                'slot': slot,
                'student_group': entry.student_group
            })
    
    # Log missing references summary
    if missing_refs['courses'] or missing_refs['faculty'] or missing_refs['rooms']:
        print(f"[TIMETABLE VIEW] Missing references - Courses: {missing_refs['courses']}, Faculty: {missing_refs['faculty']}, Rooms: {missing_refs['rooms']}")
    
    # Get days from period config or default
    period_config = PeriodConfig.query.first()
    if period_config:
        days = [d.strip() for d in period_config.days_of_week.split(',')]
    else:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # OPTIMIZATION: Use already loaded slots instead of re-querying
    periods = sorted(set(s.period for s in slots))
    
    view_time = time_module.time() - view_start
    print(f"[TIMETABLE VIEW] Loaded {len(entries)} entries in {view_time:.2f}s")
    
    teacher_availability = {}
    if faculty_profile and faculty_profile.availability:
        try:
            # Ensure availability is a string before parsing
            avail_data = faculty_profile.availability
            if isinstance(avail_data, str):
                teacher_availability = json.loads(avail_data)
            elif isinstance(avail_data, dict):
                teacher_availability = avail_data
            else:
                # If it's something else (float, int, etc.), reset to empty
                teacher_availability = {}
        except (json.JSONDecodeError, TypeError, ValueError):
            teacher_availability = {}

    # Provide data needed for manual assignments UI (serialize to plain dicts)
    raw_student_groups = StudentGroup.query.all()
    student_groups_list = []
    for g in raw_student_groups:
        batches_raw = getattr(g, 'batches', None)
        if batches_raw is None:
            batches_raw = getattr(g, 'batche', None)
        batches = []
        if batches_raw:
            try:
                parsed = json.loads(batches_raw) if isinstance(batches_raw, str) else batches_raw
                if isinstance(parsed, list):
                    batches = parsed
            except Exception:
                batches = []
        student_groups_list.append({
            'id': getattr(g, 'id', None),
            'name': getattr(g, 'name', ''),
            'description': getattr(g, 'description', ''),
            'total_students': getattr(g, 'total_students', 0),
            'batches': batches
        })

    courses_list = []
    for c in Course.query.all():
        courses_list.append({
            'id': getattr(c, 'id', None),
            'code': getattr(c, 'code', ''),
            'name': getattr(c, 'name', ''),
            'credits': getattr(c, 'credits', 0),
            'hours_per_week': getattr(c, 'hours_per_week', 0),
            'course_type': getattr(c, 'course_type', 'lecture')
        })

    faculty_list = []
    for f in Faculty.query.all():
        faculty_list.append({
            'id': getattr(f, 'id', None),
            'name': getattr(f, 'name', ''),
            'email': getattr(f, 'email', ''),
            'expertise': getattr(f, 'expertise', '')
        })

    rooms_list = []
    for r in Room.query.all():
        rooms_list.append({
            'id': getattr(r, 'id', None),
            'name': getattr(r, 'name', ''),
            'capacity': getattr(r, 'capacity', 0),
            'room_type': getattr(r, 'room_type', 'classroom'),
            'tags': getattr(r, 'tags', '')
        })
    
    # Build time_ranges dictionary from TimeSlot data
    # This will show the actual start-end time for each period based on admin settings
    time_ranges = {}
    for slot in slots:
        if slot.period not in time_ranges:
            # Format: "09:00 - 10:00"
            time_ranges[slot.period] = f"{slot.start_time} - {slot.end_time}"
    
    # Extract unique values for filter dropdowns
    all_courses = Course.query.all()
    all_groups = StudentGroup.query.all()
    
    # Get unique programs (from both courses and groups)
    from normalization import normalize_key
    programs_set = set()
    for c in all_courses:
        p = getattr(c, 'program', None)
        if p: programs_set.add(normalize_key(str(p)))
    for g in all_groups:
        p = getattr(g, 'program', None)
        if p: programs_set.add(normalize_key(str(p)))
    programs = sorted(list(programs_set))
    
    # Get unique branches
    branches_set = set()
    for c in all_courses:
        b = getattr(c, 'branch', None)
        if b: branches_set.add(normalize_key(str(b)))
    for g in all_groups:
        b = getattr(g, 'branch', None)
        if b: branches_set.add(normalize_key(str(b)))
    branches = sorted(list(branches_set))
    
    # Get unique semesters
    semesters_set = set()
    for c in all_courses:
        sem = getattr(c, 'semester', None)
        if sem is not None:
            try:
                semesters_set.add(int(sem))
            except (ValueError, TypeError): pass
    for g in all_groups:
        sem = getattr(g, 'semester', None)
        if sem is not None:
            try:
                semesters_set.add(int(sem))
            except (ValueError, TypeError): pass
    semesters = sorted(list(semesters_set))

    response = make_response(render_template('timetable.html', 
                         timetable_data=timetable_data,
                         days=days,
                         periods=periods,
                         break_map=break_map,
                         time_ranges=time_ranges,
                         user=user,
                         teacher_availability=teacher_availability,
                         student_groups=student_groups_list,
                         courses=courses_list,
                         faculty=faculty_list,
                         rooms=rooms_list,
                         programs=programs,
                         branches=branches,
                         semesters=semesters))
    
    # Disable caching - always show fresh data after generation
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response


@app.route('/timetable/entries')
@login_required
@cache_response(ttl=300, prefix='timetable_entries')
def timetable_entries():
    # Return entries for a given day to prefill manual assignment UI
    day = request.args.get('day')
    if not day:
        return jsonify({'entries': []})

    slots = TimeSlot.query.filter_by(day=day).all()
    slot_map = {s.id: s for s in slots}
    # Mongo-backed Query object does not support SQLAlchemy-style .filter or in_ operations.
    # Fetch all timetable entries and manually filter by matching time_slot_id.
    entries = [e for e in TimetableEntry.query.all() if e.time_slot_id in slot_map]
    result = []
    for e in entries:
        s = slot_map.get(e.time_slot_id)
        if not s:
            continue
        result.append({
            'period': s.period,
            'student_group': e.student_group,
            'course_id': e.course_id,
            'faculty_id': e.faculty_id,
            'room_id': e.room_id
        })
    return jsonify({'entries': result})

@app.route('/timetable/generate', methods=['POST'])
@admin_required
def generate_timetable():
    """
    Unified timetable generation endpoint.
    Attempts background generation via Celery if available, otherwise falls back to synchronous.
    """
    try:
        # Invalidate cache before generation
        invalidate_cache('timetable_view')
        invalidate_cache('timetable_entries')
        
        # Parse filters from request
        data = safe_get_request_data()
        filters = data.get('filters', {})  # Also handle nested filters if needed
        # Support both {program: '...'} and {filters: {program: '...'}}
        if not filters and any(k in data for k in ['program', 'branch', 'semester']):
            filters = data

        # NORMALIZE filter values to match database format
        from normalization import normalize_key
        if filters.get('program'):
            filters['program'] = normalize_key(filters['program'])
        if filters.get('branch'):
            filters['branch'] = normalize_key(filters['branch'])

        print(f"[GENERATE] Request received with filters (normalized): {filters}")
        
        # ============================================================================
        # PRE-FLIGHT VALIDATION CHECK
        # ============================================================================
        # Verify that every StudentGroup has corresponding Course entries
        # This prevents the generator from failing with data mismatches
        
        print("[GENERATE] Running pre-flight validation...")
        
        # Build query for student groups based on filters
        groups_query = StudentGroup.query
        if filters.get('program'):
            groups_query = groups_query.filter_by(program=filters['program'])
        if filters.get('branch'):
            groups_query = groups_query.filter_by(branch=filters['branch'])
        if filters.get('semester'):
            try:
                sem = int(filters['semester'])
                groups_query = groups_query.filter_by(semester=sem)
            except (ValueError, TypeError):
                pass
        
        target_groups = groups_query.all()
        
        if not target_groups:
            # ============================================================================
            # PRE-FLIGHT DIAGNOSTIC: Show available values for debugging
            # ============================================================================
            print("\n" + "=" * 80)
            print("[DIAGNOSTIC] NO STUDENT GROUPS FOUND - Debugging Information")
            print("=" * 80)
            
            # Show what was requested
            print(f"\n[REQUESTED] User Filter:")
            print(f"  Program: {filters.get('program', 'ANY')}")
            print(f"  Branch: {filters.get('branch', 'ANY')}")
            print(f"  Semester: {filters.get('semester', 'ANY')}")
            
            # Show what's available in database
            print(f"\n[AVAILABLE] Database Contains:")
            
            # Get all unique values from StudentGroup collection
            all_groups = StudentGroup.query.all()
            available_programs = set(g.program for g in all_groups if g.program)
            available_branches = set(g.branch for g in all_groups if g.branch)
            available_semesters = set(getattr(g, 'semester', 0) for g in all_groups if getattr(g, 'semester', 0))
            
            print(f"  Programs: {sorted(available_programs)}")
            print(f"  Branches: {sorted(available_branches)}")
            print(f"  Semesters: {sorted(available_semesters)}")
            
            # Show exact matches
            print(f"\n[MATCHING] Breakdown:")
            if filters.get('program'):
                matching_programs = [g.program for g in all_groups if g.program == filters['program']]
                print(f"  Groups with program='{filters['program']}': {len(matching_programs)}")
                if not matching_programs:
                    print(f"    WARNING: No groups found with program='{filters['program']}'")
                    print(f"    Available programs: {sorted(available_programs)}")
            
            if filters.get('branch'):
                matching_branches = [g.branch for g in all_groups if g.branch == filters['branch']]
                print(f"  Groups with branch='{filters['branch']}': {len(matching_branches)}")
                if not matching_branches:
                    print(f"    WARNING: No groups found with branch='{filters['branch']}'")
                    print(f"    Available branches: {sorted(available_branches)}")
            
            # Check for case/format mismatches
            print(f"\n[NORMALIZATION CHECK]:")
            from normalization import normalize_key
            
            if filters.get('program'):
                normalized_filter = normalize_key(filters['program'])
                if normalized_filter != filters['program']:
                    print(f"  WARNING: Filter program '{filters['program']}' normalized to '{normalized_filter}'")
                else:
                    print(f"  OK: Filter program '{filters['program']}' is already normalized")
            
            if filters.get('branch'):
                normalized_filter = normalize_key(filters['branch'])
                if normalized_filter != filters['branch']:
                    print(f"  WARNING: Filter branch '{filters['branch']}' normalized to '{normalized_filter}'")
                else:
                    print(f"  OK: Filter branch '{filters['branch']}' is already normalized")
            
            # Check database values for normalization
            print(f"\n[DATABASE NORMALIZATION CHECK]:")
            non_normalized_groups = []
            for g in all_groups:
                if g.program and normalize_key(g.program) != g.program:
                    non_normalized_groups.append(f"Group '{g.name}': program='{g.program}' (should be '{normalize_key(g.program)}')")
                if g.branch and normalize_key(g.branch) != g.branch:
                    non_normalized_groups.append(f"Group '{g.name}': branch='{g.branch}' (should be '{normalize_key(g.branch)}')")
            
            if non_normalized_groups:
                print(f"  WARNING: Found {len(non_normalized_groups)} non-normalized values:")
                for msg in non_normalized_groups[:5]:  # Show first 5
                    print(f"    - {msg}")
                if len(non_normalized_groups) > 5:
                    print(f"    ... and {len(non_normalized_groups) - 5} more")
                print(f"  SOLUTION: Run 'python migrate_normalization.py' to fix")
            else:
                print(f"  OK: All student group values are properly normalized")
            
            print("\n" + "=" * 80)
            print("[DIAGNOSTIC] End of debugging information")
            print("=" * 80 + "\n")
            
            return jsonify({
                'success': False,
                'status': 'error',
                'error': 'No student groups found matching the selected filters.',
                'details': {
                    'filters': filters,
                    'available_programs': sorted(list(available_programs)),
                    'available_branches': sorted(list(available_branches)),
                    'available_semesters': sorted(list(available_semesters)),
                    'suggestion': 'Check the diagnostic log in the console for detailed debugging information.'
                }
            }), 400
        
        print(f"[GENERATE] Found {len(target_groups)} student groups to validate")
        
        # Validate each group has corresponding courses
        validation_errors = []
        for group in target_groups:
            try:
                # Safely get group attributes
                group_name = getattr(group, 'name', 'Unknown Group')
                group_program = getattr(group, 'program', None)
                group_branch = getattr(group, 'branch', None)
                group_semester = getattr(group, 'semester', None)
                
                # Query courses for this group's program/branch/semester
                courses_query = Course.query
                
                if group_program:
                    courses_query = courses_query.filter_by(program=group_program)
                if group_branch:
                    courses_query = courses_query.filter_by(branch=group_branch)
                if group_semester:
                    courses_query = courses_query.filter_by(semester=group_semester)
                
                matching_courses = courses_query.all()
                
                if not matching_courses:
                    validation_errors.append({
                        'group': group_name,
                        'program': group_program or 'N/A',
                        'branch': group_branch or 'N/A',
                        'semester': group_semester or 'N/A',
                        'issue': 'No courses found'
                    })
                    
                    # ============================================================================
                    # DIAGNOSTIC: Show why no courses matched
                    # ============================================================================
                    print(f"\n[DIAGNOSTIC] No courses found for group '{group_name}'")
                    print(f"  Group criteria: program='{group_program}', branch='{group_branch}', semester={group_semester}")
                    
                    # Show available courses
                    all_courses = Course.query.all()
                    available_course_programs = set(c.program for c in all_courses if c.program)
                    available_course_branches = set(c.branch for c in all_courses if c.branch)
                    available_course_semesters = set(c.semester for c in all_courses if c.semester)
                    
                    print(f"  Available in Course collection:")
                    print(f"    Programs: {sorted(available_course_programs)}")
                    print(f"    Branches: {sorted(available_course_branches)}")
                    print(f"    Semesters: {sorted(available_course_semesters)}")
                    
                    # Check for exact matches
                    if group_program:
                        matching_program_courses = [c for c in all_courses if c.program == group_program]
                        print(f"    Courses with program='{group_program}': {len(matching_program_courses)}")
                    if group_branch:
                        matching_branch_courses = [c for c in all_courses if c.branch == group_branch]
                        print(f"    Courses with branch='{group_branch}': {len(matching_branch_courses)}")
                    if group_semester:
                        matching_semester_courses = [c for c in all_courses if c.semester == group_semester]
                        print(f"    Courses with semester={group_semester}: {len(matching_semester_courses)}")
                    
                    # Check normalization
                    from normalization import normalize_key
                    if group_program and normalize_key(group_program) != group_program:
                        print(f"  WARNING: Group program '{group_program}' is not normalized (should be '{normalize_key(group_program)}')")
                    if group_branch and normalize_key(group_branch) != group_branch:
                        print(f"  WARNING: Group branch '{group_branch}' is not normalized (should be '{normalize_key(group_branch)}')")
                    
                    print(f"[GENERATE] Validation failed for group '{group_name}': No courses found\n")
            except Exception as e:
                print(f"[GENERATE] ⚠️ Error validating group: {str(e)}")
                # Continue with other groups even if one fails
                continue
        
        # If validation errors found, return helpful error message
        if validation_errors:
            error_details = []
            for err in validation_errors:
                error_details.append(
                    f"Group '{err['group']}' (Program: {err['program']}, Branch: {err['branch']}, Semester: {err['semester']})"
                )
            
            return jsonify({
                'success': False,
                'status': 'error',
                'error': 'Data validation failed: Some student groups have no matching courses.',
                'validation_errors': validation_errors,
                'message': 'Cannot generate timetable. The following student groups have no courses assigned:\n\n' + '\n'.join(error_details),
                'suggestion': 'Please create courses for these groups in the Courses section, or ensure the program/branch/semester values match exactly.'
            }), 400
        
        print(f"[GENERATE] ✅ Pre-flight validation passed for all {len(target_groups)} groups")
        
        # ============================================================================
        # END PRE-FLIGHT VALIDATION
        # ============================================================================
        
        # 1. OPTIMIZATION: Check if Redis is available for Celery (fast check)
        redis_available = False
        try:
            import redis as redis_lib
            r = redis_lib.Redis.from_url(
                app.config.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                socket_connect_timeout=1,
                socket_timeout=1
            )
            r.ping()
            redis_available = True
        except Exception:
            redis_available = False
            
        # 2. Try Asynchronous Generation if Redis is up
        if redis_available:
            try:
                task = generate_timetable_task.apply_async(kwargs={'filters': filters})
                print(f"[GENERATE] Task triggered with ID: {task.id}")
                return jsonify({
                    'success': True,
                    'message': 'Timetable generation started in background.',
                    'task_id': task.id,
                    'async': True
                }), 202
            except Exception as e:
                print(f"[GENERATE] Celery trigger failed: {str(e)}. Falling back to sync.")
                
        # 3. Fallback to Synchronous Generation
        print("[GENERATE] Running synchronously (Celery/Redis unavailable)...")
        result = _generate_timetable_core(filters=filters)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Timetable generated successfully!',
                'result': result,
                'async': False
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Generation failed'),
                'async': False
            }), 400
            
    except Exception as e:
        print(f"[GENERATE] Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Generation error: {str(e)}'
        }), 500

@app.route('/tasks/<task_id>')
@login_required
def get_task_status(task_id):
    task = generate_timetable_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        # task.info is the result or status dict
        response = {
            'state': task.state,
            'status': task.info.get('status', '') if isinstance(task.info, dict) else str(task.info)
        }
        if task.state == 'SUCCESS':
            result = task.result
            if isinstance(result, dict):
                response.update(result)
    else:
        response = {
            'state': task.state,
            'status': str(task.info),  # Exception info
        }
    return jsonify(response)


@app.route('/timetable/manual-save', methods=['POST'])
@admin_required
def manual_save_timetable():
    """Save manual assignments posted from the admin UI.
    Expected JSON payload:
    { "day": "Monday", "assignments": [ {"period":1, "group":"CSE-A", "course_id":1, "faculty_id":2, "room_id":3}, ... ] }
    """
    payload = request.get_json() or {}
    day = payload.get('day')
    assignments = payload.get('assignments', [])

    if not day:
        return jsonify({'success': False, 'error': 'Day is required.'}), 400

    errors = []
    processed = 0

    # Validate per-group per-day maximums before applying changes
    period_config = PeriodConfig.query.first()
    if period_config:
        max_per_day = getattr(period_config, 'max_periods_per_day_per_group', period_config.periods_per_day)
    else:
        max_per_day = None

    # Gather slots for this day and existing entries in those slots
    day_slots = TimeSlot.query.filter_by(day=day).all()
    day_slot_ids = {s.id for s in day_slots}
    existing_entries = [e for e in TimetableEntry.query.all() if e.time_slot_id in day_slot_ids]

    # Count existing assigned periods per group for the day (only count entries with a course)
    from collections import defaultdict as _dd
    existing_count = _dd(int)
    existing_by_slot_group = {}
    for e in existing_entries:
        if getattr(e, 'course_id', None) not in (None, '', 0):
            existing_count[e.student_group] += 1
        existing_by_slot_group[(e.time_slot_id, e.student_group)] = e

    # Simulate final counts after applying incoming assignments
    final_count = existing_count.copy()
    for a in assignments:
        try:
            period = int(a.get('period'))
        except Exception:
            continue
        group_name = a.get('group')
        if not group_name:
            continue
        slot = TimeSlot.query.filter_by(day=day, period=period).first()
        if not slot:
            continue
        course_id = a.get('course_id')
        incoming_has_course = course_id not in (None, '', 0)
        currently = existing_by_slot_group.get((slot.id, group_name))
        currently_has_course = getattr(currently, 'course_id', None) not in (None, '', 0) if currently else False

        if incoming_has_course and not currently_has_course:
            final_count[group_name] += 1
        if not incoming_has_course and currently_has_course:
            final_count[group_name] -= 1

    # If any group would exceed the per-day maximum, abort early with error
    if max_per_day is not None:
        exceeded = [g for g, cnt in final_count.items() if cnt > max_per_day]
        if exceeded:
            return jsonify({'success': False, 'error': f'Per-day limit exceeded for groups: {", ".join(exceeded)}. Max per day: {max_per_day}'}), 400

    for a in assignments:
        try:
            period = int(a.get('period'))
        except Exception:
            continue

        group_name = a.get('group')
        if not group_name:
            continue

        # Find timeslot
        slot = TimeSlot.query.filter_by(day=day, period=period).first()
        if not slot:
            errors.append(f'No timeslot for {day} P{period}')
            continue

        course_id = a.get('course_id')
        faculty_id = a.get('faculty_id')
        room_id = a.get('room_id')

        # Basic conflict checks: faculty or room already assigned at this timeslot to another group
        if faculty_id:
            # Mongo Query does not support SQLAlchemy-style filter conditions; perform manual conflict check
            existing_entries = TimetableEntry.query.all()
            conflict = next((te for te in existing_entries
                             if te.time_slot_id == slot.id and te.faculty_id == faculty_id and te.student_group != group_name), None)
            if conflict:
                errors.append(f'Faculty id {faculty_id} is already assigned at {day} P{period} to {conflict.student_group}')
                continue

        if room_id:
            existing_entries = 'existing_entries' in locals() and existing_entries or TimetableEntry.query.all()
            conflict = next((te for te in existing_entries
                             if te.time_slot_id == slot.id and te.room_id == room_id and te.student_group != group_name), None)
            if conflict:
                errors.append(f'Room id {room_id} is already used at {day} P{period} by {conflict.student_group}')
                continue

        # Upsert TimetableEntry for this slot + group
        entry = TimetableEntry.query.filter_by(time_slot_id=slot.id, student_group=group_name).first()
        if course_id in (None, '', 0):
            # Delete existing entry if any
            if entry:
                db.session.delete(entry)
            processed += 1
            continue

        if not entry:
            entry = TimetableEntry(time_slot_id=slot.id, student_group=group_name)
            db.session.add(entry)

        entry.course_id = int(course_id) if course_id not in (None, '') else None
        entry.faculty_id = int(faculty_id) if faculty_id not in (None, '') else None
        entry.room_id = int(room_id) if room_id not in (None, '') else None
        processed += 1

    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Database integrity error: ' + str(e)}), 500

    result = {'success': True, 'processed': processed}
    if errors:
        result['warnings'] = errors
    return jsonify(result)

@app.route('/timetable/clear', methods=['POST'])
@admin_required
def clear_timetable():
    TimetableEntry.query.delete()
    db.session.commit()
    return jsonify({'success': True})

# Export
# Settings Management
@app.route('/settings')
@admin_required
def settings():
    user = get_current_user()
    period_config = PeriodConfig.query.first()
    
    print(f"[DEBUG SETTINGS] Loading settings page")
    if period_config:
        print(f"[DEBUG SETTINGS] Config found: periods_per_day={period_config.periods_per_day}, period_duration_minutes={period_config.period_duration_minutes}, day_start_time={period_config.day_start_time}, days_of_week={period_config.days_of_week}")
    else:
        print(f"[DEBUG SETTINGS] No config found in database!")
    
    breaks = BreakConfig.query.order_by(BreakConfig.after_period).all()
    days_list = [d.strip() for d in period_config.days_of_week.split(',')] if period_config else []
    return render_template('settings.html', period_config=period_config, breaks=breaks, days_list=days_list, user=user)

@app.route('/settings/period', methods=['POST'])
@admin_required
def update_period_config():
    try:
        data = request.json
        print(f"[DEBUG] Received data: {data}")
        
        period_config = PeriodConfig.query.first()
        
        if not period_config:
            print("[DEBUG] No existing config found, creating new one")
            period_config = PeriodConfig(id=1)
        else:
            print(f"[DEBUG] Existing config found: periods_per_day={period_config.periods_per_day}, days_of_week={period_config.days_of_week}")
        
        # Update the config fields
        period_config.periods_per_day = int(data['periods_per_day'])
        period_config.period_duration_minutes = int(data['period_duration_minutes'])
        period_config.day_start_time = data['day_start_time']
        period_config.days_of_week = ','.join(data.get('days_of_week', []))
        
        print(f"[DEBUG] Updated config: periods_per_day={period_config.periods_per_day}, days_of_week={period_config.days_of_week}")
        
        # CRITICAL FIX: Always add to session, even for existing configs
        db.session.add(period_config)
        
        # Enforce singleton: remove any stray extra PeriodConfig documents
        try:
            existing = PeriodConfig.query.all()
            for cfg in existing:
                if getattr(cfg, 'id', None) != 1:
                    db.session.delete(cfg)
            # Flush deletions before commit
            db.session.flush()
        except Exception as _:
            pass
        db.session.commit()
        
        print("[DEBUG] Committed to database")
        
        # Verify the save by querying again
        verify_config = PeriodConfig.query.first()
        if verify_config:
            print(f"[DEBUG] Verification query: periods_per_day={verify_config.periods_per_day}, days_of_week={verify_config.days_of_week}")
        
        # Regenerate time slots
        generate_time_slots()
        
        return jsonify({'success': True, 'message': 'Period configuration updated and time slots regenerated.'})
    except Exception as e:
        db.session.rollback()
        print(f"Error updating period config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings/break/add', methods=['POST'])
@admin_required
def add_break():
    try:
        data = request.json
        break_config = BreakConfig(
            break_name=data['break_name'],
            after_period=int(data['after_period']),
            duration_minutes=int(data['duration_minutes']),
            order=int(data.get('order', 1))
        )
        db.session.add(break_config)
        db.session.commit()
        
        # Regenerate time slots
        generate_time_slots()
        
        return jsonify({'success': True, 'id': break_config.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings/break/<int:break_id>/update', methods=['POST'])
@admin_required
def update_break(break_id):
    try:
        data = request.json
        break_config = BreakConfig.query.get_or_404(break_id)
        
        break_config.break_name = data['break_name']
        break_config.after_period = int(data['after_period'])
        break_config.duration_minutes = int(data['duration_minutes'])
        break_config.order = int(data.get('order', break_config.order))
        
        db.session.commit()
        
        # Regenerate time slots
        generate_time_slots()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings/break/<int:break_id>/delete', methods=['POST'])
@admin_required
def delete_break(break_id):
    try:
        break_config = BreakConfig.query.get_or_404(break_id)
        db.session.delete(break_config)
        db.session.commit()
        
        # Regenerate time slots
        generate_time_slots()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/timetable/export')
@login_required
def export_timetable():
    slots = TimeSlot.query.all()
    slots_dict = {s.id: s for s in slots}
    valid_slot_ids = set(slots_dict.keys())

    # Filter entries to only include those with valid time_slot_id
    entries = [e for e in TimetableEntry.query.all() if e.time_slot_id in valid_slot_ids]

    courses_dict = {c.id: c for c in Course.query.all()}
    faculty_dict = {f.id: f for f in Faculty.query.all()}
    rooms_dict = {r.id: r for r in Room.query.all()}
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Day', 'Period', 'Start Time', 'End Time', 'Course Code', 'Course Name', 'Faculty', 'Room'])
    
    for entry in entries:
        slot = slots_dict[entry.time_slot_id]
        course = courses_dict[entry.course_id]
        faculty = faculty_dict[entry.faculty_id]
        room = rooms_dict[entry.room_id]
        writer.writerow([
            slot.day,
            slot.period,
            slot.start_time,
            slot.end_time,
            course.code,
            course.name,
            faculty.name,
            room.name
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'timetable_{datetime.now().strftime("%Y%m%d")}.csv'
    )


# Timetable History Management
@app.route('/api/timetable/history', methods=['GET'])
@login_required
def get_timetable_history():
    """Get list of all saved timetable histories"""
    try:
        # Optimization: Use projection to EXCLUDE the heavy 'entries' field
        # This prevents loading potentially hundreds of megabytes of JSON just for a list
        histories = TimetableHistory.query.options({'entries': 0}).order_by(TimetableHistory.id.desc()).all()
        
        history_list = []
        for h in histories:
            try:
                # Use getattr with default to handle cases where field might be missing
                filters_str = getattr(h, 'filters', '{}') or '{}'
                stats_str = getattr(h, 'stats', '{}') or '{}'
                
                filters = json.loads(filters_str) if isinstance(filters_str, str) else filters_str
                stats = json.loads(stats_str) if isinstance(stats_str, str) else stats_str
            except Exception as e:
                print(f"[HISTORY API] JSON Parse Error for history {getattr(h, 'id')}: {e}")
                filters = {}
                stats = {}
            
            history_list.append({
                'id': getattr(h, 'id', None),
                'name': getattr(h, 'name', 'Unnamed Timetable'),
                'generated_at': getattr(h, 'generated_at', None),
                'filters': filters,
                'stats': stats,
                'is_active': getattr(h, 'is_active', False)
            })
        
        return jsonify({
            'success': True,
            'histories': history_list
        })
    except Exception as e:
        print(f"[HISTORY API] Error fetching history: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/timetable/history/<int:history_id>', methods=['GET'])
@login_required
def get_timetable_history_detail(history_id):
    """Get detailed information about a specific timetable history"""
    try:
        history = TimetableHistory.query.get(history_id)
        if not history:
            return jsonify({'success': False, 'error': 'History not found'}), 404
        
        # Parse stored data
        try:
            filters = json.loads(getattr(history, 'filters', '{}'))
            stats = json.loads(getattr(history, 'stats', '{}'))
            entries_data = json.loads(getattr(history, 'entries', '[]'))
        except:
            filters = {}
            stats = {}
            entries_data = []
        
        return jsonify({
            'success': True,
            'history': {
                'id': getattr(history, 'id', None),
                'name': getattr(history, 'name', 'Unnamed Timetable'),
                'generated_at': getattr(history, 'generated_at', None),
                'filters': filters,
                'stats': stats,
                'entries': entries_data,
                'is_active': getattr(history, 'is_active', False)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/timetable/history/<int:history_id>/load', methods=['POST'])
@admin_required
def load_timetable_history(history_id):
    """Load a historical timetable into the current timetable view"""
    try:
        history = TimetableHistory.query.get(history_id)
        if not history:
            return jsonify({'success': False, 'error': 'History not found'}), 404
        
        # Parse stored entries
        try:
            entries_data = json.loads(getattr(history, 'entries', '[]'))
        except:
            return jsonify({'success': False, 'error': 'Invalid history data'}), 400
        
        # Clear current timetable
        TimetableEntry.query.delete()
        db.session.commit()
        
        # Restore entries from history using bulk insert for speed
        restored_entries = []
        for entry_data in entries_data:
            restored_entries.append({
                'id': get_next_id(db._db, 'timetableentry'), # Use direct helper for IDs
                'time_slot_id': entry_data.get('time_slot_id'),
                'course_id': entry_data.get('course_id'),
                'faculty_id': entry_data.get('faculty_id'),
                'room_id': entry_data.get('room_id'),
                'student_group': entry_data.get('student_group')
            })
            
        if restored_entries:
            db._db['timetableentry'].insert_many(restored_entries, ordered=False)
        
        restored_count = len(restored_entries)
        
        # Mark this history as active
        db._db['timetablehistory'].update_many(
            {'is_active': True},
            {'$set': {'is_active': False}}
        )
        db._db['timetablehistory'].update_one({'id': history_id}, {'$set': {'is_active': True}})
        
        invalidate_cache('timetable_view')
        invalidate_cache('timetable_view')
        
        return jsonify({
            'success': True,
            'message': f'Loaded {restored_count} timetable entries from history',
            'restored_count': restored_count
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/timetable/history/<int:history_id>/delete', methods=['DELETE', 'POST'])
@admin_required
def delete_timetable_history(history_id):
    """
    Delete a timetable history record.
    If the history being deleted is active, it also clears the current timetable entries.
    """
    try:
        history = TimetableHistory.query.get(history_id)
        if not history:
            return jsonify({'success': False, 'error': 'History not found'}), 404
        
        # If it's active, clear the current timetable too
        is_active = getattr(history, 'is_active', False)
        admin_user = (User.query.get(session.get('user_id')) if session.get('user_id') else None)
        admin_name = getattr(admin_user, 'name', 'Unknown Admin')
        admin_id = session.get('user_id', 'Unknown')
        
        print(f"[AUDIT] TIMETABLE HISTORY DELETE: HistoryID={history_id}, Active={is_active}, DeletedBy={admin_name} (ID: {admin_id})")
        
        if is_active:
            print(f"[HISTORY] Deleting ACTIVE history {history_id}. Clearing current timetable.")
            TimetableEntry.query.delete()
            # No need to search for other active ones since this was the only one
        
        db.session.delete(history)
        db.session.commit()
        
        # Invalidate cache if active was deleted
        if is_active:
            invalidate_cache('timetable_view')
            invalidate_cache('timetable_entries')
        
        return jsonify({
            'success': True, 
            'message': 'History deleted successfully',
            'cleared_active': is_active
        })
    except Exception as e:
        db.session.rollback()
        print(f"[HISTORY] Error deleting history: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/timetable/history/reset', methods=['DELETE', 'POST'])
@admin_required
def reset_timetable_history():
    """
    Full reset of timetable system:
    1. Deletes all TimetableHistory records
    2. Clears current TimetableEntry records
    3. Resets all active states
    Does NOT affect Courses, Faculty, Rooms, or Student Groups.
    """
    try:
        admin_user = (User.query.get(session.get('user_id')) if session.get('user_id') else None)
        admin_name = getattr(admin_user, 'name', 'Unknown Admin')
        admin_id = session.get('user_id', 'Unknown')
        
        print(f"[AUDIT] TIMETABLE HISTORY RESET: Performed by {admin_name} (ID: {admin_id})")
        print("[HISTORY] Performing full history reset...")
        
        # 1. Delete all history records
        TimetableHistory.query.delete()
        
        # 2. Clear current timetable entries
        TimetableEntry.query.delete()
        
        db.session.commit()
        
        # 3. Invalidate caches
        invalidate_cache('timetable_view')
        invalidate_cache('timetable_entries')
        
        return jsonify({
            'success': True,
            'message': 'All timetable history and current schedule have been cleared successfully.'
        })
    except Exception as e:
        db.session.rollback()
        print(f"[HISTORY] Error during reset: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Start background worker for async user account creation
    try:
        from background_tasks import start_background_worker
        
        def get_db_session():
            """Factory function to get a fresh database session for background tasks"""
            return db.session
        
        start_background_worker(get_db_session)
        print("[Background Worker] Initialized for async user account creation")
    except Exception as e:
        print(f"[Background Worker] Failed to start: {e}")
        print("[Background Worker] Imports will still work but may be slower")
    
    # Run with reloader disabled to avoid Windows socket errors
    
    # ONE-TIME DATA MIGRATION: Normalize program/branch keys
    # This fixes existing data to use canonical format (lowercase, no dots/spaces)
    try:
        from normalization import normalize_key
        
        print("[MIGRATION] Starting program/branch normalization...")
        
        # Normalize Courses
        course_count = 0
        for course in Course.query.all():
            changed = False
            if course.program:
                normalized = normalize_key(course.program)
                if course.program != normalized:
                    print(f"  Course {course.code}: '{course.program}' → '{normalized}'")
                    course.program = normalized
                    changed = True
            if course.branch:
                normalized = normalize_key(course.branch)
                if course.branch != normalized:
                    print(f"  Course {course.code}: '{course.branch}' → '{normalized}'")
                    course.branch = normalized
                    changed = True
            if changed:
                db.session.add(course)
                course_count += 1
        
        # Normalize Student Groups
        group_count = 0
        for group in StudentGroup.query.all():
            changed = False
            if group.program:
                normalized = normalize_key(group.program)
                if group.program != normalized:
                    print(f"  Group {group.name}: '{group.program}' → '{normalized}'")
                    group.program = normalized
                    changed = True
            if group.branch:
                normalized = normalize_key(group.branch)
                if group.branch != normalized:
                    print(f"  Group {group.name}: '{group.branch}' → '{normalized}'")
                    group.branch = normalized
                    changed = True
            if changed:
                db.session.add(group)
                group_count += 1
        
        # Normalize Branches
        branch_count = 0
        for branch in Branch.query.all():
            changed = False
            if branch.program:
                normalized = normalize_key(branch.program)
                if branch.program != normalized:
                    print(f"  Branch {branch.code}: '{branch.program}' → '{normalized}'")
                    branch.program = normalized
                    changed = True
            if branch.name:
                normalized = normalize_key(branch.name)
                if branch.name != normalized:
                    print(f"  Branch {branch.code}: '{branch.name}' → '{normalized}'")
                    branch.name = normalized
                    changed = True
            if changed:
                db.session.add(branch)
                branch_count += 1
        
        if course_count + group_count + branch_count > 0:
            db.session.commit()
            print(f"[MIGRATION] ✅ Normalized {course_count} courses, {group_count} groups, {branch_count} branches")
        else:
            print("[MIGRATION] ✅ All data already normalized")
            
    except Exception as e:
        print(f"[MIGRATION] ⚠️ Normalization skipped due to error: {e}")
        import traceback
        traceback.print_exc()
    
    # Add one-time cleanup: normalize faculty availability types
    try:
        fixed = 0
        for f in Faculty.query.all():
            raw = getattr(f, 'availability', None)
            if raw and not isinstance(raw, str):
                if isinstance(raw, (dict, list)):
                    f.availability = json.dumps(raw)
                else:
                    f.availability = '{}'
                fixed += 1
            elif raw is None:
                f.availability = '{}'
                fixed += 1
        if fixed:
            db.session.commit()
            print(f"Normalized availability for {fixed} faculty records.")
    except Exception as e:
        print(f"Availability normalization skipped due to error: {e}")
    
    # MIGRATION: Fix Semester=0 issue in existing data
    try:
        print("[MIGRATION] Checking for Semester=0 issues in StudentGroups...")
        
        # Access MongoDB directly with correct collection names
        mongo_db = db.session._db
        
        # Fix StudentGroups with Semester=0 or None (correct collection: studentgroup)
        result_groups = mongo_db['studentgroup'].update_many(
            {'$or': [{'semester': 0}, {'semester': None}]},
            {'$set': {'semester': 1}}
        )
        
        # Fix Courses with Semester=0 or None (correct collection: course)
        result_courses = mongo_db['course'].update_many(
            {'$or': [{'semester': 0}, {'semester': None}]},
            {'$set': {'semester': 1}}
        )
        
        if result_groups.modified_count or result_courses.modified_count:
            print(f"[MIGRATION] [OK] Fixed {result_groups.modified_count} groups and {result_courses.modified_count} courses with invalid semesters")
        else:
            print("[MIGRATION] [OK] No semester issues found - all data is clean!")
    except Exception as e:
        print(f"[MIGRATION] Semester fix skipped due to error: {e}")
    
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)

# Vercel serverless function handler
# This is required for Vercel deployment
if __name__ != '__main__':
    # When running on Vercel, expose the app object
    application = app
