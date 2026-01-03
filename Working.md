# PlanSphere.AI - Intelligent Timetable Management System

## 1. Project Overview
PlanSphere.AI is an advanced, AI-powered automatic timetable generation system designed for educational institutions. It streamlines the complex process of scheduling classes, managing faculty workloads, and allocating rooms by using intelligent algorithms to generate conflict-free schedules that respect multiple constraints (availability, room capacity, lab requirements, etc.).

## 2. Technical Architecure

### **Frontend Layer (The "View")**
*   **Technology**: HTML5, CSS3, JavaScript (Vanilla + Bootstrap 5).
*   **Template Engine**: Jinja2 (Serverside rendering).
*   **Key Responsibilities**:
    *   Displays interactive UI for management and viewing.
    *   Handles asynchronous operations (AJAX/Fetch) for smooth UX (e.g., generation progress, toast notifications).
    *   **Standardized UI**: centralized feedback via `UI.showToast` and `UI.confirm` in `base.html`.

### **Backend Layer (The "Controller")**
*   **technology**: Python (Flask Framework).
*   **Key Responsibilities**:
    *   **REST API**: Endpoints for CRUD operations and generation triggers.
    *   **Routing**: Serves HTML pages and handles navigation.
    *   **Security**: JWT-based authentication and BCrypt password hashing.

### **Logic Layer (The "Brain")**
*   **Core Component**: `scheduler.py`
*   **Algorithm**: Constraint Satisfaction Problem (CSP) / Integer Linear Programming (ILP) techniques.
*   **Key Responsibilities**:
    *   Analyzing constraints (Room tags, Faculty hours, Concurrency).
    *   Generating valid time slots.
    *   Optimizing for soft constraints (preferred times, balanced workload).

### **Data Layer (The "Model")**
*   **Database**: MongoDB (Atlas/Local).
*   **Abstraction**: Custom ORM-like layer in `models.py`.
*   **Key Responsibilities**: Persistence of Courses, Faculty, Students, and Timetable Entries.

---

## 3. End-to-End Workflow

### **Step 1: System Initialization**
*   **Entry Point**: `app_with_navigation.py` initializes the Flask app, connects to the MongoDB database via `models.py`, and configures JWT serialization.
*   **First Run**: If the database is empty, optional seeding scripts or manual admin creation sets up the first user.

### **Step 2: Data Management (Admin)**
1.  **Faculty Management**: Admin adds faculty members via `faculty.html`.
    *   *Backend*: `models.py` -> `Faculty` collection.
    *   *Feature*: Faculty can set their "Availability" (preferred times), which `scheduler.py` respects.
2.  **Course Management**: Admin defines subjects, credits, and lab requirements in `courses.html`.
3.  **Infrastructure**: Rooms and their capabilities (e.g., "Computer Lab") are defined in `rooms.html`.
4.  **Student Groups**: Classes (e.g., "CSE-A") are created in `student_groups.html`.

### **Step 3: Timetable Generation (The Core Process)**
1.  **Trigger**: Authenticated Admin clicks "Generate Timetable" on `timetable.html`.
2.  **Request**: Frontend sends a POST request to `/timetable/generate` (optionally with filters for Branch/Semester).
3.  **Processing** (`scheduler.py`):
    *   **Fetch**: Loads all active Constraints, Faculty, Courses, and Rooms.
    *   **Solver**: Runs the scheduling algorithm. It tries to place `Course` sessions into `TimeSlot` grid cells.
    *   **Validation**: Checks for hard conflicts (Double booking faculty/room/group).
4.  **Result**:
    *   If successful, `TimetableEntry` objects are bulk-saved to MongoDB.
    *   A summary JSON response (Success + Warnings) is sent back.
5.  **Feedback**: Frontend receives the response, shows a "Timetable Generated Successfully" toast, and reloads the view.

### **Step 4: Consumption & Export**
*   **Views**: The timetable is rendered in a grid view on `timetable.html`. Users can filter by Group, Faculty, or Room.
*   **History**: Every generation is snapshotted. Admins can roll back to previous versions via the History sidebar.
*   **Export**: Users can download the schedule as CSV/Excel.

---

## 4. File Structure & Detailed Descriptions

### **Root Directory**
*   **`app_with_navigation.py`**: **The Application Core**. This is the main entry point. It defines all Flask routes (`@app.route`), handles HTTP requests, integrates the scheduler, and manages session state. It replaces the traditional `app.py`.
*   **`scheduler.py`**: **The Logic Engine**. Contains the `TimetableGenerator` class. It holds the complex logic for matching courses to slots, checking constraints (e.g., "Faculty X cannot teach two classes at once"), and handling "Soft constraints" (load balancing).
*   **`models.py`**: **Database Abstraction**. Defines Python classes (`User`, `Course`, `Faculty`, `TimetableEntry`) that map to MongoDB documents. It handles DB connections and basic CRUD operations.
*   **`auth_jwt.py`**: **Authentication**. Manages JSON Web Tokens (JWT) for secure user login and session management.
*   **`password_security.py`**: **Security**. Handles `bcrypt` password hashing and verification to ensure user credentials are safe.
*   **`csv_processor.py`**: **Data Ingestion**. A utility module for parsing and validating uploaded CSV/Excel files for bulk data import.
*   **`cache.py`**: **Performance**. Implements caching strategies (e.g., Redis or in-memory) to speed up frequently accessed lookups.
*   **`normalization.py`**: **Data Hygiene**. Helper functions to normalize strings (e.g., "Computer Science" -> "computer_science") for consistent database querying.
*   **`requirements.txt`**: List of Python dependencies (Flask, PyMongo, Pandas, etc.).
*   **`Procfile` & `vercel.json`**: Deployment configurations for hosting platforms.

### **Template Directory (`templates/`)** - The User Interface
*   **`base.html`**: The master layout file. Contains the navigation bar, footer, and **Global UI Utilities** (`UI.showToast`, `UI.confirm`) used across all pages.
*   **`timetable.html`**: The main dashboard. Features the timetable grid, generation controls, specific grouping filters, and history viewer.
*   **`courses.html`**: Management interface for adding/editing academic courses and subjects.
*   **`faculty.html`**: Management interface for faculty profiles, workload preferences, and expertise.
*   **`rooms.html`**: Management interface for physical infrastructure and room tags.
*   **`students.html`**: Student data management.
*   **`student_groups.html`**: Management of class sections/groups (e.g., "Class 10-A").
*   **`settings.html`**: System-wide configurations (e.g., Working days, Period duration).
*   **`login.html` / `register.html`**: Authentication pages.

### **Static Directory (`static/`)**
*   Contains CSS styles, images, and client-side JavaScript assets.

---

## 5. Development Notes
*   **Configuration**: Environment variables are managed via `.env` (e.g., `MONGO_URI`, `SECRET_KEY`).
*   **Deployment**: The app is production-ready with `gunicorn` configuration, but includes `debug` mode logic in `app_with_navigation.py` for local dev.
