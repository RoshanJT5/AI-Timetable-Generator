# AI Timetable Generator (PlanSphere.AI)

![PlanSphere.AI Banner](https://via.placeholder.com/1200x300?text=AI+Timetable+Generator)

A powerful, AI-driven full-stack application for automated academic timetable generation. Built with Flask, MongoDB Atlas, and PuLP optimization engine, deployed on Vercel.

## 🚀 Features

- **AI-Powered Generation**: Uses Linear Programming (PuLP) to generate conflict-free timetables.
- **Smart Constraints**: Handles faculty availability, room capacity, course requirements, and equipment matching.
- **Role-Based Access**: Secure portals for Admins, Teachers, and Students.
- **Excel/CSV Import**: Bulk import support for courses, faculty, rooms, and students.
- **Interactive Dashboard**: Modern UI with real-time statistics and management tools.
- **Cloud Database**: Scalable data storage with MongoDB Atlas.
- **Serverless Deployment**: Optimized for Vercel serverless architecture (with synchronous fallback).

## 🛠️ Tech Stack

- **Backend**: Python 3.11, Flask, Gunicorn
- **Database**: MongoDB Atlas (PyMongo)
- **Algorithm**: PuLP (Linear Programming)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap Icons
- **Deployment**: Vercel Serverless Functions

## 📋 Prerequisites

- **Python 3.11+**
- **MongoDB Atlas Account**
- **Vercel Account** (for deployment)

## 🔧 Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RoshanJT5/AI-Timetable-Generator.git
   cd AI-Timetable-Generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI=your_mongodb_connection_string
   MONGO_DBNAME=timetable
   SECRET_KEY=your_secure_random_key
   ```

5. **Run the application**
   ```bash
   python app_with_navigation.py
   ```
   Access the app at `http://localhost:5000`

## ☁️ Deployment on Vercel

This project is configured for seamless deployment on Vercel.

1. **Push to GitHub** (Done!)
2. **Import to Vercel**:
    - Go to Vercel Dashboard
    - Click **Add New...** > **Project**
    - Import `AI-Timetable-Generator` repository
3. **Configure Environment Variables**:
    - Add the following in Vercel Project Settings:
    - `MONGO_URI`: Your MongoDB Atlas connection string
    - `MONGO_DBNAME`: `timetable`
    - `SECRET_KEY`: A secure random string
4. **Deploy**: Click **Deploy** and wait for the build to finish.

## 📂 Project Structure

AI-Timetable-Generator/
├── templates/              # HTML Templates (Frontend)
├── app_with_navigation.py  # Main Flask Application (Backend)
├── models.py               # Database Models (MongoDB)
├── scheduler.py            # Timetable Generation Logic
├── vercel.json             # Vercel Configuration
├── requirements.txt        # Python Dependencies
└── Working.md              # Detailed Project Documentation

## 🛡️ Security

- **Password Hashing**: Bcrypt (via `password_security.py`)
- **Session Security**: JWT-based Authentication
- **Environment Variables**: Sensitive data isolated in `.env`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
