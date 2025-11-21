"""
Configuration file for the FastAPI application.
Loads environment variables and defines global settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Security Settings
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Database Settings (Updated for MongoDB Atlas)
MONGODB_URI = os.getenv("MONGODB_URI", os.getenv("MONGODB_URL", "mongodb://localhost:27017/"))
DATABASE_NAME = os.getenv("DATABASE_NAME", "segmentation_app")

# CORS Settings
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Server Port
PORT = int(os.getenv("PORT", "8000"))
