"""
MongoDB database connection and collection setup.
This file initializes the database connection when the server starts.
"""
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME
import certifi

# Create MongoDB client with SSL/TLS configuration for Atlas
client = MongoClient(
    MONGODB_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=10000
)

# Connect to database
db = client[DATABASE_NAME]

# Define collections
users_collection = db["users"]
segmentations_collection = db["segmentations"]

# Create indexes for better query performance (deferred to avoid startup errors)
try:
    users_collection.create_index("email", unique=True)
    segmentations_collection.create_index("user_id")
    segmentations_collection.create_index("created_at")
    print(f"✅ Connected to MongoDB: {DATABASE_NAME}")
except Exception as e:
    print(f"⚠️ MongoDB connected but indexes may not be created: {e}")
    print(f"✅ Connected to MongoDB: {DATABASE_NAME}")
