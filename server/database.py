"""
MongoDB database connection and collection setup.
This file initializes the database connection when the server starts.
"""
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME

# Create MongoDB client (works for both local and Atlas)
client = MongoClient(MONGODB_URI)

# Connect to database
db = client[DATABASE_NAME]

# Define collections
users_collection = db["users"]
segmentations_collection = db["segmentations"]

# Create indexes for better query performance
users_collection.create_index("email", unique=True)
segmentations_collection.create_index("user_id")
segmentations_collection.create_index("created_at")

print(f"✅ Connected to MongoDB: {DATABASE_NAME}")
