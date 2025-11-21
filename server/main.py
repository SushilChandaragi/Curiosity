"""
Main FastAPI application file.
Defines all API endpoints for authentication and image segmentation.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from bson import ObjectId
import base64

# Import our modules
from config import FRONTEND_URL
from database import users_collection, segmentations_collection
from models import (
    UserRegister, UserLogin, Token, UserResponse,
    SegmentationResponse, SegmentationHistory
)
from auth import (
    hash_password, verify_password, create_access_token, get_current_user
)
from ml_model import model, image_to_base64

app = FastAPI(title="Image Segmentation API")

# Enable CORS so React can communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://curiosity-frontend.onrender.com",
        "https://curiosity-frontend-*.onrender.com",
        "*"
    ],
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============= AUTHENTICATION ENDPOINTS =============

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """
    Create a new user account.
    Steps: 1) Check if email exists, 2) Hash password, 3) Save to DB, 4) Return JWT token
    """
    # Check if email already exists
    if users_collection.find_one({"email": user_data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user document
    user_doc = {
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "created_at": datetime.utcnow()
    }
    
    # Insert into database
    result = users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)
    
    # Generate JWT token
    access_token = create_access_token({"user_id": user_id})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            email=user_data.email,
            name=user_data.name
        )
    )

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Log in existing user.
    Steps: 1) Find user by email, 2) Verify password, 3) Return JWT token
    """
    # Find user in database
    user = users_collection.find_one({"email": credentials.email})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify password
    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Generate JWT token
    user_id = str(user["_id"])
    access_token = create_access_token({"user_id": user_id})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            email=user["email"],
            name=user["name"]
        )
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_me(user_id: str = Depends(get_current_user)):
    """
    Get current logged-in user's profile.
    This endpoint is protected - requires valid JWT token.
    """
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    
    return UserResponse(
        id=user_id,
        email=user["email"],
        name=user["name"]
    )

# ============= SEGMENTATION ENDPOINTS =============

@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Run ML model on uploaded image.
    Steps:
    1) Read uploaded image
    2) Run segmentation model
    3) Save original + result to database
    4) Return result to frontend
    """
    # Read uploaded file
    file_bytes = await file.read()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Run ML model
        segmentation_mask = model.segment_image(file_bytes)
        
        # Convert images to Base64 for storage/transmission
        original_b64 = base64.b64encode(file_bytes).decode()
        mask_b64 = image_to_base64(segmentation_mask)
        
        # Save to database
        doc = {
            "user_id": user_id,
            "filename": file.filename,
            "original_image": original_b64,
            "segmentation_mask": mask_b64,
            "created_at": datetime.utcnow()
        }
        
        result = segmentations_collection.insert_one(doc)
        
        return SegmentationResponse(
            id=str(result.inserted_id),
            filename=file.filename,
            original_image=original_b64,
            segmentation_mask=mask_b64,
            created_at=doc["created_at"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.get("/history")
async def get_history(user_id: str = Depends(get_current_user)):
    """
    Get all past segmentations for the logged-in user.
    Returns list sorted by newest first.
    """
    # Find all segmentations for this user
    segmentations = segmentations_collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1)  # Newest first
    
    # Convert to list and format response
    history = []
    for seg in segmentations:
        history.append(SegmentationHistory(
            id=str(seg["_id"]),
            filename=seg["filename"],
            thumbnail=seg["segmentation_mask"],  # Using mask as thumbnail
            created_at=seg["created_at"]
        ))
    
    return {"history": history}

@app.get("/segmentation/{segmentation_id}", response_model=SegmentationResponse)
async def get_segmentation(
    segmentation_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get a specific segmentation result by ID.
    User can only access their own segmentations.
    """
    try:
        seg = segmentations_collection.find_one({
            "_id": ObjectId(segmentation_id),
            "user_id": user_id  # Security: ensure user owns this segmentation
        })
        
        if not seg:
            raise HTTPException(status_code=404, detail="Segmentation not found")
        
        return SegmentationResponse(
            id=str(seg["_id"]),
            filename=seg["filename"],
            original_image=seg["original_image"],
            segmentation_mask=seg["segmentation_mask"],
            created_at=seg["created_at"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid segmentation ID")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Image Segmentation API"}

# Run with: uvicorn main:app --reload
