"""
Pydantic models for request/response validation.
These define the structure of data coming in and going out of the API.
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# ============= USER MODELS =============
class UserRegister(BaseModel):
    """Data needed to create a new user account"""
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    """Data needed to log in"""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """User data sent back to client (without password)"""
    id: str
    email: str
    name: str

class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str
    user: UserResponse

# ============= SEGMENTATION MODELS =============
class SegmentationResponse(BaseModel):
    """Response after running segmentation"""
    id: str
    filename: str
    original_image: str  # Base64 string
    segmentation_mask: str  # Base64 string
    created_at: datetime

class SegmentationHistory(BaseModel):
    """List of past segmentations"""
    id: str
    filename: str
    thumbnail: str  # Small preview
    created_at: datetime
