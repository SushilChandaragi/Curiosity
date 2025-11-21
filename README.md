# 🔮 AI Image Segmentation Platform

A professional, dark-mode web application for running image segmentation models. Built with **FastAPI** (Python backend) and **React** (frontend) with MongoDB for data storage.

## ✨ Features

- 🔐 **User Authentication** - Secure login/registration with JWT tokens
- 🖼️ **Image Upload** - Upload images for segmentation
- 🤖 **ML Model Integration** - Runs PyTorch/Hugging Face models (.pth/.hf files)
- 📊 **Results Display** - View original image + segmentation mask side-by-side
- 📜 **History** - Browse all previous segmentations
- 🌙 **Dark Mode UI** - Professional gradient design

---

## 🏗️ Architecture

```
Frontend (React) → Backend (FastAPI) → ML Models (.pth/.hf)
                          ↓
                    MongoDB (Storage)
```

**Flow:**
1. User logs in → Gets JWT token
2. Uploads image → Sent to FastAPI backend
3. Backend runs ML model → Returns segmentation mask
4. Results saved to MongoDB → Displayed in UI

---

## 📋 Prerequisites

Install these before starting:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download here](https://nodejs.org/)
3. **MongoDB** - [Download Community Edition](https://www.mongodb.com/try/download/community)
   - Or use MongoDB Atlas (cloud): [Sign up free](https://www.mongodb.com/cloud/atlas/register)

---

## 🚀 Step-by-Step Setup

### **Step 1: Install MongoDB**

#### Option A: Local MongoDB (Windows)
```powershell
# Download MongoDB Community Server from:
# https://www.mongodb.com/try/download/community

# After installation, start MongoDB:
net start MongoDB

# Verify it's running:
mongo --version
```

#### Option B: MongoDB Atlas (Cloud - Easier)
1. Create free account at https://www.mongodb.com/cloud/atlas
2. Create a cluster
3. Get connection string (looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)
4. Use this in your `.env` file instead of `mongodb://localhost:27017/`

---

### **Step 2: Set Up Backend (FastAPI)**

```powershell
# Navigate to server folder
cd C:\Users\SushilSC\Desktop\Curiosity\server

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from example)
Copy-Item .env.example .env

# Edit .env file with your settings
notepad .env
```

**Important: Edit `.env` file:**
```env
SECRET_KEY=your-secret-key-change-this-in-production-make-it-long-and-random
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=segmentation_app
FRONTEND_URL=http://localhost:3000
```

---

### **Step 3: Configure Your ML Model**

Open `server/ml_model.py` and customize for your model:

#### **For Hugging Face Models (.hf):**
The default code already loads a Hugging Face model. Just change the model name:
```python
model_name = "your-huggingface-model-id"
```

#### **For Custom PyTorch Models (.pth):**
Uncomment and modify this section in `ml_model.py`:
```python
# Import your model architecture
from your_model_file import YourModelClass

# In __init__ method:
self.custom_model = YourModelClass()
self.custom_model.load_state_dict(torch.load("path/to/your/model.pth"))
self.custom_model.eval()

# In segment_image method, replace the inference code with your logic
```

**Example for .pth file:**
```python
# If you have a file called my_segmentation_model.pth
import torch
from PIL import Image

class MyModel(torch.nn.Module):
    # Your model architecture here
    pass

# In ml_model.py __init__:
self.model = MyModel()
self.model.load_state_dict(torch.load("my_segmentation_model.pth"))
self.model.eval()
```

---

### **Step 4: Start Backend Server**

```powershell
# Make sure you're in server folder with venv activated
cd C:\Users\SushilSC\Desktop\Curiosity\server
.\venv\Scripts\Activate

# Start FastAPI server
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
✅ Connected to MongoDB: segmentation_app
🔄 Loading ML models...
✅ Models loaded successfully!
```

**Test the API:** Open http://localhost:8000 in your browser. You should see:
```json
{"status": "running", "message": "Image Segmentation API"}
```

---

### **Step 5: Set Up Frontend (React)**

Open a **new PowerShell window**:

```powershell
# Navigate to client folder
cd C:\Users\SushilSC\Desktop\Curiosity\client

# Install dependencies (this takes a few minutes)
npm install

# Start React development server
npm start
```

The browser will automatically open to http://localhost:3000

---

## 🎮 How to Use

### **1. Create Account**
- Open http://localhost:3000
- Click "Sign up"
- Enter your name, email, and password
- You'll be automatically logged in

### **2. Upload & Segment Image**
- Click "Choose Image"
- Select an image from your computer
- Click "Run Model"
- Wait for processing (shows loading spinner)
- View results: Original image + Segmentation mask

### **3. View History**
- Scroll down to see "Previous Results"
- Click any thumbnail to view that segmentation again
- Results are saved forever in MongoDB

### **4. Logout**
- Click "Logout" button in top-right corner

---

## 📂 Project Structure

```
Curiosity/
├── server/                    # Python Backend
│   ├── main.py               # FastAPI app & API endpoints
│   ├── auth.py               # JWT authentication logic
│   ├── database.py           # MongoDB connection
│   ├── models.py             # Data validation schemas
│   ├── ml_model.py           # ML model loader & inference
│   ├── config.py             # Configuration settings
│   ├── requirements.txt      # Python dependencies
│   ├── .env                  # Environment variables (create this!)
│   └── .env.example          # Example environment file
│
└── client/                    # React Frontend
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── components/
    │   │   ├── Login.js      # Login form
    │   │   ├── Register.js   # Registration form
    │   │   ├── Dashboard.js  # Main app (upload + history)
    │   │   └── Navbar.js     # Top navigation bar
    │   ├── App.js            # Main app component
    │   ├── App.css           # Dark mode styles
    │   ├── AuthContext.js    # User state management
    │   ├── api.js            # API calls to backend
    │   ├── index.js          # React entry point
    │   └── index.css         # Global styles
    └── package.json          # Node dependencies
```

---

## 🔍 How It Works (Technical Details)

### **Backend (FastAPI)**

1. **Authentication Flow:**
   - User registers → Password is hashed with bcrypt → Saved to MongoDB
   - User logs in → Password verified → JWT token generated
   - Token sent with every request → Validated in `get_current_user()`

2. **Image Processing:**
   ```python
   # User uploads image → Sent as multipart/form-data
   # Backend reads bytes → Passes to ML model
   # Model returns segmentation mask (PIL Image)
   # Both images converted to Base64 strings
   # Saved to MongoDB with user_id reference
   ```

3. **Database Schema:**
   ```python
   # Users Collection
   {
     "_id": ObjectId,
     "email": str,
     "password": str (hashed),
     "name": str,
     "created_at": datetime
   }
   
   # Segmentations Collection
   {
     "_id": ObjectId,
     "user_id": str,
     "filename": str,
     "original_image": str (Base64),
     "segmentation_mask": str (Base64),
     "created_at": datetime
   }
   ```

### **Frontend (React)**

1. **State Management:**
   - `AuthContext` stores user login state globally
   - JWT token saved in `localStorage`
   - Token automatically attached to all API requests

2. **Component Hierarchy:**
   ```
   App
   ├── AuthContext (wraps everything)
   └── AppContent
       ├── Login / Register (if not logged in)
       └── Dashboard (if logged in)
           ├── Navbar
           ├── Upload Section
           ├── Results Display
           └── History Grid
   ```

3. **API Communication:**
   ```javascript
   // All requests go through axios instance
   // JWT token automatically added via interceptor
   // Base64 images displayed using data URIs
   ```

---

## 🛠️ Customization Guide

### **Change Colors**
Edit `client/src/App.css`:
```css
/* Main gradient */
background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);

/* Accent color (change #00d4ff to your color) */
background: linear-gradient(90deg, #00d4ff, #7b2ff7);
```

### **Add More Fields to User Profile**
1. Update `server/models.py` → Add field to `UserRegister`
2. Update `server/main.py` → Save field in `/auth/register`
3. Update `client/src/components/Register.js` → Add input field

### **Store Images in Files Instead of Database**
Replace Base64 storage in `server/main.py`:
```python
# Save to disk
with open(f"uploads/{filename}", "wb") as f:
    f.write(file_bytes)

# Save only path in MongoDB
doc = {"user_id": user_id, "image_path": f"uploads/{filename}"}
```

### **Use Your Custom .pth Model**
Edit `server/ml_model.py`:
```python
# Define your model architecture
class MySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your layers here
    
    def forward(self, x):
        # Your forward pass
        return x

# Load in __init__
self.model = MySegmentationModel()
self.model.load_state_dict(torch.load("path/to/your_model.pth"))
self.model.eval()

# Update segment_image() method with your preprocessing
```

---

## 🐛 Troubleshooting

### **MongoDB Connection Error**
```
Error: MongoClient connection failed
```
**Solution:**
- Make sure MongoDB is running: `net start MongoDB` (Windows)
- Check connection string in `.env` file
- If using Atlas, whitelist your IP address

### **Module Not Found Errors**
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution:**
```powershell
# Backend
cd server
.\venv\Scripts\Activate
pip install -r requirements.txt

# Frontend
cd client
npm install
```

### **CORS Errors in Browser**
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```
**Solution:**
- Make sure backend is running on port 8000
- Check `FRONTEND_URL` in `.env` matches React port (3000)

### **Model Loading Fails**
```
RuntimeError: Error(s) in loading state_dict
```
**Solution:**
- Make sure model architecture matches the saved weights
- Check file path is correct
- If using .pth, you need the model class definition

### **Port Already in Use**
```
ERROR: [Errno 10048] error while attempting to bind on address
```
**Solution:**
```powershell
# Kill process on port 8000 (backend)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill process on port 3000 (frontend)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 📊 API Endpoints Reference

### **Authentication**
- `POST /auth/register` - Create new account
- `POST /auth/login` - Login existing user
- `GET /auth/me` - Get current user (requires token)

### **Segmentation**
- `POST /segment` - Upload image and run model (requires token)
- `GET /history` - Get all segmentations for user (requires token)
- `GET /segmentation/{id}` - Get specific segmentation (requires token)

### **Health Check**
- `GET /` - Check if API is running

---

## 🚀 Production Deployment

### **For Backend:**
1. Use a production WSGI server (Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. Set strong `SECRET_KEY` in `.env`

3. Use MongoDB Atlas (cloud) instead of local MongoDB

4. Add environment variables for production

### **For Frontend:**
1. Build production bundle:
   ```bash
   npm run build
   ```

2. Serve the `build/` folder with Nginx or similar

3. Update API URL in `src/api.js` to production backend URL

---

## 💡 Next Steps / Ideas

- [ ] Add image download button for segmentation results
- [ ] Add confidence scores for each segment
- [ ] Support multiple models (let user choose)
- [ ] Add image filters/preprocessing options
- [ ] Export results as JSON
- [ ] Add batch processing (upload multiple images)
- [ ] Add model comparison feature
- [ ] Dark/light mode toggle

---

## 🤝 Need Help?

If you get stuck, check:
1. Make sure MongoDB is running
2. Both servers (backend + frontend) are running
3. Check browser console for errors (F12)
4. Check backend terminal for error messages

---

## 📝 License

This project is open source and available for educational purposes.

---

**Built with ❤️ using FastAPI, React, and PyTorch**
