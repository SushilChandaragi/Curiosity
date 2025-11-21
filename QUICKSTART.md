# Quick Start Guide

## For Beginners - Start Here! 👇

### 1️⃣ Install MongoDB (5 minutes)
**Windows:**
```powershell
# Download from: https://www.mongodb.com/try/download/community
# Run the installer (keep clicking Next)
# Start MongoDB:
net start MongoDB
```

**OR use MongoDB Atlas (Cloud - Even Easier!):**
- Go to https://www.mongodb.com/cloud/atlas
- Sign up free → Create cluster → Get connection string
- Skip local MongoDB installation!

---

### 2️⃣ Start Backend (Python Server)
```powershell
# Open PowerShell in project folder
cd C:\Users\SushilSC\Desktop\Curiosity\server

# Setup (only needed first time)
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
Copy-Item .env.example .env

# Edit .env file - change SECRET_KEY to something random!
notepad .env

# Start server
uvicorn main:app --reload
```

**You should see:** `✅ Models loaded successfully!`

---

### 3️⃣ Start Frontend (React Website)
Open a **NEW** PowerShell window:
```powershell
cd C:\Users\SushilSC\Desktop\Curiosity\client

# Setup (only needed first time)
npm install

# Start website
npm start
```

Browser opens automatically to http://localhost:3000

---

### 4️⃣ Use the App
1. **Sign Up** - Create account with email/password
2. **Upload Image** - Click "Choose Image" → Select photo
3. **Run Model** - Click "Run Model" button
4. **See Results** - View original + segmentation side-by-side
5. **Check History** - Scroll down to see all past uploads

---

## Daily Use (After First Setup)

**Every time you want to use the app:**

**Terminal 1 (Backend):**
```powershell
cd C:\Users\SushilSC\Desktop\Curiosity\server
.\venv\Scripts\Activate
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**
```powershell
cd C:\Users\SushilSC\Desktop\Curiosity\client
npm start
```

---

## What Each File Does

### Backend (server/)
- **main.py** - Main API server (handles login, image upload, database)
- **ml_model.py** - **← EDIT THIS to use your .pth model!**
- **auth.py** - Password hashing & JWT tokens
- **database.py** - MongoDB connection
- **config.py** - Settings (loads from .env)
- **.env** - Secret keys & database URL (create from .env.example)

### Frontend (client/src/)
- **App.js** - Main app component (decides login vs dashboard)
- **components/Dashboard.js** - Upload page + results + history
- **components/Login.js** - Login form
- **components/Register.js** - Sign up form
- **api.js** - Talks to backend API
- **App.css** - **← EDIT THIS to change colors/design!**

---

## Using Your Own .pth Model

**Step 1:** Put your model file in `server/` folder

**Step 2:** Edit `server/ml_model.py`:

```python
# At the top, import your model class
from your_model_file import YourModelClass

# In __init__ method (line ~15), uncomment and change:
self.custom_model = YourModelClass()
self.custom_model.load_state_dict(torch.load("your_model.pth"))
self.custom_model.eval()

# In segment_image method (line ~45), replace inference code:
with torch.no_grad():
    # Your preprocessing
    input_tensor = your_preprocessing(image)
    
    # Run your model
    output = self.custom_model(input_tensor)
    
    # Your postprocessing
    mask = your_postprocessing(output)
```

**Step 3:** Restart backend server

---

## Common Issues

**"MongoDB connection failed"**
→ Start MongoDB: `net start MongoDB`

**"Module not found"**
→ Reinstall: `pip install -r requirements.txt`

**"Port already in use"**
→ Close other terminals, or change port in code

**"Models failed to load"**
→ Check your model file path in ml_model.py

---

## Full Documentation
See [README.md](README.md) for complete details!
