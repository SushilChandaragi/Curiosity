# 🚀 Setting Up Your Martian Segmentation Model

## Step 1: Place Your Model File

Copy your trained `.pth` model file into the `server/` folder:

```
Curiosity/
└── server/
    ├── main.py
    ├── ml_model.py
    ├── martian_segmentation_model.pth  ← Put your .pth file here
    └── ...
```

## Step 2: Configure the Model

Open `server/ml_model.py` and update these settings:

### A. Update Model Path (Line 18)
```python
model_path = "your_actual_model_filename.pth"  # Change to your file name
```

### B. Update Input Size (Line 40)
Match the size your model was trained on:
```python
transforms.Resize((256, 256)),  # Change to your training size (e.g., 512, 512)
```

### C. Update Class Labels (Line 45)
Define the classes your model can detect:
```python
self.class_labels = {
    0: "Background",
    1: "Sand",
    2: "Bedrock", 
    3: "Big Rock",
    4: "Soil",
    # Add all your classes here
}
```

### D. Update Class Colors (Line 53)
Choose colors for visualization (RGB format):
```python
self.class_colors = {
    0: [0, 0, 0],         # Background - Black
    1: [194, 178, 128],   # Sand - Sandy color
    2: [139, 90, 43],     # Bedrock - Brown
    3: [105, 105, 105],   # Big Rock - Gray
    4: [160, 82, 45],     # Soil - Reddish brown
    # Match colors to your classes
}
```

## Step 3: Handle Different Model Types

### If your model was saved as a complete model:
```python
# No changes needed, the current code handles this
self.model = torch.load(model_path, map_location="cpu")
```

### If you only saved state_dict (weights):
Uncomment lines 32-36 in `ml_model.py` and import your model architecture:
```python
from your_model_architecture import YourModelClass
self.model = YourModelClass()
self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
self.model.eval()
```

## Step 4: Adjust Preprocessing (if needed)

If your model uses different normalization values:
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # Change to your training mean
    std=[0.229, 0.224, 0.225]     # Change to your training std
)
```

## Step 5: Test the Model

```powershell
cd server
.\venv\Scripts\Activate
uvicorn main:app --reload
```

You should see:
```
✅ Martian segmentation model loaded successfully!
📊 Model supports X classes
```

## What the Model Does

Your trained model will:
1. ✅ Take uploaded Martian images
2. ✅ Run semantic segmentation
3. ✅ Generate colored output with each terrain type
4. ✅ Add a legend showing:
   - Class names (Sand, Bedrock, etc.)
   - Colors used
   - Coverage percentage for each class

## Example Output

The result image will show:
- **Left side**: Segmented image with colors
- **Right side**: Legend with class names and percentages

Example:
```
[Segmented Image]  | Legend
                   | 🟫 Sand (45.2%)
                   | 🟤 Bedrock (30.1%)
                   | ⬜ Big Rock (15.3%)
                   | 🟠 Soil (9.4%)
```

## Troubleshooting

### Model file not found
```
❌ Model file not found: martian_segmentation_model.pth
```
**Fix**: Make sure your .pth file is in the `server/` folder and the filename matches in line 18 of `ml_model.py`

### Wrong input size
```
RuntimeError: size mismatch
```
**Fix**: Update the `Resize()` value to match your training input size

### Missing classes in legend
**Fix**: Make sure all class IDs in your model output are defined in `self.class_labels` dictionary

---

**Need help?** Check the class IDs your model outputs and match them exactly in the config!
