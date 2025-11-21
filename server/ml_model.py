"""
Machine Learning model loader and inference.
This file loads your PyTorch models when the server starts.
"""
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np
import os
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class SegmentationModel:
    """Wrapper class for your trained Martian segmentation model"""
    
    def __init__(self):
        print("🔄 Loading Martian SegFormer segmentation model...")
        
        # ============= LOAD YOUR TRAINED SEGFORMER MODEL =============
        # Define the path to your .pth model file
        model_path = "best_model.pth"  # Change this to your actual filename
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Model file not found: {model_path}\n"
                f"Please place your .pth file in the server/ folder and update the path in ml_model.py"
            )
        
        # ============= LOAD SEGFORMER MODEL ARCHITECTURE =============
        # This matches your training code exactly
        NUM_CLASSES = 10
        
        # Load the SegFormer architecture (same as training)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
        
        # Load your trained weights
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ Loaded SegFormer model weights successfully")
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
            raise
        
        # Load the image processor (same as training)
        self.image_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/mit-b0",
            do_resize=True,
            size={"height": 512, "width": 512},
            do_normalize=True
        )
        
        # Define class labels for Martian terrain (EXACT match with training code)
        self.class_labels = {
            0: "Sky",
            1: "Ridge",
            2: "Soil",
            3: "Sand",
            4: "Bedrock",
            5: "Rock",
            6: "Rover",
            7: "Trace",
            8: "Hole",
            9: "Class_9"
        }
        
        # Define colors for each class (RGB values)
        self.class_colors = {
            0: [135, 206, 235],   # Sky - Light blue
            1: [139, 90, 43],     # Ridge - Brown
            2: [160, 82, 45],     # Soil - Sienna brown
            3: [194, 178, 128],   # Sand - Sandy beige
            4: [105, 90, 75],     # Bedrock - Dark brown
            5: [105, 105, 105],   # Rock - Gray
            6: [255, 140, 0],     # Rover - Orange
            7: [255, 255, 150],   # Trace - Light yellow
            8: [50, 50, 50],      # Hole - Dark gray
            9: [200, 200, 200]    # Class 9 - Light gray
        }
        
        print("✅ Martian SegFormer model loaded successfully!")
        print(f"📊 Model supports {len(self.class_labels)} classes")
    
    def segment_image(self, image_bytes: bytes) -> Image.Image:
        """
        Run segmentation on uploaded Martian image using SegFormer.
        
        Args:
            image_bytes: Raw bytes of uploaded image
        
        Returns:
            PIL Image of the segmentation mask with legend
        """
        # Load image
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = original_image.size
        
        # Preprocess using SegFormer's image processor
        inputs = self.image_processor(images=original_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        
        # ============= RUN SEGFORMER INFERENCE =============
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # Shape: [batch, num_classes, height, width]
        
        # Upsample to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        )
        
        # Get class predictions
        predictions = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # Create colored segmentation mask
        segmentation_mask = self._create_colored_mask(predictions)
        
        # Add legend to the mask
        result_with_legend = self._add_legend(segmentation_mask, predictions)
        
        return result_with_legend
    
    def _create_colored_mask(self, predictions: np.ndarray) -> Image.Image:
        """
        Convert class predictions to colored visualization.
        
        Args:
            predictions: 2D array of class indices
        
        Returns:
            RGB image with colors mapped to classes
        """
        h, w = predictions.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each class to its color
        for class_id, color in self.class_colors.items():
            mask = predictions == class_id
            colored_mask[mask] = color
        
        return Image.fromarray(colored_mask)
    
    def _add_legend(self, mask_image: Image.Image, predictions: np.ndarray) -> Image.Image:
        """
        Add legend showing class names and colors to the segmentation result.
        
        Args:
            mask_image: Colored segmentation mask
            predictions: 2D array of class predictions
        
        Returns:
            Image with legend overlay
        """
        # Find which classes are present in the image
        unique_classes = np.unique(predictions)
        
        # Calculate legend dimensions
        legend_height = len(unique_classes) * 40 + 40  # 40px per class + padding
        legend_width = 250
        
        # Create new image with space for legend
        img_width, img_height = mask_image.size
        result_width = img_width + legend_width
        result_height = max(img_height, legend_height)
        
        result_image = Image.new('RGB', (result_width, result_height), color=(20, 20, 20))
        result_image.paste(mask_image, (0, 0))
        
        # Draw legend
        draw = ImageDraw.Draw(result_image)
        
        # Try to use a nice font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            title_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw legend title
        draw.text((img_width + 10, 10), "Legend", fill=(255, 255, 255), font=title_font)
        
        # Draw each class present in the image
        y_offset = 50
        for class_id in unique_classes:
            if class_id in self.class_labels:
                # Draw color box
                box_x = img_width + 10
                box_y = y_offset
                color = tuple(self.class_colors[class_id])
                draw.rectangle(
                    [box_x, box_y, box_x + 30, box_y + 25],
                    fill=color,
                    outline=(255, 255, 255),
                    width=2
                )
                
                # Draw class label
                label = self.class_labels[class_id]
                
                # Calculate percentage of image this class covers
                percentage = (predictions == class_id).sum() / predictions.size * 100
                text = f"{label} ({percentage:.1f}%)"
                
                draw.text((box_x + 40, box_y + 5), text, fill=(255, 255, 255), font=font)
                
                y_offset += 40
        
        return result_image

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to Base64 string for sending to frontend"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Global model instance (loaded lazily on first request)
_model_instance = None

def get_model():
    """Get or initialize the model instance (lazy loading)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = SegmentationModel()
    return _model_instance
