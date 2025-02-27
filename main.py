from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import time
import sys
import json

# Simple image tagging API
app = FastAPI(
    title="Image Tagging API",
    description="API service for analyzing images and providing tags",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Predefined tags organized by color and content
COLOR_TAGS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "yellow": np.array([255, 255, 0]),
    "cyan": np.array([0, 255, 255]),
    "magenta": np.array([255, 0, 255]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
    "gray": np.array([128, 128, 128]),
    "orange": np.array([255, 165, 0]),
    "purple": np.array([128, 0, 128]),
    "pink": np.array([255, 192, 203]),
    "brown": np.array([165, 42, 42]),
}

# Basic content tags based on image properties
CONTENT_TAGS = [
    "portrait", "landscape", "close-up", "wide-shot", "indoor", "outdoor",
    "bright", "dark", "colorful", "monochrome", "high_contrast", "low_contrast",
    "sharp", "blurry", "textured", "smooth", "detailed", "simple",
]

# Gender tags
GENDER_TAGS = ["woman", "female", "girl"]

@app.on_event("startup")
async def startup_event():
    """Startup event for the application."""
    print("Image Tagging API is starting up...")

@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "Image Tagging API is running",
    }

def analyze_image_content(image):
    """Simple image content analysis using color and basic properties."""
    # Convert image to RGB and resize
    image = image.convert("RGB")
    
    # Get image dimensions
    width, height = image.size
    
    # Determine orientation
    orientation = "portrait" if height > width else "landscape" if width > height else "square"
    
    # Get the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.5:
        shot_type = "extreme_portrait"
    elif aspect_ratio < 0.8:
        shot_type = "portrait"
    elif aspect_ratio < 1.2:
        shot_type = "close-up"
    elif aspect_ratio < 2:
        shot_type = "landscape"
    else:
        shot_type = "wide-shot"
        
    # Convert to numpy array for analysis
    img_array = np.array(image)
    
    # Calculate average brightness
    avg_brightness = np.mean(img_array)
    brightness = "bright" if avg_brightness > 150 else "dark" if avg_brightness < 80 else "medium_light"
    
    # Calculate color variance for determining colorfulness
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    color_variance = np.var(r) + np.var(g) + np.var(b)
    colorfulness = "colorful" if color_variance > 5000 else "monochrome" if color_variance < 1000 else "moderate_color"
    
    # Calculate average color
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Find the closest color tag
    closest_color = None
    min_distance = float('inf')
    for color_name, color_value in COLOR_TAGS.items():
        distance = np.linalg.norm(avg_color - color_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    # Calculate edge data for sharpness
    from PIL import ImageFilter
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    edge_intensity = np.mean(edges_array)
    sharpness = "sharp" if edge_intensity > 30 else "blurry" if edge_intensity < 15 else "medium_sharpness"
    
    # Estimate if image has a person (very basic heuristic)
    # In reality, this would require a proper model
    skin_tones = ["pink", "brown", "orange"]
    has_face = closest_color in skin_tones and shot_type in ["close-up", "portrait"]
    
    # Add automatic clothing and style guesses based on color
    clothing_tags = []
    if has_face:
        clothing_tags = ["clothing", "outfit"]
        if closest_color in ["black", "gray", "white"]:
            clothing_tags.append("formal")
        elif closest_color in ["red", "pink", "purple"]:
            clothing_tags.append("stylish")
        
    # Return tags
    tags = [
        orientation,
        shot_type,
        brightness,
        colorfulness,
        closest_color,
        sharpness
    ]
    
    # Add person-related tags if likely a portrait
    if has_face:
        tags.extend(GENDER_TAGS)
        tags.extend(clothing_tags)
        
    # Remove duplicates and return
    return list(set(tags))

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image and return tags.
    
    - **file**: The image file to analyze
    """
    try:
        # Read image data
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Analyze image content
        tags = analyze_image_content(image)
        
        # Create structured tags with confidence scores
        tags_with_confidence = [
            {"name": tag, "confidence": 0.9 - (i * 0.05)}  # Simulated confidence
            for i, tag in enumerate(tags)
        ]
        
        processing_time = time.time() - start_time
        
        return {
            "tags": tags_with_confidence,
            "tag_names": tags,
            "tag_count": len(tags),
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)