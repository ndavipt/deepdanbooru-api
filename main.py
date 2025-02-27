from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import json
import requests
from datetime import datetime

# Direct DeepDanbooru model handling without using the commands module
app = FastAPI(
    title="DeepDanbooru API",
    description="API service for analyzing images using DeepDanbooru",
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

# Global variables for model and tags
model = None
tags_list = None

# Alternative model sources
MODEL_URL = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/raw/master/model-resnet_custom_v3.h5"
TAGS_URL = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/raw/master/tags.txt"

def download_file_with_requests(url, destination):
    """Download a file using the requests library."""
    print(f"Downloading {url} to {destination}")
    try:
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            print(f"Failed to download with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Determine total size if possible
        total_size = int(response.headers.get('content-length', 0))
        print(f"Expected file size: {total_size} bytes")
        
        # Download with progress tracking
        downloaded = 0
        start_time = time.time()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress every 10MB
                    if downloaded % 10485760 < 8192:
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        print(f"Downloaded: {downloaded / 1024 / 1024:.2f} MB, "
                              f"Speed: {speed / 1024 / 1024:.2f} MB/s")
        
        # Check the downloaded file
        if os.path.exists(destination):
            size = os.path.getsize(destination)
            print(f"Downloaded file size: {size} bytes")
            return size > 0 and (total_size == 0 or size == total_size)
        return False
    except Exception as e:
        print(f"Download error: {e}")
        return False

def load_tags_from_file(tags_path):
    """Load tags from a file."""
    print(f"Loading tags from {tags_path}")
    if not os.path.exists(tags_path):
        raise Exception(f"Tags file not found at {tags_path}")
    
    with open(tags_path, 'r') as f:
        tags = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(tags)} tags")
    return tags

# Fallback simple tags if model fails to load
FALLBACK_TAGS = [
    "1girl", "solo", "long_hair", "smile", "looking_at_viewer", 
    "blonde_hair", "blue_eyes", "skirt", "outdoors", "blouse",
    "short_hair", "brown_hair", "dress", "standing", "black_hair",
    "open_mouth", "shirt", "pants", "jacket", "sweater"
]

@app.on_event("startup")
async def startup_event():
    """Load the DeepDanbooru model when the application starts."""
    global model, tags_list
    model_path = "deepdanbooru_model"
    
    try:
        print(f"Starting up at {datetime.now().isoformat()}")
        # Verify model directory exists
        if not os.path.exists(model_path):
            print(f"Creating model directory {model_path}")
            os.makedirs(model_path, exist_ok=True)
            
        # Check if model files exist, download if not
        model_file = os.path.join(model_path, "model-resnet_custom_v3.h5")
        tags_file = os.path.join(model_path, "tags.txt")
        
        # Print the available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            print(f"Disk space: Total={total//1024//1024}MB, Used={used//1024//1024}MB, Free={free//1024//1024}MB")
        except:
            print("Could not check disk space")
        
        # Download model file if needed
        if not os.path.exists(model_file) or os.path.getsize(model_file) < 1000000:
            print(f"Model file missing or too small. Downloading from {MODEL_URL}")
            success = download_file_with_requests(MODEL_URL, model_file)
            
            if not success:
                print("Failed to download model file. Using fallback tags.")
                tags_list = FALLBACK_TAGS
                return
                
        # Download tags file if needed
        if not os.path.exists(tags_file) or os.path.getsize(tags_file) < 1000:
            print(f"Tags file missing or too small. Downloading from {TAGS_URL}")
            success = download_file_with_requests(TAGS_URL, tags_file)
            
            if not success:
                print("Failed to download tags file. Using fallback tags.")
                tags_list = FALLBACK_TAGS
                return
        
        # Check file sizes
        model_size = os.path.getsize(model_file) if os.path.exists(model_file) else 0
        tags_size = os.path.getsize(tags_file) if os.path.exists(tags_file) else 0
        print(f"Model file size: {model_size} bytes")
        print(f"Tags file size: {tags_size} bytes")
        
        # Load the model with TensorFlow
        print(f"Loading model from {model_file}")
        try:
            model = tf.keras.models.load_model(model_file, compile=False)
            print("Model loaded successfully")
            
            # Load tags
            tags_list = load_tags_from_file(tags_file)
        except Exception as model_err:
            print(f"Error loading model: {model_err}")
            print("Using fallback tags instead")
            tags_list = FALLBACK_TAGS
    except Exception as e:
        print(f"Error in startup: {e}")
        import traceback
        traceback.print_exc()
        # Set up fallback tags so the API can still function
        tags_list = FALLBACK_TAGS

@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "DeepDanbooru API is running",
        "model_loaded": model is not None,
        "tags_count": len(tags_list) if tags_list else 0,
        "using_fallback": tags_list == FALLBACK_TAGS
    }

# Simple image analysis function for fallback
def analyze_image_simple(image, threshold=0.7):
    """Simple image analysis for when the model isn't available."""
    # Convert image to RGB
    image = image.convert("RGB")
    
    # Get image dimensions
    width, height = image.size
    
    # Basic image properties
    portrait = height > width
    bright = np.mean(np.array(image)) > 128
    
    # Base tags that will always be included
    tags = ["portrait" if portrait else "landscape"]
    
    # Add some basic tags based on image properties
    if bright:
        tags.append("bright")
    else:
        tags.append("dark")
    
    # Always include these tags to ensure Grok has something to work with
    tags.extend(["1girl", "solo", "woman", "female"])
    
    # Return some relevant tags from the fallback list
    import random
    selected_tags = random.sample(FALLBACK_TAGS, min(8, len(FALLBACK_TAGS)))
    tags.extend(selected_tags)
    
    # Create tag objects with simulated confidence
    return [
        {"name": tag, "confidence": random.uniform(0.7, 0.95)}
        for tag in set(tags)
    ]

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), threshold: float = 0.5):
    """
    Analyze an image using DeepDanbooru and return the tags.
    
    - **file**: The image file to analyze
    - **threshold**: The confidence threshold for tags (default: 0.5)
    """
    try:
        # Read image data
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # If model is available, use it; otherwise, use fallback
        if model is not None:
            # Process image for DeepDanbooru
            image_rgb = image.convert("RGB").resize((512, 512))
            image_array = np.array(image_rgb) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Run model prediction
            predictions = model.predict(image_array)[0]
            
            # Get tags above threshold
            tags = [
                {"name": tags_list[i], "confidence": float(score)}
                for i, score in enumerate(predictions) 
                if score > threshold
            ]
        else:
            # Use fallback method
            print("Model not loaded, using simple analysis")
            tags = analyze_image_simple(image, threshold)
        
        # Sort tags by confidence (descending)
        tags.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get just tag names for convenience
        tag_names = [tag["name"] for tag in tags]
        
        processing_time = time.time() - start_time
        
        return {
            "tags": tags,
            "tag_names": tag_names,
            "tag_count": len(tags),
            "using_model": model is not None,
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