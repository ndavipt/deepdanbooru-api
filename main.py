from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import json
import subprocess
import sys

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

def download_file(url, destination):
    """Download a file with better error handling."""
    print(f"Downloading {url} to {destination}")
    try:
        # First try with curl command
        result = subprocess.run(
            ["curl", "-L", "-o", destination, url],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"curl output: {result.stdout}")
        if result.stderr:
            print(f"curl stderr: {result.stderr}")
            
        # Verify file exists and has content
        if os.path.exists(destination):
            size = os.path.getsize(destination)
            print(f"File downloaded successfully. Size: {size} bytes")
            return True
        else:
            print(f"File doesn't exist after download")
            return False
    except subprocess.CalledProcessError as e:
        print(f"curl download failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        # Try with python's urllib as fallback
        try:
            import urllib.request
            print("Trying download with urllib")
            urllib.request.urlretrieve(url, destination)
            
            if os.path.exists(destination):
                size = os.path.getsize(destination)
                print(f"File downloaded with urllib. Size: {size} bytes")
                return True
            else:
                print("File still doesn't exist after urllib download")
                return False
        except Exception as urllib_e:
            print(f"urllib download also failed: {urllib_e}")
            return False
    except Exception as e:
        print(f"Unexpected error in download_file: {e}")
        return False

def load_tags_from_project(project_path):
    """Load tags from a DeepDanbooru project directory."""
    tags_path = os.path.join(project_path, 'tags.txt')
    if not os.path.exists(tags_path):
        raise Exception(f"Tags file not found at {tags_path}")
        
    with open(tags_path, 'r') as f:
        tags = [line.strip() for line in f.readlines()]
    return tags

@app.on_event("startup")
async def startup_event():
    """Load the DeepDanbooru model when the application starts."""
    global model, tags_list
    model_path = "deepdanbooru_model"
    
    try:
        print(f"Loading model from {model_path}...")
        # Verify model directory exists
        if not os.path.exists(model_path):
            print(f"Creating model directory {model_path}")
            os.makedirs(model_path, exist_ok=True)
            
        # Check if model files exist, download if not
        model_file = os.path.join(model_path, "model-resnet_custom_v3.h5")
        tags_file = os.path.join(model_path, "tags.txt")
        
        # Print the available disk space
        try:
            disk_usage = subprocess.check_output(["df", "-h", "."]).decode()
            print(f"Disk usage:\n{disk_usage}")
        except:
            print("Could not check disk usage")
        
        if not os.path.exists(model_file) or os.path.getsize(model_file) < 1000000:
            print("Model file missing or incomplete. Downloading...")
            model_url = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
            success = download_file(model_url, model_file)
            
            if not success:
                raise Exception("Failed to download model file")
                
            # Verify the downloaded file
            if os.path.exists(model_file):
                size = os.path.getsize(model_file)
                print(f"Downloaded model file size: {size} bytes")
                if size < 1000000:
                    raise Exception(f"Downloaded model file too small: {size} bytes")
            else:
                raise Exception("Model file not found after download attempt")
            
        if not os.path.exists(tags_file):
            print("Tags file missing. Downloading...")
            tags_url = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/tags.txt"
            success = download_file(tags_url, tags_file)
            
            if not success:
                raise Exception("Failed to download tags file")
        
        # Check file sizes and contents
        model_size = os.path.getsize(model_file) if os.path.exists(model_file) else 0
        tags_size = os.path.getsize(tags_file) if os.path.exists(tags_file) else 0
        print(f"Model file size: {model_size} bytes")
        print(f"Tags file size: {tags_size} bytes")
        
        # Check model file header to see if it's a valid HDF5 file
        with open(model_file, 'rb') as f:
            header = f.read(8)
            print(f"Model file header bytes: {header}")
            
        # Load model with TensorFlow
        print(f"Loading model from {model_file}")
        model = tf.keras.models.load_model(model_file, compile=False)
        print("Model loaded successfully")
        
        # Load tags
        tags_list = load_tags_from_project(model_path)
        print(f"Loaded {len(tags_list)} tags")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "DeepDanbooru API is running",
        "model_loaded": model is not None,
        "tags_count": len(tags_list) if tags_list else 0
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), threshold: float = 0.5):
    """
    Analyze an image using DeepDanbooru and return the tags.
    
    - **file**: The image file to analyze
    - **threshold**: The confidence threshold for tags (default: 0.5)
    """
    # Ensure model is loaded
    if model is None or tags_list is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read image data
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image
        image = image.convert("RGB").resize((512, 512))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Run model prediction
        predictions = model.predict(image_array)[0]
        
        # Get tags above threshold
        tags = [
            {"name": tags_list[i], "confidence": float(score)}
            for i, score in enumerate(predictions) 
            if score > threshold
        ]
        
        # Sort tags by confidence (descending)
        tags.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get just tag names for convenience
        tag_names = [tag["name"] for tag in tags]
        
        processing_time = time.time() - start_time
        
        return {
            "tags": tags,
            "tag_names": tag_names,
            "tag_count": len(tags),
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)