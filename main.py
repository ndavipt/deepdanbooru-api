from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import json

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
            os.makedirs(model_path, exist_ok=True)
            
        # Check if model files exist, download if not
        model_file = os.path.join(model_path, "model-resnet_custom_v3.h5")
        tags_file = os.path.join(model_path, "tags.txt")
        
        if not os.path.exists(model_file) or os.path.getsize(model_file) < 1000000:
            print("Model file missing or incomplete. Downloading...")
            os.system(f"curl -L -o {model_file} https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/model-resnet_custom_v3.h5")
            
        if not os.path.exists(tags_file):
            print("Tags file missing. Downloading...")
            os.system(f"curl -L -o {tags_file} https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/tags.txt")
        
        # Load model with TensorFlow
        model = tf.keras.models.load_model(model_file, compile=False)
        print("Model loaded successfully")
        
        # Load tags
        tags_list = load_tags_from_project(model_path)
        print(f"Loaded {len(tags_list)} tags")
    except Exception as e:
        print(f"Error loading model: {e}")
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