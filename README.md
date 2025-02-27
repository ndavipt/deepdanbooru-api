# DeepDanbooru API Service

This is a standalone API service that uses DeepDanbooru to analyze images and return tags.

## Features

- REST API endpoints for image analysis
- Uses DeepDanbooru for accurate image tagging
- Configurable confidence threshold
- Optimized for deployment on Render

## API Endpoints

### GET /

Root endpoint to verify the API is running.

### GET /health

Health check endpoint.

### POST /analyze

Analyze an image using DeepDanbooru and return the tags.

**Parameters:**
- `file`: The image file to analyze (multipart/form-data)
- `threshold`: (Optional) The confidence threshold for tags (default: 0.5)

**Example Response:**
```json
{
  "tags": [
    {"name": "1girl", "confidence": 0.9998},
    {"name": "solo", "confidence": 0.9994},
    ...
  ],
  "tag_names": ["1girl", "solo", ...],
  "tag_count": 25,
  "processing_time_seconds": 1.23
}
```

## Deployment

This service is designed to be deployed on Render. It requires:
- Standard instance (at least 1GB RAM)
- Docker environment

## Local Development

To run the service locally:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI application:
   ```
   uvicorn main:app --reload
   ```

3. Access the API at http://localhost:8000