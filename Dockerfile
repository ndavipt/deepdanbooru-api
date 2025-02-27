FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Create model directory
RUN mkdir -p deepdanbooru_model

# Download model files during build for better caching
RUN curl -L -o deepdanbooru_model/model-resnet_custom_v3.h5 https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/model-resnet_custom_v3.h5 && \
    curl -L -o deepdanbooru_model/tags.txt https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/tags.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE ${PORT:-8000}

# Command to run the application
CMD ["python", "main.py"]