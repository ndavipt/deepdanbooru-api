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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p deepdanbooru_model

# Copy application code
COPY . .

# Expose the port
EXPOSE ${PORT:-8000}

# Command to run the application
CMD ["python", "main.py"]