FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with increased timeout
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Create model directory
RUN mkdir -p deepdanbooru_model

# Copy only the necessary files for model initialization
COPY main.py .

# Download model files during build using the download_file method
RUN python -c "import sys; \
    sys.path.append('.'); \
    from main import download_file; \
    download_file('https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/model-resnet_custom_v3.h5', 'deepdanbooru_model/model-resnet_custom_v3.h5'); \
    download_file('https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/tags.txt', 'deepdanbooru_model/tags.txt'); \
    import os; \
    model_size = os.path.getsize('deepdanbooru_model/model-resnet_custom_v3.h5'); \
    tags_size = os.path.getsize('deepdanbooru_model/tags.txt'); \
    print(f'Model size: {model_size} bytes, Tags size: {tags_size} bytes');"

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE ${PORT:-8000}

# Command to run the application
CMD ["python", "main.py"]