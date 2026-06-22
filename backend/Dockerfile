FROM python:3.11-slim

# Install system dependencies for OpenCV and GLib
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-download YOLOv8n model
RUN pip install --no-cache-dir ultralytics==8.2.0
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
