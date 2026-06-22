FROM python:3.11-slim

# 1. Install system graphics libraries required by OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Hugging Face Space user setup (runs with UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# 3. Memory-Safe Sequential Installation Hack to avoid out-of-memory errors on Hugging Face Spaces basic CPU
# Step 3a: Install core web & network libraries
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0 \
    python-multipart==0.0.9 \
    slowapi==0.1.9 \
    limits==3.12.0 \
    osmnx==1.9.3 \
    networkx==3.3 \
    numpy \
    joblib \
    scikit-learn

# Step 3b: Install lightweight CPU-only TensorFlow (avoids massive GPU binary extraction OOM)
RUN pip install --no-cache-dir tensorflow-cpu

# Step 3c: Install CPU-only PyTorch 2.5.1 and Torchvision (avoids massive 2.7GB GPU package download)
# Pinned to <2.6 because PyTorch 2.6 changed torch.load default to weights_only=True,
# which breaks ultralytics 8.2.0's model loading.
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 3d: Install headless OpenCV and Ultralytics YOLO (skips torch download since it's already installed)
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80
RUN pip install --no-cache-dir ultralytics==8.2.0

# 4. Pre-download YOLOv8n model inside the container to avoid runtime downloads
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Copy the backend application files into the working directory
COPY --chown=user backend/ /app/

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# 6. Start the FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
