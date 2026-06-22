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

# 3. Install requirements
COPY --chown=user backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre-download YOLOv8n model inside the container to avoid runtime downloads
RUN pip install --no-cache-dir ultralytics==8.2.0
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 5. Copy the backend application files into the working directory
COPY --chown=user backend/ /app/

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# 6. Start the FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
