<div align="center">
  <h1>🚦 Argus AI</h1>
  <p><b>Enterprise Spatial Intelligence & Real-time Urban Traffic Orchestration Platform</b></p>
</div>

---

**Argus AI** is a state-of-the-art urban mobility platform engineered to monitor, predict, and autonomously orchestrate traffic anomaly resolutions at city-scale. By fusing edge-deployed real-time computer vision with our proprietary predictive Spatio-Temporal Graph Neural Networks (ST-GCN + Mamba), Argus AI equips traffic command centers and municipal operators with sub-second insights and deep predictive analysis of cascading gridlocks.

Built for scale, resilience, and actionable intelligence, Argus AI transcends traditional passive monitoring by modeling the future state of urban road networks and automating emergency response dispatch.

---

## 🌟 Core Capabilities

- **Real-Time Edge Vision (Computer Vision)**: Leverages YOLOv8 for sub-millisecond object detection on live CCTV and drone feeds, identifying localized congestion, accidents, and unauthorized encroachments instantly.
- **Predictive Network Modeling (ST-GCN)**: Utilizes a custom Spatio-Temporal Graph Convolutional Network enhanced with Mamba state-space blocks to accurately forecast traffic flow deterioration across complex urban grids (e.g., modeling thousands of nodes simultaneously).
- **Cascading Ripple Effect Simulation**: Proprietary graph algorithms instantly calculate how a micro-delay at a single intersection will cascade and paralyze neighboring arterial roads over the next 60 minutes.
- **Automated Dispatch & Mitigation**: Simulates and recommends dynamic mitigation strategies, including the automated dispatching of rapid-response units with traffic-aware ETA routing to neutralize anomalies before they propagate.

## 🏗️ Architecture

Argus AI consists of three main pillars:
1. **Frontend**: Next.js 14 web application featuring a highly interactive WebGL map built with Deck.gl and MapLibre.
2. **Backend**: A high-performance FastAPI server managing global state, polling models, and providing REST APIs.
3. **Machine Learning**: TensorFlow/Keras-based GNNs and OpenCV-based object detection pipelines.

*For complete architectural details, please see the [Project Description](project_description.md).*

## 🚀 Comprehensive Setup & Execution Guide

To run Argus AI locally, you need to set up the Python backend (including optional model training) and the Next.js frontend.

### Prerequisites
- **Python 3.9+** (for the backend & ML models)
- **Node.js 18+** and **npm** (for the frontend)
- (Optional) CUDA-enabled GPU for faster ST-GCN model training and YOLOv8 inference.

### 1. Train the ST-GCN Model (Optional)
If you do not have the pre-trained weights (`nexus_flow_model_weights.weights.h5`), you must train the model first. The training script will generate the `.keras` model and `.h5` weights and save them directly to `backend/data/`.

```bash
# From the root directory:
python -m pip install -r backend/requirements.txt
python train_stgcn.py
```

### 2. Start the Backend Server
The backend is a FastAPI application that serves the ML models and graph data.

```bash
# 1. Navigate to the backend directory
cd backend

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install the required Python packages
python -m pip install -r requirements.txt

# 4. Run the FastAPI server
python -m uvicorn main:app --port 8000 --reload
```
*The backend will now be running at [http://localhost:8000](http://localhost:8000). It will automatically load the YOLOv8 model and the ST-GCN weights from the `data/` directory.*

### 3. Start the Frontend Application
The frontend is a Next.js application that provides the interactive WebGL interface. Open a **new terminal window** for this step.

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start the development server
npm run dev
```

### 4. View the Application
Open your web browser and navigate to **[http://localhost:3000](http://localhost:3000)**. 
- The interactive map and dashboard will load.
- It will communicate with the backend at `localhost:8000` to fetch graph topologies, anomaly streams, and ripple effect predictions.

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check system status and model load state |
| `/api/graph` | GET | Fetch the complete network topology (nodes & edges) |
| `/api/anomalies` | GET | Poll for real-time anomalies detected by YOLO |
| `/simulate` | POST | Generate a full-network traffic volume prediction |
| `/api/ripple` | POST | Calculate cascading delay from a specific source node |
| `/api/dispatch` | POST | Dispatch a resolution unit to an anomaly |

## 🛠️ Technology Stack

- **UI**: Next.js, React, Tailwind CSS, Deck.gl, MapLibre GL
- **API**: FastAPI, Pydantic, Uvicorn
- **ML**: TensorFlow, Keras, Ultralytics YOLOv8, Scikit-Learn
- **Data**: OSMnx, NetworkX, NumPy, OpenCV
