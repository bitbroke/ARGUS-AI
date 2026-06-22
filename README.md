---
title: Flipkart Gridlock
emoji: 🚦
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Argus AI

> Spatial Intelligence & Real-time Urban Anomaly Resolution Platform


Argus AI is a cutting-edge platform designed to monitor, predict, and resolve urban traffic anomalies. By combining real-time computer vision with a custom Graph Neural Network (ST-GCN + Mamba), Argus AI provides city operators with instantaneous insights and predictive cascading delay analysis.

---

## 🌟 Key Features

- **Real-Time Computer Vision**: YOLOv8-powered object detection running on live camera feeds to identify congestion and anomalies.
- **Predictive Graph Modeling**: A custom Spatio-Temporal Graph Convolutional Network (ST-GCN) using Mamba state-space blocks to predict traffic flow across 2,290 nodes in Koramangala, Bangalore.
- **Ripple Effect Simulation**: Instantly calculate how a delay at one intersection cascades through the entire urban grid.
- **Automated Dispatch**: Simulate emergency unit dispatching to resolve anomalies with accurate ETA predictions.

## 🏗️ Architecture

Argus AI consists of three main pillars:
1. **Frontend**: Next.js 14 web application featuring a highly interactive WebGL map built with Deck.gl and MapLibre.
2. **Backend**: A high-performance FastAPI server managing global state, polling models, and providing REST APIs.
3. **Machine Learning**: TensorFlow/Keras-based GNNs and OpenCV-based object detection pipelines.

*For complete architectural details, please see the [Project Description](project_description.md).*

## 🚀 Quick Start

To run Argus AI locally, you need two terminal windows: one for the backend, one for the frontend.

### 1. Start the Backend

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```
*Note: Make sure the ST-GCN weights (`nexus_flow_model_weights.weights.h5`) are in `backend/data/`.*

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

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
