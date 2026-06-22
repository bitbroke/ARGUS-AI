import os
import sys
import json
import time
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

import threading

from state import app_state
from graph_engine import calculate_ripple, reset_graph
from video_processor import process_frame

model_lock = threading.Lock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_stgcn import GraphAttentionLayer, SelectiveStateSpaceLayer, RestoreTemporalOutput, ReshapeTemporalInputSequence, TransposeExpandLayer, AsymmetricFocalRegressionLoss

app = FastAPI(title="Urban Command Center API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRAPH_FILE = os.path.join(DATA_DIR, "graph.json")
INDEX_FILE = os.path.join(DATA_DIR, "node_index.json")
MAPPING_FILE = os.path.join(DATA_DIR, "node_mapping.json")

def load_data():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, 'r') as f:
            data = json.load(f)
            app_state["graph_nodes"] = {n["node_id"]: n for n in data.get("nodes", [])}
            app_state["graph_edges"] = data.get("edges", [])
            app_state["graph_bbox"] = data.get("bbox", {})
            app_state["corridor_name"] = data.get("corridor_name", "Unknown")
            
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            app_state["node_index"] = json.load(f)
    else:
        app_state["node_index"] = {}

    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            app_state["node_mapping"] = json.load(f)
    else:
        app_state["node_mapping"] = []

load_data()

VIDEO_FILE = os.path.join(DATA_DIR, "loop.mp4")
video_cap = None
rf_pipeline = None
stgcn_model = None
A_tf = None

class RippleRequest(BaseModel):
    node_id: str

class DispatchRequest(BaseModel):
    node_id: str

class SimulateRequest(BaseModel):
    latitude: float
    longitude: float
    hour_of_day: int
    day_of_week: int

@app.on_event("startup")
def startup_event():
    global video_cap, rf_pipeline, stgcn_model, A_tf
    
    rf_path = os.path.join(DATA_DIR, "random_forest_pipeline.joblib")
    if os.path.exists(rf_path):
        try:
            rf_pipeline = joblib.load(rf_path)
            print("✅ Random Forest loaded.")
        except Exception as e:
            print(f"Failed to load RF: {e}")
            
    adj_path = os.path.join(DATA_DIR, "adjacency_matrix.npy")
    weights_path = os.path.join(DATA_DIR, "nexus_flow_model_weights.weights.h5")
    
    if os.path.exists(adj_path) and os.path.exists(weights_path):
        try:
            A_tf = np.load(adj_path)
            num_nodes = A_tf.shape[0]
            seq_len, pred_len, num_features = 12, 3, 3
            
            input_layer = tf.keras.layers.Input(shape=(seq_len, num_nodes, num_features), name='Spatio_Temporal_Input')
            x = tf.keras.layers.TimeDistributed(GraphAttentionLayer(units=16, adjacency_matrix=A_tf))(input_layer)
            x = ReshapeTemporalInputSequence()(x)
            x = SelectiveStateSpaceLayer(state_dim=64, output_dim=32)(x)
            x = RestoreTemporalOutput(num_nodes=num_nodes)(x)
            x = tf.keras.layers.Dense(pred_len)(x)
            output_layer = TransposeExpandLayer()(x)
            
            stgcn_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='Nexus_Flow_ST_GCN_Mamba')
            dummy_input = tf.zeros((1, seq_len, num_nodes, num_features))
            stgcn_model(dummy_input)
            stgcn_model.load_weights(weights_path)
            print("✅ Nexus Flow ST-GCN loaded.")
        except Exception as e:
            print(f"Failed to load ST-GCN: {e}")
    else:
        print("Model files not found!")
        
    print("Application startup complete.")
    
    if os.path.exists(VIDEO_FILE):
        video_cap = cv2.VideoCapture(VIDEO_FILE)

@app.post("/simulate")
def simulate(req: SimulateRequest):
    if not (0 <= req.hour_of_day <= 23):
        raise HTTPException(status_code=400, detail="hour_of_day must be between 0 and 23")
    if not (0 <= req.day_of_week <= 6):
        raise HTTPException(status_code=400, detail="day_of_week must be between 0 and 6")
        
    if stgcn_model is None or A_tf is None:
        raise HTTPException(status_code=500, detail="ST-GCN model not loaded")
        
    num_nodes = A_tf.shape[0]
    seq_len, num_features = 12, 3
    
    X_input = np.ones((1, seq_len, num_nodes, num_features), dtype=np.float32)
    traffic_base = 0.5 + 0.5 * np.sin((req.hour_of_day - 6) * np.pi / 12)
    X_input = X_input * traffic_base
    
    with model_lock:
        preds_tensor = stgcn_model(X_input, training=False)
        preds = preds_tensor.numpy()
    future_volumes = preds[0, 0, :, 0]
    
    nodes_payload = []
    node_index = app_state.get("node_index", {})
    index_to_node = {int(v): k for k, v in node_index.items()}
    graph_nodes = app_state.get("graph_nodes", {})
    
    for i in range(num_nodes):
        node_id = index_to_node.get(i)
        if not node_id:
            continue
        
        node_data = graph_nodes.get(node_id, {})
        volume = float(future_volumes[i])
        congested = bool(volume < 20.0)
        
        # If node has been dispatched, override congestion in simulation
        if node_id in app_state.get("dispatch_log", []):
            congested = False
            
        nodes_payload.append({
            "node_id": node_id,
            "lat": node_data.get("lat", req.latitude),
            "lon": node_data.get("lon", req.longitude),
            "predicted_volume": volume,
            "congested": congested
        })
        
    return {
        "nodes": nodes_payload,
        "metadata": {
            "hour": req.hour_of_day,
            "day": req.day_of_week
        }
    }

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "frame_count": app_state["frame_counter"]}

@app.get("/api/graph")
def get_graph():
    return {
        "nodes": list(app_state["graph_nodes"].values()),
        "edges": app_state["graph_edges"],
        "corridor_name": app_state.get("corridor_name", "Unknown"),
        "bbox": app_state.get("graph_bbox", {})
    }

@app.get("/api/anomalies")
def get_anomalies():
    global video_cap
    if not video_cap or not video_cap.isOpened():
        return {"anomalies": app_state["current_anomalies"], "timestamp": time.time(), "frame_index": app_state["frame_counter"]}
    ret, frame = video_cap.read()
    if not ret:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_cap.read()
        if not ret:
            return {"anomalies": [], "timestamp": time.time(), "frame_index": app_state["frame_counter"]}
    app_state["frame_counter"] += 1
    if app_state["frame_counter"] % 5 == 0:
        anomalies = process_frame(app_state, frame, app_state["node_mapping"])
        if anomalies:
            app_state["current_anomalies"] = anomalies
    return {"anomalies": app_state["current_anomalies"], "timestamp": time.time(), "frame_index": app_state["frame_counter"]}

@app.post("/api/ripple")
def trigger_ripple(req: RippleRequest):
    node_id = req.node_id
    if node_id not in app_state["graph_nodes"]:
        raise HTTPException(status_code=404, detail="Node not found")
    result = calculate_ripple(node_id, app_state)
    app_state["active_ripple"] = {"source_node": node_id, "start_time": time.time()}
    return result

@app.post("/api/dispatch")
def dispatch_unit(req: DispatchRequest):
    node_id = req.node_id
    if node_id not in app_state["graph_nodes"]: raise HTTPException(status_code=404, detail="Node not found")
    dispatch_id = f"unit-{int(time.time())}"
    dispatch_info = {"dispatch_id": dispatch_id, "target_node": node_id, "status": "en_route", "eta_minutes": 15, "timestamp": time.time()}
    app_state["dispatch_log"].append(node_id)
    app_state["current_anomalies"] = [a for a in app_state["current_anomalies"] if a["node_id"] != node_id]
    reset_edges = reset_graph(app_state)
    return {"success": True, "node_id": node_id, "dispatch": dispatch_info, "reset_edges": reset_edges, "message": "Dispatched successfully"}

@app.post("/api/reset")
def reset_state():
    reset_graph(app_state)
    app_state["active_ripple"] = None
    app_state["current_anomalies"] = []
    return {"status": "success"}

@app.get("/api/diagnostics")
def get_diagnostics():
    num_nodes = A_tf.shape[0] if A_tf is not None else 0
    model_params = stgcn_model.count_params() if stgcn_model else 0
    return {
        "model": "Nexus Flow v2 (GAT + Mamba)",
        "loss": "Asymmetric Focal Loss",
        "status": "Training completed",
        "nodes": num_nodes,
        "parameters": model_params,
        "is_model_loaded": stgcn_model is not None
    }

@app.get("/api/predict/{node_id}")
def get_node_prediction(node_id: str, hour_of_day: int = 12):
    if stgcn_model is None or A_tf is None:
        raise HTTPException(status_code=500, detail="ST-GCN model not loaded")
        
    node_index = app_state.get("node_index", {})
    # reverse mapping: node_id (string) -> matrix_index (int)
    # The file shows index_to_node = {int(v): k ...} so node_index is {node_id: index}
    if node_id not in node_index:
        raise HTTPException(status_code=404, detail="Node not found in graph")
        
    node_idx = int(node_index[node_id])
    num_nodes = A_tf.shape[0]
    seq_len, num_features = 12, 3
    
    X_input = np.ones((1, seq_len, num_nodes, num_features), dtype=np.float32)
    traffic_base = 0.5 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12)
    X_input = X_input * traffic_base
    
    with model_lock:
        preds_tensor = stgcn_model(X_input, training=False)
        preds = preds_tensor.numpy() # shape (1, pred_len, num_nodes, 1)
    # Extract prediction for the specific node across all predicted timesteps
    node_preds = preds[0, :, node_idx, 0].tolist()
    
    return {
        "node_id": node_id,
        "hour_of_day": hour_of_day,
        "predicted_volumes": node_preds,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
