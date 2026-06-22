import os
import json
import sys
import numpy as np
import networkx as nx
from typing import List, Dict

# Import custom registered Keras layers from the separated modules
from layers import GraphAttentionLayer, GraphConvLayer, ReshapeTemporalInput, RestoreTemporalOutput, TransposeExpandLayer
from losses import AsymmetricFocalLoss
from mamba_block import SelectiveSSMBlock

import tensorflow as tf

# Global variables to cache model resources
model = None
adjacency_matrix = None
node_index = None

def load_stgcn_resources():
    """
    Loads model and adjacency resources. Cached after first load.
    """
    global model, adjacency_matrix, node_index
    if model is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        model_path = os.path.join(data_dir, "nexus_flow_v2.keras")
        adj_path = os.path.join(data_dir, "adjacency_matrix.npy")
        idx_path = os.path.join(data_dir, "node_index.json")
        
        if os.path.exists(model_path) and os.path.exists(adj_path) and os.path.exists(idx_path):
            try:
                print("ST-GCN v2: Loading trained model from", model_path)
                model = tf.keras.models.load_model(model_path)
                adjacency_matrix = np.load(adj_path)
                with open(idx_path, 'r') as f:
                    node_index = json.load(f)
                print("ST-GCN v2: Resources loaded successfully. Active nodes in network:", len(node_index))
            except Exception as e:
                print("ST-GCN Error: Failed to load resources:", str(e))
                model = None
        else:
            print("ST-GCN Warning: Model or adjacency files not found. Running Dijkstra fallback.")

def calculate_ripple_dijkstra(origin_node_id: str, edges: list, nodes: dict) -> dict:
    """
    Calculates the cascading traffic delay using Dijkstra decay (Fallback).
    """
    G = nx.Graph()
    for e in edges:
        G.add_edge(e['source_node_id'], e['target_node_id'], edge_id=e['edge_id'], length=e['length_m'])
        
    if origin_node_id not in G.nodes:
        return {
            "origin_node_id": origin_node_id,
            "affected_edges": [],
            "affected_nodes": [],
            "total_delay_minutes": 0.0,
            "affected_street_names": [],
            "calculation_method": "dijkstra-fallback"
        }
        
    lengths = nx.single_source_shortest_path_length(G, origin_node_id, cutoff=4)
    base_delay = 7.0
    affected_edges_result = []
    affected_nodes_result = []
    total_delay = 0.0
    affected_streets = set()
    
    for e in edges:
        d_src = lengths.get(e['source_node_id'], float('inf'))
        d_tgt = lengths.get(e['target_node_id'], float('inf'))
        min_d = min(d_src, d_tgt)
        
        if min_d == 0:
            e['flow_state'] = "anomaly"
            e['color_rgb'] = [255, 0, 255]
            delay = base_delay
        elif min_d in [1, 2, 3]:
            e['flow_state'] = "bottleneck"
            e['color_rgb'] = [255, 94, 0]
            delay = base_delay * (0.85 ** min_d)
        elif min_d == 4:
            e['flow_state'] = "ripple"
            e['color_rgb'] = [255, 94, 0, 128]
            delay = base_delay * (0.85 ** min_d)
        else:
            e['flow_state'] = "healthy"
            e['color_rgb'] = [0, 243, 255]
            delay = 0.0
            
        e['delay_minutes'] = delay
        if delay > 0:
            total_delay += delay
            affected_edges_result.append(e)
            if e['street_name'] and e['street_name'] != "Unknown Segment":
                affected_streets.add(e['street_name'])
                
    for n_id, n_data in nodes.items():
        if n_id == origin_node_id:
            n_data['flow_state'] = 'anomaly'
            affected_nodes_result.append(n_data)
        elif n_id in lengths and lengths[n_id] <= 4:
            n_data['flow_state'] = 'bottleneck'
            affected_nodes_result.append(n_data)
        else:
            n_data['flow_state'] = 'healthy'
            
    return {
        "origin_node_id": origin_node_id,
        "affected_edges": edges,
        "affected_nodes": affected_nodes_result,
        "total_delay_minutes": total_delay,
        "affected_street_names": list(affected_streets),
        "calculation_method": "dijkstra-fallback"
    }

def calculate_ripple(origin_node_id: str, app_state: dict) -> dict:
    """
    Calculates the cascading traffic delay using the trained Spatio-Temporal GCN model.
    """
    load_stgcn_resources()
    
    edges = app_state.get("graph_edges", [])
    nodes = app_state.get("graph_nodes", {})
    
    if model is None or node_index is None:
        # Fallback to original Dijkstra decay if TF model could not load
        return calculate_ripple_dijkstra(origin_node_id, edges, nodes)
        
    num_nodes = len(node_index)
    origin_idx = node_index.get(origin_node_id)
    
    if origin_idx is None:
        print(f"ST-GCN Warning: Origin node {origin_node_id} not found in model index. Falling back.")
        return calculate_ripple_dijkstra(origin_node_id, edges, nodes)
        
    # Construct input sequence: (1, seq_len, num_nodes, num_features)
    seq_len = 12
    
    # 1. Synthesize base traffic volume for H-11 to H (diurnal cycle)
    # Assume current hour is 12 (midday)
    H = 12
    hours = [(H - 11 + i) % 24 for i in range(seq_len)]
    base_traffic_factor = [
        0.1 + 0.3 * np.sin((h - 4) * np.pi / 12) + 0.4 * np.exp(-((h - 9)**2)/8) + 0.4 * np.exp(-((h - 18)**2)/8)
        for h in hours
    ]
    
    # Node degrees for traffic scale (sum of rows in adjacency matrix)
    node_degrees = np.sum(adjacency_matrix, axis=1)
    node_traffic_scales = node_degrees * 50.0
    
    # Base traffic: shape (seq_len, num_nodes)
    traffic_volume = np.outer(base_traffic_factor, node_traffic_scales)
    
    # 2. Setup active violation & encroachment spike at the origin node
    violations = np.zeros((seq_len, num_nodes), dtype=np.float32)
    encroachment = np.zeros((seq_len, num_nodes), dtype=np.float32)
    
    # Spike starts in the middle of history (step 6 onwards)
    violations[6:, origin_idx] = 6.0
    encroachment[6:, origin_idx] = 0.95
    
    # Assemble input tensor: shape (1, seq_len, num_nodes, 3)
    X_input = np.stack([traffic_volume, violations, encroachment], axis=-1)
    X_input = np.expand_dims(X_input, axis=0) # add batch dim
    
    # 3. Model Inference (forward pass)
    try:
        predictions = model(X_input, training=False) # shape (1, 3, num_nodes, 1)
        pred_traffic = np.mean(predictions[0, :, :, 0], axis=0) # average over predicted 3 hours, shape (num_nodes,)
    except Exception as e:
        print("ST-GCN Inference failed:", str(e))
        return calculate_ripple_dijkstra(origin_node_id, edges, nodes)
        
    # Calculate base future traffic volume
    future_hours = [(H + 1 + i) % 24 for i in range(3)]
    future_factors = [
        0.1 + 0.3 * np.sin((h - 4) * np.pi / 12) + 0.4 * np.exp(-((h - 9)**2)/8) + 0.4 * np.exp(-((h - 18)**2)/8)
        for h in future_hours
    ]
    future_base_factor = np.mean(future_factors)
    base_traffic_future = node_traffic_scales * future_base_factor
    
    # Calculate drop in predicted traffic compared to base traffic
    # Higher drops represent severe cascading blockages
    node_drops = np.clip((base_traffic_future - pred_traffic) / (base_traffic_future + 1e-5), 0.0, 1.0)
    
    # Scale drops to delay minutes (max 10 minutes delay)
    node_delays = node_drops * 10.0
    
    # Peripheral Node Bypass: If a node has Degree <= 2, bypass GCN and fallback to baseline (0.0 delay)
    for i in range(num_nodes):
        if node_degrees[i] <= 2:
            node_delays[i] = 0.0
            
    node_delays[origin_idx] = max(node_delays[origin_idx], 8.5) # ensure origin is highest
    
    affected_edges_result = []
    affected_nodes_result = []
    total_delay = 0.0
    affected_streets = set()
    
    # Update edges state
    for e in edges:
        src_id = e['source_node_id']
        tgt_id = e['target_node_id']
        
        src_idx = node_index.get(src_id)
        tgt_idx = node_index.get(tgt_id)
        
        if src_idx is not None and tgt_idx is not None:
            # Edge delay is the average of source and target delays
            edge_delay = float((node_delays[src_idx] + node_delays[tgt_idx]) / 2.0)
        else:
            edge_delay = 0.0
            
        e['delay_minutes'] = edge_delay
        
        if edge_delay > 4.0:
            e['flow_state'] = "anomaly"
            e['color_rgb'] = [255, 0, 255] # Magenta
        elif edge_delay > 1.5:
            e['flow_state'] = "bottleneck"
            e['color_rgb'] = [255, 94, 0] # Hot Orange
        elif edge_delay > 0.5:
            e['flow_state'] = "ripple"
            e['color_rgb'] = [255, 94, 0, 128] # Orange 50%
        else:
            e['flow_state'] = "healthy"
            e['color_rgb'] = [0, 243, 255] # Cyan
            
        if edge_delay > 0.5:
            total_delay += edge_delay
            affected_edges_result.append(e)
            if e['street_name'] and e['street_name'] != "Unknown Segment":
                affected_streets.add(e['street_name'])
                
    # Update nodes state
    for n_id, n_data in nodes.items():
        n_idx = node_index.get(n_id)
        if n_idx is not None:
            n_delay = float(node_delays[n_idx])
            if n_id == origin_node_id:
                n_data['flow_state'] = 'anomaly'
                n_data['delay_minutes'] = n_delay
                affected_nodes_result.append(n_data)
            elif n_delay > 1.5:
                n_data['flow_state'] = 'bottleneck'
                n_data['delay_minutes'] = n_delay
                affected_nodes_result.append(n_data)
            else:
                n_data['flow_state'] = 'healthy'
                n_data['delay_minutes'] = 0.0
        else:
            n_data['flow_state'] = 'healthy'
            n_data['delay_minutes'] = 0.0
            
    # Calculate a scaled average delay of affected edges for the UI readout
    if len(affected_edges_result) > 0:
        ui_delay = (total_delay / len(affected_edges_result)) * 2.5
    else:
        ui_delay = 0.0
        
    return {
        "origin_node_id": origin_node_id,
        "affected_edges": edges,
        "affected_nodes": affected_nodes_result,
        "total_delay_minutes": float(ui_delay),
        "affected_street_names": list(affected_streets),
        "calculation_method": "st-gcn"
    }

def reset_graph(app_state: dict):
    edges = app_state.get("graph_edges", [])
    for e in edges:
        e['flow_state'] = 'healthy'
        e['delay_minutes'] = 0.0
        e['color_rgb'] = [0, 243, 255]
    return edges
