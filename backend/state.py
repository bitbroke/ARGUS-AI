# In-memory application state
app_state = {
    "frame_counter": 0,
    "current_anomalies": [], # List of AnomalyNode dicts
    "graph_nodes": {},       # dict of node_id -> GraphNode
    "graph_edges": [],       # List of GraphEdge dicts
    "dispatch_log": []
}
