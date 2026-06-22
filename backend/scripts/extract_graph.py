import osmnx as ox
import networkx as nx
import json
import os
import numpy as np
import scipy.sparse as sp

# Create data directory if it doesn't exist
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
os.makedirs(data_dir, exist_ok=True)

# 1. Create the Graph (Centered on your hotspot)
center_point = (12.925557, 77.618665) 
print(f"Extracting graph around Koramangala: {center_point}")
G = ox.graph_from_point(center_point, dist=1500, network_type='drive')

# Simplify to just intersections (nodes)
nodes_list = list(G.nodes())
num_nodes = len(nodes_list)
node_mapping = {node: i for i, node in enumerate(nodes_list)}

# 2. Build the Adjacency Matrix
A = nx.adjacency_matrix(G, nodelist=nodes_list).astype(np.float32)

# 3. Graph Math: Normalize the Adjacency Matrix (Crucial for GCN stability)
# Formula: A_hat = D^{-1/2} * (A + I) * D^{-1/2}
A_tilde = A + sp.eye(num_nodes) # Add self-loops so nodes consider their own traffic
D = np.array(A_tilde.sum(axis=1)).flatten()
D_inv_sqrt = np.power(D, -0.5)
D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
D_mat_inv_sqrt = sp.diags(D_inv_sqrt)

A_normalized = A_tilde.dot(D_mat_inv_sqrt).transpose().dot(D_mat_inv_sqrt).todense()
A_normalized = np.array(A_normalized, dtype=np.float32)

# Save adjacency matrix
np.save(os.path.join(data_dir, 'adjacency_matrix.npy'), A_normalized)
print(f"Adjacency matrix shape: {A_normalized.shape} saved to data/adjacency_matrix.npy")

# Save node_index map to match adjacency matrix rows/cols
with open(os.path.join(data_dir, 'node_index.json'), 'w') as f:
    json.dump({str(node): i for node, i in node_mapping.items()}, f)

# Extract bounding box from nodes
lats = [data['y'] for node, data in G.nodes(data=True)]
lons = [data['x'] for node, data in G.nodes(data=True)]
min_lat, max_lat = min(lats), max(lats)
min_lon, max_lon = min(lons), max(lons)

# Format nodes list
nodes_data = []
for node, data in G.nodes(data=True):
    nodes_data.append({
        "node_id": str(node),
        "lat": float(data['y']),
        "lon": float(data['x']),
        "street_name": "Koramangala Intersection",
        "flow_state": "healthy",
        "delay_minutes": 0.0
    })

# Format edges list
edges_data = []
for u, v, k, data in G.edges(keys=True, data=True):
    edge_id = f"edge_{u}_{v}"
    # Try to find street name
    name = data.get('name', 'Unknown Segment')
    if isinstance(name, list):
        name = name[0]
    
    # Coordinates of u and v to draw path
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    
    # Path is a list of [lon, lat] coordinates for Deck.gl
    if 'geometry' in data:
        path = [[float(coords[0]), float(coords[1])] for coords in data['geometry'].coords]
    else:
        path = [[float(u_data['x']), float(u_data['y'])], [float(v_data['x']), float(v_data['y'])]]
        
    edges_data.append({
        "edge_id": edge_id,
        "source_node_id": str(u),
        "target_node_id": str(v),
        "street_name": str(name),
        "length_m": float(data.get('length', 0)),
        "flow_state": "healthy",
        "delay_minutes": 0.0,
        "color_rgb": [0, 243, 255],
        "path": path
    })

graph_data = {
    "nodes": nodes_data,
    "edges": edges_data,
    "corridor_name": "Koramangala Corridor",
    "bbox": {
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
        "min_lon": float(min_lon),
        "max_lon": float(max_lon)
    }
}

with open(os.path.join(data_dir, 'graph.json'), 'w') as f:
    json.dump(graph_data, f, indent=2)

print(f"Graph extracted successfully! Saved {len(nodes_data)} nodes and {len(edges_data)} edges to data/graph.json")
