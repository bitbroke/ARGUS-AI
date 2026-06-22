import tensorflow as tf
import numpy as np
import json
import gc
import os

# Set paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
model_path = os.path.join(data_dir, 'nexus_flow_model.keras')
graph_path = os.path.join(data_dir, 'graph.json')
export_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ui_traffic_simulation.json'))

print("Loading Graph Data...")
with open(graph_path, 'r') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']
edges = graph_data['edges']
num_nodes = len(nodes)

# Calculate node degrees from edges
node_degrees = {node['node_id']: 0 for node in nodes}
for edge in edges:
    if edge['source_node_id'] in node_degrees:
        node_degrees[edge['source_node_id']] += 1
    if edge['target_node_id'] in node_degrees:
        node_degrees[edge['target_node_id']] += 1

from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Custom")
class GraphConvLayer(layers.Layer):
    def __init__(self, units, adjacency_matrix=None, activation='relu', **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.initial_adjacency_matrix = adjacency_matrix

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )
        if self.initial_adjacency_matrix is not None:
            adj = np.array(self.initial_adjacency_matrix, dtype=np.float32)
        else:
            nodes_dim = input_shape[-2]
            adj = np.zeros((nodes_dim, nodes_dim), dtype=np.float32)
            
        self.adj_matrix = self.add_weight(
            name='adj_matrix',
            shape=adj.shape,
            initializer=tf.keras.initializers.Constant(adj),
            trainable=False
        )
        super(GraphConvLayer, self).build(input_shape)

    def call(self, x):
        x_transformed = tf.matmul(x, self.kernel)
        node_features = tf.einsum('vw,bwc->bvc', self.adj_matrix, x_transformed)
        return self.activation(node_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        config = super(GraphConvLayer, self).get_config()
        config.update({
            "units": self.units,
            "activation": self.activation_name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReshapeTemporalInput(layers.Layer):
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        shape = tf.shape(x)
        return tf.reshape(x, shape=[-1, shape[2] * shape[3]])
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] * input_shape[3])

@tf.keras.utils.register_keras_serializable(package="Custom")
class RestoreTemporalOutput(layers.Layer):
    def __init__(self, num_nodes, **kwargs):
        super(RestoreTemporalOutput, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        
    def call(self, x):
        shape = tf.shape(x)
        batch_size = shape[0] // self.num_nodes
        features_dim = shape[1]
        return tf.reshape(x, shape=[batch_size, self.num_nodes, features_dim])
        
    def compute_output_shape(self, input_shape):
        return (None, self.num_nodes, input_shape[-1])
        
    def get_config(self):
        config = super(RestoreTemporalOutput, self).get_config()
        config.update({"num_nodes": self.num_nodes})
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class TransposeExpandLayer(layers.Layer):
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        return tf.expand_dims(x, axis=-1)
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[2], input_shape[1], 1)

print("Loading Nexus Flow model...")
nexus_model = tf.keras.models.load_model(model_path)

print("Synthesizing memory-safe Demo Sequence (Morning Rush Hour)...")
# Shape: (1 sequence, 12 time steps, 2290 nodes, 3 features)
# Features: [traffic_volume, violation_matrix, encroachment_severity]
seq_len = 12
demo_X = np.zeros((1, seq_len, num_nodes, 3), dtype=np.float32)

for i, node in enumerate(nodes):
    degree = max(2, node_degrees[node['node_id']])
    base_vol = degree * 50.0  # Scale by degree
    
    for t in range(seq_len):
        # Morning rush hour multiplier (ramping up)
        time_mult = 0.5 + (t / seq_len) * 0.5 
        vol = base_vol * time_mult + np.random.normal(0, 5.0)
        
        # Synthetic violations and encroachments
        violations = np.random.poisson(1.0) if np.random.rand() > 0.8 else 0
        encroachment = min(1.0, violations * 0.3 + np.random.normal(0.05, 0.02))
        
        demo_X[0, t, i, 0] = max(5.0, vol)
        demo_X[0, t, i, 1] = float(violations)
        demo_X[0, t, i, 2] = max(0.0, encroachment)

print("Generating 3-hour future simulation...")
demo_predictions = nexus_model.predict(demo_X) # Shape: (1, 3, num_nodes, 1)

print("Applying Dynamic Thresholds and formatting payload...")
ui_payload = {
    "simulation_hours": 3,
    "nodes": []
}

for i, node in enumerate(nodes):
    node_id = node['node_id']
    degree = node_degrees[node_id]
    
    # Apply Fix #1 Logic (Dynamic Thresholding)
    if degree <= 2:
        threshold_multiplier = 0.75 # 25% drop
    elif degree == 3:
        threshold_multiplier = 0.85 # 15% drop
    else:
        threshold_multiplier = 0.95 # 5% drop (Hyper-sensitive)
        
    baseline = demo_X[0, -1, i, 0] # Traffic volume at Hour 0 (last step of input sequence)
    
    node_data = {
        "node_id": node_id,
        "lat": float(node['lat']),
        "lon": float(node['lon']),
        "degree": degree,
        "predictions": []
    }
    
    for future_hour in range(3):
        pred_vol = float(demo_predictions[0, future_hour, i, 0])
        is_congested = bool(pred_vol < (baseline * threshold_multiplier))
        
        node_data["predictions"].append({
            "hour": future_hour + 1,
            "predicted_volume": max(0, round(pred_vol, 2)),
            "congested": is_congested
        })
        
    ui_payload["nodes"].append(node_data)

with open(export_path, 'w') as f:
    json.dump(ui_payload, f)

print(f"✅ Success! UI Payload saved to {export_path}")
print("Hand this file over to the Next.js frontend!")
