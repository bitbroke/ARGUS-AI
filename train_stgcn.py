import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc

# Configure GPU memory growth to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[SUCCESS] GPU DETECTED: Found {len(gpus)} GPU(s). Training will proceed on the GPU.")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU detected! TensorFlow will fall back to CPU. Please check your CUDA/cuDNN setup.")

@tf.keras.utils.register_keras_serializable(package="Custom")
class AsymmetricFocalRegressionLoss(tf.keras.losses.Loss):
    def __init__(self, tau=0.85, gamma=2.0, name="asymmetric_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tau = tau
        self.gamma = gamma

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.clip_by_value(tf.abs(error), 0.0, 100.0) # Prevent exploding gradients
        weight = tf.where(error > 0, self.tau, 1.0 - self.tau)
        focal_loss = weight * tf.pow(abs_error, self.gamma)
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tau": self.tau,
            "gamma": self.gamma
        })
        return config

# Define custom GraphConvLayer first so Keras can serialize/deserialize it
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
        
        # If adjacency matrix was provided in __init__, use it, otherwise initialize with zeros
        if self.initial_adjacency_matrix is not None:
            adj = np.array(self.initial_adjacency_matrix, dtype=np.float32)
        else:
            # Fallback size based on nodes dimension
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
        # x shape: (batch_size, nodes, features)
        # Step 1: X * W (Feature transformation)
        x_transformed = tf.matmul(x, self.kernel)
        
        # Step 2: A * (X*W) (Spatial Message Passing)
        node_features = tf.einsum('vw,bwc->bvc', self.adj_matrix, x_transformed)
        
        return self.activation(node_features)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, nodes, features) -> output_shape: (batch_size, nodes, units)
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        config = super(GraphConvLayer, self).get_config()
        config.update({
            "units": self.units,
            "activation": self.activation_name,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class GraphAttentionLayer(layers.Layer):
    def __init__(self, units, adjacency_matrix=None, activation='relu', **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.initial_adjacency_matrix = adjacency_matrix

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(shape=(feature_dim, self.units), initializer='glorot_uniform', name='W_kernel', trainable=True)
        self.a_1 = self.add_weight(shape=(self.units, 1), initializer='glorot_uniform', name='a_1', trainable=True)
        self.a_2 = self.add_weight(shape=(self.units, 1), initializer='glorot_uniform', name='a_2', trainable=True)
        
        if self.initial_adjacency_matrix is not None:
            adj = np.array(self.initial_adjacency_matrix, dtype=np.float32)
            # Add self-loops to prevent disconnected nodes from becoming NaN in softmax
            adj = np.clip(adj + np.eye(adj.shape[0], dtype=np.float32), 0.0, 1.0)
        else:
            nodes_dim = input_shape[-2]
            adj = np.zeros((nodes_dim, nodes_dim), dtype=np.float32)
            adj = np.clip(adj + np.eye(adj.shape[0], dtype=np.float32), 0.0, 1.0)
            
        self.adj_matrix = self.add_weight(name='adj_matrix', shape=adj.shape, initializer=tf.keras.initializers.Constant(adj), trainable=False)
        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, x):
        Wh = tf.matmul(x, self.W)
        f_1 = tf.matmul(Wh, self.a_1)
        f_2 = tf.matmul(Wh, self.a_2)
        e = f_1 + tf.transpose(f_2, perm=[0, 2, 1])
        e = tf.nn.leaky_relu(e)
        zero_vec = -9e15 * tf.ones_like(e)
        adj_expanded = tf.expand_dims(self.adj_matrix, axis=0)
        attention = tf.where(adj_expanded > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=-1)
        node_features = tf.matmul(attention, Wh)
        return self.activation(node_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update({"units": self.units, "activation": self.activation_name})
        return config

# Helper custom layer to reshape (batch, seq_len, nodes, features) to (batch * nodes, seq_len * features)
@tf.keras.utils.register_keras_serializable(package="Custom")
class ReshapeTemporalInput(layers.Layer):
    def call(self, x):
        # Transpose to: (batch, nodes, seq_len, features)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # Reshape to: (batch * nodes, seq_len * features)
        shape = tf.shape(x)
        return tf.reshape(x, shape=[-1, shape[2] * shape[3]])
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] * input_shape[3])

# Helper custom layer to restore shape to (batch, nodes, features)
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

# Helper custom layer to transpose and expand to (batch, pred_len, nodes, 1)
@tf.keras.utils.register_keras_serializable(package="Custom")
class TransposeExpandLayer(layers.Layer):
    def call(self, x):
        # x shape: (batch, nodes, pred_len)
        # Transpose to: (batch, pred_len, nodes)
        x = tf.transpose(x, perm=[0, 2, 1])
        # Expand dim to: (batch, pred_len, nodes, 1)
        return tf.expand_dims(x, axis=-1)
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[2], input_shape[1], 1)

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReshapeTemporalInputSequence(layers.Layer):
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        shape = tf.shape(x)
        return tf.reshape(x, shape=[-1, shape[2], shape[3]])
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], input_shape[3])

@tf.keras.utils.register_keras_serializable(package="Custom")
class SelectiveStateSpaceLayer(layers.Layer):
    def __init__(self, state_dim, output_dim, **kwargs):
        super(SelectiveStateSpaceLayer, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        import numpy as np
        # Initialize A as a negative identity matrix for stability (prevent exponential growth)
        self.A = self.add_weight(
            shape=(self.state_dim, self.state_dim), 
            initializer=tf.keras.initializers.Constant(-np.eye(self.state_dim, dtype=np.float32)), 
            trainable=True, name='A'
        )
        self.proj_delta = layers.Dense(self.state_dim, use_bias=False)
        self.proj_B = layers.Dense(self.state_dim, use_bias=False)
        self.proj_C = layers.Dense(self.state_dim, use_bias=False)
        self.out_dense = layers.Dense(self.output_dim)
        super(SelectiveStateSpaceLayer, self).build(input_shape)

    def call(self, x):
        x_list = tf.unstack(x, axis=1)
        B_total = tf.shape(x)[0]
        h_curr = tf.zeros((B_total, self.state_dim))
        
        for x_t in x_list:
            delta_t = tf.math.sigmoid(self.proj_delta(x_t))
            B_t = self.proj_B(x_t)
            Ah = tf.matmul(h_curr, self.A, transpose_b=True)
            A_bar_h = h_curr + delta_t * Ah
            B_bar_x = delta_t * B_t
            h_curr = A_bar_h + B_bar_x
            
        h_final = h_curr
        
        y = self.out_dense(h_final)
        return y
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.state_dim, "output_dim": self.output_dim})
        return config

def main():
    # 1. Load Adjacency Matrix and Graph
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend/data'))
    adj_path = os.path.join(data_dir, 'adjacency_matrix.npy')
    graph_path = os.path.join(data_dir, 'graph.json')
    node_idx_path = os.path.join(data_dir, 'node_index.json')
    
    if not os.path.exists(adj_path) or not os.path.exists(graph_path):
        print("Graph data files not found. Please run backend/scripts/extract_graph.py first.")
        return
        
    print("Loading graph data...")
    A_tf = np.load(adj_path)
    num_nodes = A_tf.shape[0]
    
    with open(node_idx_path, 'r') as f:
        node_mapping = json.load(f) # node_id -> matrix_index
        # Reverse mapping: index -> node_id
        index_to_node = {int(v): k for k, v in node_mapping.items()}
        
    # Re-extract OSMnx graph using bounding box or cache
    center_point = (12.925557, 77.618665) 
    print("Re-creating OSMnx Graph for spatial mapping...")
    G = ox.graph_from_point(center_point, dist=1500, network_type='drive')
    
    # 2. Load and Preprocess Violation Data
    csv_path = r"C:\Users\M S I\Downloads\cancer_dataset\jan to may police violation_anonymized791b166.csv"
    print(f"Loading violation data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing coordinates or datetime
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime']).copy()
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime']).copy()
    
    print("Mapping coordinates to graph nodes (Vectorized)...")
    # ox.nearest_nodes takes (G, X, Y) where X is longitude, Y is latitude
    mapped_nodes = ox.nearest_nodes(G, df['longitude'].values, df['latitude'].values)
    df['node_id'] = [str(n) for n in mapped_nodes]
    
    min_time = df['created_datetime'].min()
    max_time = df['created_datetime'].max()
    
    print(f"Data range: {min_time} to {max_time}")
    
    # Build hour index
    df['hour_index'] = ((df['created_datetime'] - min_time.floor('h')) / pd.Timedelta(hours=1)).astype(int)
    
    total_hours = int(df['hour_index'].max()) + 1
    print(f"Total hours in dataset timeframe: {total_hours}")
    
    # Create violation grid: Shape (hours, nodes)
    print("Aggregating violations by hour and node...")
    violation_matrix = np.zeros((total_hours, num_nodes), dtype=np.float32)
    
    grouped = df.groupby(['hour_index', 'node_id']).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        h_idx = int(row['hour_index'])
        n_id = str(row['node_id'])
        n_idx = node_mapping.get(n_id)
        if n_idx is not None and 0 <= h_idx < total_hours:
            violation_matrix[h_idx, n_idx] = float(row['count'])
            
    # 3. Simulate Traffic Volume and Encroachment Severity
    print("Generating simulated traffic volume and encroachment metrics...")
    
    # Base traffic has diurnal pattern
    hours_range = pd.date_range(start=min_time.floor('h'), periods=total_hours, freq='h')
    hour_of_day = hours_range.hour
    
    # Diurnal factor: Peak at 9 AM and 6 PM (18:00)
    base_traffic_factor = [
        0.1 + 0.3 * np.sin((h - 4) * np.pi / 12) + 0.4 * np.exp(-((h - 9)**2)/8) + 0.4 * np.exp(-((h - 18)**2)/8)
        for h in hour_of_day
    ]
    base_traffic_factor = np.array(base_traffic_factor, dtype=np.float32)
    
    # Node traffic scale based on node connection degree
    node_degrees = [G.degree[int(index_to_node[i])] if int(index_to_node[i]) in G.nodes else 2 for i in range(num_nodes)]
    node_traffic_scales = np.array(node_degrees, dtype=np.float32) * 50.0 # multiplier
    
    traffic_volume = np.outer(base_traffic_factor, node_traffic_scales)
    
    # Add noise to traffic volume
    traffic_noise = np.random.normal(0.0, 5.0, size=traffic_volume.shape).astype(np.float32)
    traffic_volume = np.clip(traffic_volume + traffic_noise, 5.0, None)
    
    # Encroachment severity is triggered by local violations with some background noise
    encroachment_severity = np.clip(violation_matrix * 0.3 + np.random.normal(0.05, 0.02, size=violation_matrix.shape), 0.0, 1.0).astype(np.float32)
    
    # 4. Formulate Spatio-Temporal Data Windows
    seq_len = 12
    pred_len = 3
    num_features = 3
    
    print("Formulating 4D Spatio-Temporal sliding windows...")
    X_all = np.stack([traffic_volume, violation_matrix, encroachment_severity], axis=-1)
    
    X_samples = []
    Y_samples = []
    
    # Use a stride of 2 to reduce memory footprint and sample overlap
    for t in range(0, total_hours - seq_len - pred_len, 2):
        x_seq = X_all[t : t + seq_len, :, :]
        y_seq = traffic_volume[t + seq_len : t + seq_len + pred_len, :, np.newaxis]
        X_samples.append(x_seq)
        Y_samples.append(y_seq)
        if len(X_samples) >= 1000: # Limit to 1000 samples for efficient training
            break
            
    X_train = np.array(X_samples, dtype=np.float32)
    Y_train = np.array(Y_samples, dtype=np.float32)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    
    # Free up memory
    del df
    del grouped
    del X_all
    del X_samples
    del Y_samples
    gc.collect()

    # 5. Build ST-GCN Model
    def build_nexus_flow_model(num_nodes, seq_len, num_features, pred_len, adjacency_matrix):
        input_seq = layers.Input(shape=(seq_len, num_nodes, num_features), name="Spatio_Temporal_Input")
        
        # Spatial propagation (GAT) applied across every time step
        gat_layer = GraphAttentionLayer(units=16, adjacency_matrix=adjacency_matrix, activation='relu')
        spatial_features = layers.TimeDistributed(gat_layer)(input_seq)
        
        # Reshape to keep sequence dimension for Mamba: (batch * nodes, seq_len, 16)
        reshaped_spatial = ReshapeTemporalInputSequence()(spatial_features)
        
        # Temporal propagation (Mamba / Selective State Space)
        ssm_layer = SelectiveStateSpaceLayer(state_dim=64, output_dim=32)
        temporal_out = ssm_layer(reshaped_spatial)
        
        # Restore shape
        restored_temporal = RestoreTemporalOutput(num_nodes=num_nodes)(temporal_out)
        
        # Dense decoder
        dense_out = layers.Dense(pred_len, activation='linear')(restored_temporal)
        
        # Transpose and expand shape to match target
        final_output = TransposeExpandLayer()(dense_out)
        
        model = Model(inputs=input_seq, outputs=final_output, name="Nexus_Flow_ST_GCN_Mamba")
        return model
        
    print("Building model...")
    model = build_nexus_flow_model(num_nodes, seq_len, num_features, pred_len, A_tf)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=AsymmetricFocalRegressionLoss(tau=0.85, gamma=2.0),
        metrics=['mse', 'mae']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    
    print("Initiating training sequence...")
    # Train the model
    model.fit(
        X_train, Y_train,
        batch_size=8,
        epochs=15, # Quick training for hackathon
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    model_save_path = os.path.join(data_dir, 'nexus_flow_model.keras')
    weights_save_path = os.path.join(data_dir, 'nexus_flow_model_weights.weights.h5')
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print(f"Saving pure weights to {weights_save_path}...")
    model.save_weights(weights_save_path)
    print("Model and weights saved successfully!")

if __name__ == '__main__':
    main()
