import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, precision_recall_curve, auc, confusion_matrix, f1_score

# Import custom layers from the training script
import sys
sys.path.append(os.path.dirname(__file__))
from train_stgcn import GraphAttentionLayer, SelectiveStateSpaceLayer, ReshapeTemporalInputSequence, TransposeExpandLayer, RestoreTemporalOutput, AsymmetricFocalRegressionLoss

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend/data'))
    adj_path = os.path.join(data_dir, 'adjacency_matrix.npy')
    graph_path = os.path.join(data_dir, 'graph.json')
    node_idx_path = os.path.join(data_dir, 'node_index.json')
    model_path = os.path.join(data_dir, 'nexus_flow_model.keras')
    
    print("Loading graph data...")
    A_tf = np.load(adj_path)
    num_nodes = A_tf.shape[0]
    
    with open(node_idx_path, 'r') as f:
        node_mapping = json.load(f)
        index_to_node = {int(v): k for k, v in node_mapping.items()}
        
    center_point = (12.925557, 77.618665)
    print("Re-creating OSMnx Graph for node degrees...")
    G = ox.graph_from_point(center_point, dist=1500, network_type='drive')
    
    node_degrees = [G.degree[int(index_to_node[i])] if int(index_to_node[i]) in G.nodes else 2 for i in range(num_nodes)]
    node_degrees = np.array(node_degrees)
    
    print("Loading data...")
    csv_path = r"C:\Users\M S I\Downloads\cancer_dataset\jan to may police violation_anonymized791b166.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime']).copy()
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime']).copy()
    
    mapped_nodes = ox.nearest_nodes(G, df['longitude'].values, df['latitude'].values)
    df['node_id'] = [str(n) for n in mapped_nodes]
    
    min_time = df['created_datetime'].min()
    total_hours = int(((df['created_datetime'].max() - min_time.floor('h')) / pd.Timedelta(hours=1))) + 1
    
    violation_matrix = np.zeros((total_hours, num_nodes), dtype=np.float32)
    df['hour_index'] = ((df['created_datetime'] - min_time.floor('h')) / pd.Timedelta(hours=1)).astype(int)
    grouped = df.groupby(['hour_index', 'node_id']).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        h_idx, n_id = int(row['hour_index']), str(row['node_id'])
        n_idx = node_mapping.get(n_id)
        if n_idx is not None and 0 <= h_idx < total_hours:
            violation_matrix[h_idx, n_idx] = float(row['count'])
            
    hours_range = pd.date_range(start=min_time.floor('h'), periods=total_hours, freq='h')
    hour_of_day = hours_range.hour
    base_traffic = np.array([0.1 + 0.3 * np.sin((h - 4) * np.pi / 12) + 0.4 * np.exp(-((h - 9)**2)/8) + 0.4 * np.exp(-((h - 18)**2)/8) for h in hour_of_day], dtype=np.float32)
    node_scales = np.array(node_degrees, dtype=np.float32) * 50.0
    traffic_volume = np.outer(base_traffic, node_scales)
    
    np.random.seed(42)
    traffic_volume = np.clip(traffic_volume + np.random.normal(0.0, 5.0, size=traffic_volume.shape), 5.0, None).astype(np.float32)
    encroachment = np.clip(violation_matrix * 0.3 + np.random.normal(0.05, 0.02, size=violation_matrix.shape), 0.0, 1.0).astype(np.float32)
    
    # Force a massive traffic drop on high violations to simulate anomalies
    # This ensures anomalies exist in the test set to evaluate PR-AUC properly!
    anomaly_mask = (violation_matrix > np.percentile(violation_matrix[violation_matrix > 0], 90))
    traffic_volume[anomaly_mask] = traffic_volume[anomaly_mask] * 0.2
    
    seq_len, pred_len, num_features = 12, 3, 3
    X_all = np.stack([traffic_volume, violation_matrix, encroachment], axis=-1)
    
    X_samples, Y_samples = [], []
    for t in range(0, total_hours - seq_len - pred_len, 2):
        X_samples.append(X_all[t : t + seq_len, :, :])
        Y_samples.append(traffic_volume[t + seq_len : t + seq_len + pred_len, :, np.newaxis])
        if len(X_samples) >= 1000: break
            
    X_train_full = np.array(X_samples, dtype=np.float32)
    Y_train_full = np.array(Y_samples, dtype=np.float32)
    
    split_idx = int(len(X_train_full) * 0.8)
    X_train, Y_train = X_train_full[:split_idx], Y_train_full[:split_idx]
    X_test, Y_test = X_train_full[split_idx:], Y_train_full[split_idx:]
    
    print("Rebuilding Keras model architecture...")
    weights_path = os.path.join(data_dir, 'nexus_flow_model_weights.weights.h5')
    
    try:
        input_layer = tf.keras.layers.Input(shape=(seq_len, num_nodes, num_features), name='Spatio_Temporal_Input')
        x = tf.keras.layers.TimeDistributed(GraphAttentionLayer(units=16, adjacency_matrix=A_tf))(input_layer)
        x = ReshapeTemporalInputSequence()(x)
        x = SelectiveStateSpaceLayer(state_dim=64, output_dim=32)(x)
        x = RestoreTemporalOutput(num_nodes=num_nodes)(x)
        x = tf.keras.layers.Dense(pred_len)(x)
        output_layer = TransposeExpandLayer()(x)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='Nexus_Flow_ST_GCN_Mamba')
        
        # Force build of all internal custom layers by running a dummy batch
        dummy_input = tf.zeros((1, seq_len, num_nodes, num_features))
        model(dummy_input)
        
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    print("Evaluating Model...")
    y_train_pred = model.predict(X_train, batch_size=8)
    y_test_pred = model.predict(X_test, batch_size=8)
    
    train_mae = mean_absolute_error(Y_train.flatten(), y_train_pred.flatten())
    test_mae = mean_absolute_error(Y_test.flatten(), y_test_pred.flatten())
    
    print("\n--- Training Dynamics ---")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Val MAE:   {test_mae:.4f}")
    
    print("\n--- Topological & Spatial Error ---")
    # Y_test shape: (batch, pred_len, nodes, 1)
    for deg in [1, 2, 3, 4, 5]:
        deg_mask = (node_degrees == deg)
        if np.any(deg_mask):
            deg_y_true = Y_test[:, :, deg_mask, :]
            deg_y_pred = y_test_pred[:, :, deg_mask, :]
            deg_mae = mean_absolute_error(deg_y_true.flatten(), deg_y_pred.flatten())
            print(f"Degree {deg} Nodes MAE: {deg_mae:.4f}")

    print("\n--- Anomaly Detection (PR-AUC & Confusion Matrix) ---")
    # Identify anomalies: where traffic suddenly drops significantly below expected scale
    # For a simple binary classification on regression outputs:
    # A true anomaly is defined where actual traffic is < 40% of standard traffic
    # But for a direct comparison, let's look at relative drop
    y_true_flat = Y_test.flatten()
    y_pred_flat = y_test_pred.flatten()
    
    # Ground truth anomaly: Traffic is heavily congested (e.g., volume is extremely low compared to node scale)
    # Let's say anomaly if y_true < 20
    anomaly_threshold = 20.0
    true_anomalies = (y_true_flat < anomaly_threshold).astype(int)
    pred_anomalies = (y_pred_flat < anomaly_threshold).astype(int)
    
    if np.sum(true_anomalies) > 0:
        precision, recall, _ = precision_recall_curve(true_anomalies, -y_pred_flat)
        pr_auc = auc(recall, precision)
        f1 = f1_score(true_anomalies, pred_anomalies)
        cm = confusion_matrix(true_anomalies, pred_anomalies)
        
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
    else:
        print("No ground truth anomalies found in test set to evaluate PR-AUC.")

if __name__ == '__main__':
    main()
