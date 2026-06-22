"""
Nexus Flow - Custom Keras Layers for Spatio-Temporal Traffic Prediction
=======================================================================
Contains:
  1. GraphAttentionLayer (GAT) - Multi-head attention over graph topology
  2. ReshapeTemporalInput - Flattens temporal dim for per-node processing
  3. RestoreTemporalOutput - Restores spatial dimension after temporal processing
  4. TransposeExpandLayer - Final output reshaping
  5. Legacy GraphConvLayer - For loading Phase 1 models
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# =============================================================================
# PHASE 2.2: Graph Attention Layer (GAT)
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="NexusFlow")
class GraphAttentionLayer(layers.Layer):
    """
    Multi-Head Graph Attention Network (GAT) layer.
    
    Uses additive attention: e_ij = LeakyReLU(a_src @ Wh_i + a_tgt @ Wh_j)
    This decomposes the score into source and target terms, avoiding 
    materialization of an N x N x H x 2D tensor (which would be ~672MB 
    for 2290 nodes).
    
    Args:
        num_heads: Number of independent attention heads (default: 4)
        head_dim: Dimensionality of each attention head (default: 16)
        adjacency_matrix: Normalized adjacency matrix (num_nodes x num_nodes)
        dropout_rate: Attention dropout for regularization (default: 0.1)
    """
    
    def __init__(self, num_heads=4, head_dim=16, adjacency_matrix=None, 
                 dropout_rate=0.1, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim
        self.initial_adjacency_matrix = adjacency_matrix
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Shared feature projection W: (features, heads, head_dim)
        self.W = self.add_weight(
            shape=(feature_dim, self.num_heads, self.head_dim),
            initializer='glorot_uniform',
            name='W', trainable=True
        )
        
        # Additive attention vectors (per head)
        # Instead of concat [Wh_i || Wh_j] @ a, we use:
        #   e_ij = (Wh_i @ a_src) + (Wh_j @ a_tgt)
        # This gives identical expressiveness but avoids N x N materialization
        self.attn_src = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer='glorot_uniform',
            name='attn_src', trainable=True
        )
        self.attn_tgt = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer='glorot_uniform',
            name='attn_tgt', trainable=True
        )
        
        # Output projection
        self.W_output = self.add_weight(
            shape=(self.output_dim, self.output_dim),
            initializer='glorot_uniform',
            name='W_output', trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            name='bias', trainable=True
        )
        
        # Adjacency mask (non-trainable)
        if self.initial_adjacency_matrix is not None:
            adj = np.array(self.initial_adjacency_matrix, dtype=np.float32)
            adj_mask = (adj > 0).astype(np.float32)
        else:
            nodes_dim = input_shape[-2]
            adj_mask = np.ones((nodes_dim, nodes_dim), dtype=np.float32)
        
        self.adj_mask = self.add_weight(
            name='adj_mask',
            shape=adj_mask.shape,
            initializer=tf.keras.initializers.Constant(adj_mask),
            trainable=False
        )
        
        super(GraphAttentionLayer, self).build(input_shape)
    
    def call(self, x, training=None):
        # x shape: (batch, num_nodes, features)
        
        # Project features: (batch, N, H, D)
        Wh = tf.einsum('bnf,fhd->bnhd', x, self.W)
        
        # Compute attention scores via additive decomposition:
        # score_src_i = Wh_i @ a_src  →  (batch, N, H)
        # score_tgt_j = Wh_j @ a_tgt  →  (batch, N, H)
        # e_ij = score_src_i + score_tgt_j  →  (batch, N_i, N_j, H) via broadcasting
        score_src = tf.einsum('bnhd,hd->bnh', Wh, self.attn_src)  # (batch, N, H)
        score_tgt = tf.einsum('bnhd,hd->bnh', Wh, self.attn_tgt)  # (batch, N, H)
        
        # Broadcast to pairwise: (batch, N, 1, H) + (batch, 1, N, H) → (batch, N, N, H)
        attention_scores = (tf.expand_dims(score_src, 2) + 
                          tf.expand_dims(score_tgt, 1))
        attention_scores = tf.nn.leaky_relu(attention_scores, alpha=0.2)
        
        # Apply adjacency mask: non-neighbors get -inf
        mask = tf.expand_dims(tf.expand_dims(self.adj_mask, 0), -1)  # (1, N, N, 1)
        attention_scores = tf.where(
            mask > 0,
            attention_scores,
            tf.constant(-1e9, dtype=tf.float32)
        )
        
        # Softmax over neighbors (axis=2 = source nodes j)
        attention_weights = tf.nn.softmax(attention_scores, axis=2)  # (batch, N, N, H)
        
        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)
        
        # Aggregate neighbor features: (batch, N_i, H, D)
        output = tf.einsum('bijh,bjhd->bihd', attention_weights, Wh)
        
        # Concatenate heads: (batch, N, H*D)
        batch_size = tf.shape(output)[0]
        num_nodes = tf.shape(output)[1]
        output = tf.reshape(output, [batch_size, num_nodes, self.output_dim])
        
        # Output projection + bias + activation
        output = tf.matmul(output, self.W_output) + self.bias
        output = tf.nn.elu(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    
    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# =============================================================================
# LEGACY: GraphConvLayer (for loading Phase 1 models)
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class GraphConvLayer(layers.Layer):
    """Static Graph Convolution Layer (Phase 1). Kept for backward compatibility."""
    
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
            name='kernel', trainable=True
        )
        if self.initial_adjacency_matrix is not None:
            adj = np.array(self.initial_adjacency_matrix, dtype=np.float32)
        else:
            nodes_dim = input_shape[-2]
            adj = np.zeros((nodes_dim, nodes_dim), dtype=np.float32)
        self.adj_matrix = self.add_weight(
            name='adj_matrix', shape=adj.shape,
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
        config.update({"units": self.units, "activation": self.activation_name})
        return config


# =============================================================================
# UTILITY LAYERS
# =============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReshapeTemporalInput(layers.Layer):
    """Flattens (batch, time, nodes, features) → (batch*nodes, time*features)."""
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        shape = tf.shape(x)
        return tf.reshape(x, shape=[-1, shape[2] * shape[3]])
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] * input_shape[3])


@tf.keras.utils.register_keras_serializable(package="Custom")
class RestoreTemporalOutput(layers.Layer):
    """Restores (batch*nodes, features) → (batch, nodes, features)."""
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
    """Reshapes (batch, nodes, pred_len) → (batch, pred_len, nodes, 1)."""
    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        return tf.expand_dims(x, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[2], input_shape[1], 1)
