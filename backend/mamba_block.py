"""
Nexus Flow - Selective State Space Model (Mamba Block) for TensorFlow
=====================================================================
A TensorFlow-native implementation of the Selective Scan mechanism from
the Mamba architecture (Gu & Dao, 2023). This replaces Dense temporal
bottlenecks with a proper sequence model that has:
  - O(N) linear time complexity (vs O(N^2) for attention)
  - Input-dependent state transitions (selective mechanism)
  - Long-range memory without vanishing gradients
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


@tf.keras.utils.register_keras_serializable(package="NexusFlow")
class SelectiveSSMBlock(layers.Layer):
    """
    Selective State Space Model block (Mamba-style).
    
    Processes temporal sequences by maintaining a hidden state that evolves
    according to learned transition matrices. The "selective" mechanism makes
    the transition matrices input-dependent, allowing the model to dynamically
    decide what to remember and what to forget at each timestep.
    
    State-space equations:
        h(t) = A_bar * h(t-1) + B_bar * x(t)    (state update)
        y(t) = C * h(t) + D * x(t)                (output)
    
    Where A_bar, B_bar are discretized from continuous A, B using:
        delta = softplus(Linear(x))    (input-dependent step size)
        A_bar = exp(delta * A)
        B_bar = delta * B
    
    Args:
        state_dim: Dimensionality of the hidden state (default: 32)
        output_dim: Output feature dimension (default: 32)
        dt_min: Minimum step size for discretization (default: 0.001)
        dt_max: Maximum step size for discretization (default: 0.1)
    """
    
    def __init__(self, state_dim=32, output_dim=32, dt_min=0.001, dt_max=0.1, **kwargs):
        super(SelectiveSSMBlock, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
    
    def build(self, input_shape):
        # input_shape: (batch, seq_len, input_dim)
        input_dim = input_shape[-1]
        
        # Input projection to inner dimension
        self.input_proj = layers.Dense(self.output_dim, name='input_proj')
        
        # State transition matrix A: initialized as negative diagonal (stable)
        # Using HiPPO-inspired initialization for long-range memory
        A_init = -np.exp(np.linspace(np.log(self.dt_min), np.log(self.dt_max), 
                                      self.state_dim)).astype(np.float32)
        self.A_log = self.add_weight(
            shape=(self.output_dim, self.state_dim),
            initializer=tf.keras.initializers.Constant(
                np.broadcast_to(np.log(-A_init), (self.output_dim, self.state_dim))
            ),
            name='A_log', trainable=True
        )
        
        # Selective mechanism: input-dependent delta, B, C
        # Delta (step size) projection
        self.delta_proj = layers.Dense(self.output_dim, name='delta_proj')
        
        # B projection (input → state)
        self.B_proj = layers.Dense(self.state_dim, name='B_proj')
        
        # C projection (state → output)  
        self.C_proj = layers.Dense(self.state_dim, name='C_proj')
        
        # D (skip connection, learnable scalar per dimension)
        self.D = self.add_weight(
            shape=(self.output_dim,),
            initializer='ones',
            name='D', trainable=True
        )
        
        # Output projection
        self.output_proj = layers.Dense(self.output_dim, name='output_proj')
        
        # Layer norm for stability
        self.layer_norm = layers.LayerNormalization(name='ssm_layer_norm')
        
        super(SelectiveSSMBlock, self).build(input_shape)
    
    def selective_scan(self, x, delta, A, B, C, D):
        """
        Performs the selective scan operation using tf.scan for efficiency.
        
        Args:
            x: Input tensor (batch, seq_len, output_dim)
            delta: Step sizes (batch, seq_len, output_dim)
            A: Continuous state matrix (output_dim, state_dim) — negative values
            B: Input-to-state projection (batch, seq_len, state_dim)
            C: State-to-output projection (batch, seq_len, state_dim)
            D: Skip connection (output_dim,)
        
        Returns:
            y: Output tensor (batch, seq_len, output_dim)
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # delta: (batch, seq, D), A: (D, N) → delta_A: (batch, seq, D, N)
        delta_expanded = tf.expand_dims(delta, -1)  # (batch, seq, D, 1)
        A_expanded = tf.expand_dims(tf.expand_dims(A, 0), 0)  # (1, 1, D, N)
        
        A_bar = tf.exp(delta_expanded * A_expanded)  # (batch, seq, D, N)
        
        # B_bar = delta * B
        # delta: (batch, seq, D), B: (batch, seq, N)
        # We need: (batch, seq, D, N)
        B_expanded = tf.expand_dims(B, 2)  # (batch, seq, 1, N)
        B_bar = delta_expanded * B_expanded  # (batch, seq, D, N)
        
        # x contribution: x * B_bar → (batch, seq, D, N)
        x_expanded = tf.expand_dims(x, -1)  # (batch, seq, D, 1)
        x_B = x_expanded * B_bar  # (batch, seq, D, N)
        
        # Transpose for tf.scan: (seq, batch, D, N)
        A_bar_t = tf.transpose(A_bar, [1, 0, 2, 3])
        x_B_t = tf.transpose(x_B, [1, 0, 2, 3])
        C_t = tf.transpose(C, [1, 0, 2])  # (seq, batch, N)
        
        # Initial hidden state: zeros
        h0 = tf.zeros((batch_size, self.output_dim, self.state_dim))
        
        # Scan function: h(t) = A_bar(t) * h(t-1) + x(t) * B_bar(t)
        def scan_step(h_prev, inputs):
            a_bar_t, x_b_t = inputs
            h_new = a_bar_t * h_prev + x_b_t
            return h_new
        
        # Run sequential scan
        all_h = tf.scan(
            scan_step,
            (A_bar_t, x_B_t),
            initializer=h0
        )  # (seq, batch, D, N)
        
        # Output: y(t) = C(t) @ h(t) + D * x(t)
        # all_h: (seq, batch, D, N), C_t: (seq, batch, N)
        C_t_expanded = tf.expand_dims(C_t, 2)  # (seq, batch, 1, N)
        y_ssm = tf.reduce_sum(all_h * C_t_expanded, axis=-1)  # (seq, batch, D)
        y_ssm = tf.transpose(y_ssm, [1, 0, 2])  # (batch, seq, D)
        
        # Skip connection
        y = y_ssm + tf.expand_dims(D, 0) * x
        
        return y
    
    def call(self, x, training=None):
        # x: (batch, seq_len, input_dim)
        residual = x
        
        # Project input
        x_proj = self.input_proj(x)  # (batch, seq, output_dim)
        
        # Compute selective parameters from input
        delta = tf.nn.softplus(self.delta_proj(x))  # (batch, seq, output_dim)
        B = self.B_proj(x)  # (batch, seq, state_dim)
        C = self.C_proj(x)  # (batch, seq, state_dim)
        
        # Continuous A (always negative for stability)
        A = -tf.exp(self.A_log)  # (output_dim, state_dim)
        
        # Run selective scan
        y = self.selective_scan(x_proj, delta, A, B, C, self.D)
        
        # Output projection + normalization
        y = self.output_proj(y)
        y = self.layer_norm(y)
        
        # Take the last timestep as the summary representation
        # (batch, seq, output_dim) → (batch, output_dim)
        y = y[:, -1, :]
        
        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(SelectiveSSMBlock, self).get_config()
        config.update({
            "state_dim": self.state_dim,
            "output_dim": self.output_dim,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
        })
        return config
