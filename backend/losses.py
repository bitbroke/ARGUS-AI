"""
Nexus Flow - Custom Loss Functions for Spatio-Temporal Traffic Prediction
=========================================================================
Asymmetric Focal Loss for regression: heavily penalizes under-prediction
of traffic drops (missed congestion events) while being lenient on
over-predictions (false alarms).
"""

import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package="NexusFlow")
class AsymmetricFocalLoss(tf.keras.losses.Loss):
    """
    Asymmetric Focal Regression Loss for congestion-sensitive traffic prediction.
    
    The key insight: Missing a traffic jam (under-predicting a drop) is FAR more
    dangerous than a false alarm (over-predicting a drop). This loss function
    encodes that asymmetry directly into the gradient signal.
    
    Math:
        error = y_true - y_pred
        weight = alpha   if y_true < y_pred  (model missed a drop → heavy penalty)
        weight = 1.0     if y_true >= y_pred  (model over-predicted drop → light penalty)
        focal_weight = |error / scale|^gamma  (focus on hard examples)
        loss = mean(weight * focal_weight * |error|)
    
    Args:
        alpha: Asymmetric penalty multiplier for under-prediction (default: 3.0)
        gamma: Focal exponent to down-weight easy samples (default: 2.0)
        scale: Normalization scale for the focal term (default: 50.0, ~mean traffic volume)
    """
    
    def __init__(self, alpha=3.0, gamma=2.0, scale=50.0, **kwargs):
        super(AsymmetricFocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
    
    def call(self, y_true, y_pred):
        # Compute signed error: positive when y_true > y_pred (model under-predicted)
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        # Asymmetric weighting:
        # When y_true < y_pred → model thought traffic was higher than reality
        #   (model MISSED a traffic drop / congestion event) → penalize heavily
        # When y_true >= y_pred → model predicted lower traffic than reality
        #   (false alarm, predicted congestion that didn't happen) → normal penalty
        under_prediction_mask = tf.cast(y_true < y_pred, tf.float32)
        asymmetric_weight = 1.0 + (self.alpha - 1.0) * under_prediction_mask
        
        # Focal weighting: focus on hard examples (large errors relative to scale)
        # Normalized error magnitude in [0, ~1] range
        normalized_error = tf.minimum(abs_error / self.scale, 1.0)
        focal_weight = tf.pow(normalized_error, self.gamma)
        
        # Combined loss: asymmetric × focal × MAE
        # The +0.1 ensures minimum gradient flow for easy samples
        loss = asymmetric_weight * (focal_weight + 0.1) * abs_error
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(AsymmetricFocalLoss, self).get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "scale": self.scale,
        })
        return config
