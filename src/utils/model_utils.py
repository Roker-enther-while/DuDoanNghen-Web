import os
import tensorflow as tf
from src.models.attention_layer import FeatureAttention, TemporalAttention

def load_web_tab_model(model_path, compile=False):
    """
    Centralized model loader for WebTAB.
    Handles custom attention layers and ensures forward/backward compatibility.
    """
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        return None
        
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'FeatureAttention': FeatureAttention,
                'TemporalAttention': TemporalAttention,
                'Attention': TemporalAttention # Legacy support
            },
            compile=compile
        )
        return model
    except Exception as e:
        print(f"[!] Error loading model at {model_path}: {e}")
        return None
