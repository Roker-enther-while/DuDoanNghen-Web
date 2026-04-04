import sys
import os
import glob
import json
import time
import numpy as np
import pandas as pd

# Suppress TensorFlow GPU warnings on Windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.data_preprocessing import prepare_data_v2
from src.utils.data_loaders import UniversalDataLoader
from src.models.tcn_attention_bilstm import build_advanced_model

# ==========================================
# 1. CONFIG (PHASE 3)
# ==========================================
DATA_DIR = "Data/"
WINDOW_SIZE = 60
NUM_FEATURES = 10  # Upgraded from 8
USE_LOG = True
EPOCHS = 100
BATCH_SIZE = 512
MODEL_DIR = "models/checkpoints_advanced"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# ==========================================
# 2. ADVANCED AUTOGRAD TRAINING LOOP
# ==========================================
class AdvancedTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss_metric.update_state(loss)
        return loss

    @tf.function
    def test_step(self, x, y):
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        self.val_loss_metric.update_state(loss)
        return loss

def run_advanced_training_v3():
    # Support multiple formats in training
    extensions = ['*.csv', '*.json', '*.xlsx', '*.xls']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
    
    files = files[:50]  # Limit for speed in this demo
    print(f"[*] Phase 3 Training: Loading {len(files)} multi-format files...")

    loader = UniversalDataLoader()
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for f in files:
        df = loader.load(f)
        if df is not None and len(df) > WINDOW_SIZE:
            try:
                # Use upgraded preprocessing (10 features)
                X_tr, y_tr, X_te, y_te, _ = prepare_data_v2(df, window_size=WINDOW_SIZE, use_log=USE_LOG)
                X_train_list.append(X_tr.astype(np.float32))
                y_train_list.append(y_tr.astype(np.float32))
                X_test_list.append(X_te.astype(np.float32))
                y_test_list.append(y_te.astype(np.float32))
            except: continue

    if not X_train_list:
        print("[!] No valid data found for training.")
        return

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Automatically adapt to 10 features
    model = build_advanced_model((WINDOW_SIZE, X_train.shape[2]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    trainer = AdvancedTrainer(model, optimizer, loss_fn)

    print(f"\n[START] Phase 3 Training (Features: {X_train.shape[2]}) ...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        trainer.train_loss_metric.reset_state()
        trainer.val_loss_metric.reset_state()

        for x, y in train_ds:
            trainer.train_step(x, y)
        for x, y in test_ds:
            trainer.test_step(x, y)

        val_loss = trainer.val_loss_metric.result()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(MODEL_DIR, "best_attention_model_v3.h5"))

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {trainer.train_loss_metric.result():.6f} - val: {val_loss:.6f} - {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_advanced_training_v3()
