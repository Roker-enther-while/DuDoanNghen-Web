"""GPU/VRAM configuration helpers for TensorFlow training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class GpuMemoryPlan:
    enabled: bool
    gpu_found: bool
    logical_memory_limit_mb: int | None
    data_buffer_budget_mb: int
    model_budget_mb: int
    mixed_precision: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def configure_tensorflow_gpu(config: dict | None = None) -> GpuMemoryPlan:
    """Configure TensorFlow GPU memory before model creation.

    Ghi chú tiếng Việt:
    - TensorFlow không cho chia cứng VRAM thành "vùng data" và "vùng model" theo kiểu thủ công.
      Dữ liệu train nên nằm ở RAM CPU/NumPy, chỉ mini-batch hiện tại được copy lên GPU.
    - Tham khảo TensorFlow GPU guide: dùng memory growth hoặc LogicalDeviceConfiguration(memory_limit)
      để tránh TensorFlow chiếm toàn bộ VRAM ngay từ đầu.
    - Tham khảo Micikevicius et al., "Mixed Precision Training" (ICLR 2018): dùng FP16 cho tensor
      tính toán chính và giữ phần nhạy cảm/loss ở FP32 để giảm VRAM và tăng throughput.
    - Tham khảo Chen et al., "Training Deep Nets with Sublinear Memory Cost" (arXiv 2016):
      activation checkpointing/recomputation có thể giảm bộ nhớ activation, nhưng chưa bật mặc định
      trong quick mode để giữ code ổn định và dễ kiểm thử.
    """
    config = config or {}
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    notes: list[str] = []
    gpu_memory_limit_mb = config.get("gpu_memory_limit_mb")
    data_buffer_budget_mb = int(config.get("gpu_data_buffer_mb", 1024))
    model_budget_mb = int(config.get("gpu_model_budget_mb", 4096))
    mixed_precision = bool(config.get("mixed_precision", False))
    try:
        import tensorflow as tf
    except Exception:
        return GpuMemoryPlan(False, False, None, data_buffer_budget_mb, model_budget_mb, mixed_precision, ["TensorFlow unavailable"])

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("float32")
        return GpuMemoryPlan(True, False, None, data_buffer_budget_mb, model_budget_mb, mixed_precision, ["No GPU found; using CPU"])

    try:
        if gpu_memory_limit_mb:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(gpu_memory_limit_mb))],
            )
            notes.append(f"Logical GPU memory limit set to {int(gpu_memory_limit_mb)} MB")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            notes.append("TensorFlow GPU memory growth enabled")
    except RuntimeError as exc:
        notes.append(f"GPU memory config skipped because TensorFlow was already initialized: {exc}")

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        notes.append("Mixed precision policy enabled: mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    return GpuMemoryPlan(True, True, int(gpu_memory_limit_mb) if gpu_memory_limit_mb else None, data_buffer_budget_mb, model_budget_mb, mixed_precision, notes)
