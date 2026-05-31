"""PyTorch CUDA training loop for deep sequence models."""

from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np

from src.training.torch_models import build_torch_model


def train_torch_model(model_name: str, X_train, y_train, X_val, y_val, X_test, config: dict, output_dir: str | Path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if config.get("require_gpu", False) and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA GPU is required by config but is not available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config.get("seed", 42)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("seed", 42)))
        torch.backends.cudnn.benchmark = True

    # Dữ liệu vẫn được giữ ở RAM CPU; DataLoader chỉ đưa từng mini-batch lên GPU.
    # Cách này tránh chiếm VRAM bằng toàn bộ dataset và phù hợp chiến lược chia ngân sách:
    # RAM/CPU cho data buffer, VRAM cho model + activation + batch hiện tại.
    train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 128)), shuffle=False, pin_memory=(device.type == "cuda"))
    val_x = torch.as_tensor(X_val, dtype=torch.float32)
    val_y = torch.as_tensor(y_val, dtype=torch.float32)
    test_x = torch.as_tensor(X_test, dtype=torch.float32)

    artifact_name = config.get("artifact_name", model_name)
    model_dir = Path(output_dir) / "models" / artifact_name
    checkpoint_dir = model_dir / "checkpoints"
    metrics_dir = Path(output_dir) / "metrics" / artifact_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = build_torch_model(model_name, tuple(X_train.shape[1:]), config).to(device)
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.001)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    criterion = torch.nn.MSELoss()
    use_amp = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"loss": [], "val_loss": [], "val_rmse": [], "epoch_time_seconds": [], "learning_rate": []}
    best_val = float("inf")
    best_state = None
    start_epoch = 0
    patience = int(config.get("patience", 3))
    stale = 0
    if config.get("resume", False) and not config.get("restart", False):
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scaler_state_dict"):
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            history = checkpoint.get("history", history)
            best_val = float(checkpoint.get("best_val_rmse", float("inf"))) ** 2
            start_epoch = int(checkpoint.get("epoch", 0))
    started = time.perf_counter()

    for epoch_index in range(start_epoch, int(config.get("epochs", 5))):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            if config.get("gradient_clip_norm") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm")))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x.to(device)).detach().cpu()
            val_loss = float(criterion(val_pred, val_y).detach().cpu()) if len(val_y) else 0.0
        val_rmse = float(np.sqrt(val_loss))
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["epoch_time_seconds"].append(float(time.perf_counter() - epoch_started))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"model_name": model_name, "state_dict": best_state, "config": config, "input_shape": tuple(X_train.shape[1:])},
                model_dir / "best_model.pt",
            )
            stale = 0
        else:
            stale += 1
        if config.get("checkpoint_every_epoch", False):
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "best_val_rmse": float(np.sqrt(best_val)) if np.isfinite(best_val) else float("inf"),
                    "config": config,
                    "history": history,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch_index + 1:03d}.pt",
            )
        if config.get("early_stopping", True) and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.perf_counter() - started
    best_model_path = model_dir / "best_model.pt"
    last_model_path = model_dir / "last_model.pt"
    model_path = best_model_path if best_model_path.exists() else model_dir / "model.pt"
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "config": config, "input_shape": tuple(X_train.shape[1:])}, last_model_path)
    if not best_model_path.exists():
        torch.save({"model_name": model_name, "state_dict": model.state_dict(), "config": config, "input_shape": tuple(X_train.shape[1:])}, model_path)
    (metrics_dir / "history.json").write_text(json.dumps({"model": model_name, "history": history}, indent=2), encoding="utf-8")
    with (metrics_dir / "training_log.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss,val_rmse,learning_rate,epoch_time_seconds\n")
        for i, loss in enumerate(history["loss"]):
            handle.write(
                f"{i + 1},{loss},{history['val_loss'][i]},{history['val_rmse'][i]},"
                f"{history['learning_rate'][i]},{history['epoch_time_seconds'][i]}\n"
            )

    infer_started = time.perf_counter()
    model.eval()
    preds = []
    test_loader = DataLoader(torch.as_tensor(test_x, dtype=torch.float32), batch_size=int(config.get("batch_size", 128)), shuffle=False, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for xb in test_loader:
            preds.append(model(xb.to(device, non_blocking=True)).detach().cpu().numpy())
    inference_time = time.perf_counter() - infer_started
    y_pred = np.concatenate(preds).astype(np.float32) if preds else np.array([], dtype=np.float32)
    gpu_info = {
        "backend": "torch",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_budget_mb": int(config.get("gpu_memory_limit_mb", 5120)),
        "data_buffer_budget_mb": int(config.get("gpu_data_buffer_mb", 1024)),
        "model_budget_mb": int(config.get("gpu_model_budget_mb", 4096)),
        "mixed_precision": use_amp,
        "checkpoint_dir": str(checkpoint_dir),
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
    }
    return str(model_path), history, y_pred, float(train_time), float(inference_time), gpu_info
