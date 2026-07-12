"""Transparent synthetic stress benchmark generation from public baseline windows."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCENARIOS = [
    "flash_crowd",
    "burst_traffic",
    "error_surge",
    "slow_ramp",
    "periodic_spike",
    "mixed_incident",
]


def _feature_index(feature_columns: np.ndarray, name: str) -> int | None:
    names = [str(item) for item in feature_columns.tolist()]
    return names.index(name) if name in names else None


def _boost(X: np.ndarray, feature_columns: np.ndarray, feature_name: str, start: int, end: int, amount: float) -> None:
    idx = _feature_index(feature_columns, feature_name)
    if idx is not None:
        X[start:end, idx] = np.clip(X[start:end, idx] + amount, 0.0, 1.0)


def apply_scenario(base_window: np.ndarray, feature_columns: np.ndarray, scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one documented stress scenario to a normalized window."""
    X = base_window.astype(np.float32).copy()
    lookback = X.shape[0]
    start = int(rng.integers(max(1, lookback // 4), max(2, lookback // 2)))
    duration = int(rng.integers(5, max(6, min(30, lookback - start + 1))))
    end = min(lookback, start + duration)
    peak = min(end - 1, start + max(1, duration // 2))
    severity = float(rng.uniform(0.65, 0.98))

    load_features = ["request_count", "bytes_sum", "bytes_mean", "unique_hosts", "unique_clients", "throughput_bytes_per_min"]
    error_features = ["error_rate", "status_5xx", "error_count"]

    if scenario == "flash_crowd":
        amount = float(rng.uniform(0.35, 0.75))
        for feature in load_features:
            _boost(X, feature_columns, feature, start, end, amount)
        _boost(X, feature_columns, "error_rate", min(end, start + 3), end, 0.15)
    elif scenario == "burst_traffic":
        for offset in range(start, end, int(rng.integers(2, 5))):
            burst_end = min(end, offset + int(rng.integers(1, 4)))
            for feature in load_features:
                _boost(X, feature_columns, feature, offset, burst_end, float(rng.uniform(0.3, 0.6)))
    elif scenario == "error_surge":
        for feature in error_features:
            _boost(X, feature_columns, feature, start, end, float(rng.uniform(0.45, 0.85)))
    elif scenario == "slow_ramp":
        ramp = np.linspace(0.05, float(rng.uniform(0.45, 0.75)), max(1, end - start), dtype=np.float32)
        for feature in load_features:
            idx = _feature_index(feature_columns, feature)
            if idx is not None:
                X[start:end, idx] = np.clip(X[start:end, idx] + ramp, 0.0, 1.0)
    elif scenario == "periodic_spike":
        period = int(rng.integers(4, 9))
        for offset in range(start, end, period):
            for feature in load_features:
                _boost(X, feature_columns, feature, offset, min(end, offset + 2), float(rng.uniform(0.25, 0.55)))
    elif scenario == "mixed_incident":
        ramp = np.linspace(0.05, 0.45, max(1, end - start), dtype=np.float32)
        for feature in load_features:
            idx = _feature_index(feature_columns, feature)
            if idx is not None:
                X[start:end, idx] = np.clip(X[start:end, idx] + ramp, 0.0, 1.0)
        for feature in error_features:
            _boost(X, feature_columns, feature, peak, end, float(rng.uniform(0.25, 0.65)))
    else:
        raise ValueError(f"Unknown synthetic scenario: {scenario}")

    # If the proxy score is part of features, align it with the synthetic incident pressure.
    _boost(X, feature_columns, "congestion_score_proxy", start, end, severity * 0.4)
    y = np.float16(np.clip(max(float(base_window[-1].mean()), severity), 0.0, 1.0))
    label = {
        "is_synthetic": True,
        "scenario_name": scenario,
        "incident_start": start,
        "incident_peak": peak,
        "incident_end": end,
        "true_alert_label": True,
        "severity": severity,
    }
    return X.astype(np.float16), label | {"target": float(y)}


def apply_phase_scenario(
    base_window: np.ndarray,
    feature_columns: np.ndarray,
    scenario: str,
    phase: str,
    rng: np.random.Generator,
    recovery_positive: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a background/pre/incident/recovery sample for one scenario."""
    if phase == "background":
        return base_window.astype(np.float16), {
            "target": float(np.clip(base_window[-1].mean(), 0.0, 0.45)),
            "is_synthetic": True,
            "scenario_name": scenario,
            "phase": phase,
            "incident_start": -1,
            "incident_peak": -1,
            "incident_end": -1,
            "true_alert_label": False,
            "severity": 0.0,
        }

    incident_X, label = apply_scenario(base_window, feature_columns, scenario, rng)
    if phase == "incident":
        label["phase"] = phase
        label["true_alert_label"] = True
        return incident_X, label

    base = base_window.astype(np.float32)
    stressed = incident_X.astype(np.float32)
    if phase == "pre_incident":
        # Dau hieu truoc su co: tang nhe nhung mac dinh chua gan alert label.
        blend = 0.25
        severity = float(label["severity"]) * blend
        true_alert = False
    elif phase == "recovery":
        # Pha hoi phuc: tin hieu con du am huong nhung giam dan; nua dau recovery duoc gan positive.
        blend = 0.45 if recovery_positive else 0.20
        severity = float(label["severity"]) * blend
        true_alert = bool(recovery_positive)
    else:
        raise ValueError(f"Unknown synthetic phase: {phase}")

    X = np.clip(base * (1.0 - blend) + stressed * blend, 0.0, 1.0).astype(np.float16)
    label["phase"] = phase
    label["true_alert_label"] = true_alert
    label["severity"] = severity
    label["target"] = float(np.clip(max(float(base_window[-1].mean()), severity), 0.0, 1.0))
    return X, label


def _phase_counts(samples_per_scenario: int, ratios: dict[str, float]) -> dict[str, int]:
    phases = ["background", "pre_incident", "incident", "recovery"]
    if samples_per_scenario < len(phases):
        raise ValueError("samples_per_scenario must be at least 4 to include all phases")
    raw = {phase: int(round(samples_per_scenario * float(ratios.get(f"{phase}_ratio", 0.0)))) for phase in phases}
    for phase in phases:
        raw[phase] = max(1, raw[phase])
    diff = samples_per_scenario - sum(raw.values())
    while diff != 0:
        phase = "background" if diff < 0 and raw["background"] > 1 else "incident"
        raw[phase] += 1 if diff > 0 else -1
        diff = samples_per_scenario - sum(raw.values())
    return raw


def generate_synthetic_stress_windows(
    base_npz_path: str | Path,
    output_path: str | Path,
    samples_per_scenario: int = 200,
    seed: int = 42,
    phase_ratios: dict[str, float] | None = None,
    target_positive_ratio_min: float = 0.20,
    target_positive_ratio_max: float = 0.40,
    labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate stress-test windows from public real baseline windows."""
    rng = np.random.default_rng(seed)
    with np.load(base_npz_path, allow_pickle=True) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
        ts_train = data.get("ts_train", np.arange(len(y_train)))
        X_val = data["X_val"]
        y_val = data["y_val"]
        ts_val = data.get("ts_val", np.arange(len(y_val)))
        base_X_test = data["X_test"]
        feature_columns = data["feature_columns"]
        target_column = data["target_column"]

    phase_ratios = phase_ratios or {
        "background_ratio": 0.50,
        "pre_incident_ratio": 0.15,
        "incident_ratio": 0.25,
        "recovery_ratio": 0.10,
    }
    phase_counts = _phase_counts(samples_per_scenario, phase_ratios)
    config_base = {
        "base_data": str(base_npz_path),
        "seed": seed,
        "samples_per_scenario": samples_per_scenario,
        "scenarios": SCENARIOS,
        "phase_counts": phase_counts,
        "phase_ratios": phase_ratios,
        "target_positive_ratio_min": target_positive_ratio_min,
        "target_positive_ratio_max": target_positive_ratio_max,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_notice": "Generated stress benchmark, not real-world data.",
    }
    generation_config_id = hashlib.sha256(json.dumps(config_base, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    synthetic_X: list[np.ndarray] = []
    synthetic_y: list[float] = []
    labels: list[dict[str, Any]] = []
    sample_index = 0
    for scenario in SCENARIOS:
        for phase, count in phase_counts.items():
            for phase_i in range(count):
                idx = int(rng.integers(0, len(base_X_test)))
                recovery_positive = phase == "recovery" and phase_i < max(1, count // 2)
                X, label = apply_phase_scenario(
                    base_X_test[idx],
                    feature_columns,
                    scenario,
                    phase,
                    rng,
                    recovery_positive=recovery_positive,
                )
                synthetic_X.append(X)
                synthetic_y.append(label.pop("target"))
                label.update(
                    {
                        "sample_index": sample_index,
                        "timestamp_index": sample_index,
                        "source_id": "synthetic_stress_public_baseline",
                        "generation_config_id": generation_config_id,
                    }
                )
                labels.append(label)
                sample_index += 1

    X_test = np.stack(synthetic_X, axis=0).astype(np.float16)
    y_test = np.array(synthetic_y, dtype=np.float16)
    ts_test = np.arange(len(y_test), dtype=np.int64)
    scenario_names = np.array([label["scenario_name"] for label in labels], dtype=object)
    phases = np.array([label["phase"] for label in labels], dtype=object)
    source_ids = np.array([label["source_id"] for label in labels], dtype=object)
    generation_config_ids = np.array([label["generation_config_id"] for label in labels], dtype=object)
    is_synthetic = np.array([True] * len(labels), dtype=bool)
    true_alert_label = np.array([label["true_alert_label"] for label in labels], dtype=bool)
    severity = np.array([label["severity"] for label in labels], dtype=np.float32)
    incident_start = np.array([label["incident_start"] for label in labels], dtype=np.int16)
    incident_peak = np.array([label["incident_peak"] for label in labels], dtype=np.int16)
    incident_end = np.array([label["incident_end"] for label in labels], dtype=np.int16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(config_base)
    config["generation_config_id"] = generation_config_id
    positive_cases = int(true_alert_label.sum())
    negative_cases = int(len(true_alert_label) - positive_cases)
    positive_ratio = float(positive_cases / len(true_alert_label)) if len(true_alert_label) else 0.0
    if not (target_positive_ratio_min <= positive_ratio <= target_positive_ratio_max):
        raise ValueError(
            f"Synthetic positive ratio {positive_ratio:.4f} outside "
            f"[{target_positive_ratio_min}, {target_positive_ratio_max}]"
        )
    np.savez_compressed(
        output_path,
        X_train=X_train.astype(np.float16),
        y_train=y_train.astype(np.float16),
        ts_train=ts_train,
        X_val=X_val.astype(np.float16),
        y_val=y_val.astype(np.float16),
        ts_val=ts_val,
        X_test=X_test,
        y_test=y_test,
        ts_test=ts_test,
        feature_columns=feature_columns,
        target_column=target_column,
        scenario_name=scenario_names,
        phase=phases,
        source_id=source_ids,
        generation_config_id=generation_config_ids,
        is_synthetic=is_synthetic,
        true_alert_label=true_alert_label,
        severity=severity,
        incident_start=incident_start,
        incident_peak=incident_peak,
        incident_end=incident_end,
        generation_config=np.array(json.dumps(config), dtype=object),
    )
    if labels_path is None:
        labels_path = output_path.parent.parent / "labels" / "synthetic_stress_labels.csv"
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels)[
        [
            "sample_index",
            "timestamp_index",
            "is_synthetic",
            "source_id",
            "scenario_name",
            "phase",
            "true_alert_label",
            "severity",
            "incident_start",
            "incident_peak",
            "incident_end",
            "generation_config_id",
        ]
    ].to_csv(labels_path, index=False)
    return {
        "path": str(output_path),
        "labels_path": str(labels_path),
        "base_data": str(base_npz_path),
        "samples_per_scenario": samples_per_scenario,
        "scenario_count": len(SCENARIOS),
        "synthetic_test_samples": int(len(y_test)),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "positive_ratio": positive_ratio,
        "phase_counts": {phase: int(np.sum(phases == phase)) for phase in sorted(set(phases.tolist()))},
        "scenario_counts": {scenario: int(np.sum(scenario_names == scenario)) for scenario in SCENARIOS},
        "scenarios": SCENARIOS,
        "generation_config": config,
    }
