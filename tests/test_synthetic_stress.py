import numpy as np
import pandas as pd

from src.data.synthetic_stress import SCENARIOS, generate_synthetic_stress_windows


def _base_npz(path):
    rng = np.random.default_rng(1)
    X_train = rng.random((12, 10, 6), dtype=np.float32).astype(np.float16)
    y_train = rng.random(12, dtype=np.float32).astype(np.float16)
    X_val = rng.random((8, 10, 6), dtype=np.float32).astype(np.float16)
    y_val = rng.random(8, dtype=np.float32).astype(np.float16)
    X_test = rng.random((20, 10, 6), dtype=np.float32).astype(np.float16)
    y_test = rng.random(20, dtype=np.float32).astype(np.float16)
    np.savez_compressed(
        path,
        X_train=X_train,
        y_train=y_train,
        ts_train=np.arange(12),
        X_val=X_val,
        y_val=y_val,
        ts_val=np.arange(8),
        X_test=X_test,
        y_test=y_test,
        ts_test=np.arange(20),
        feature_columns=np.array(["request_count", "bytes_sum", "unique_hosts", "error_rate", "status_5xx", "congestion_score_proxy"], dtype=object),
        target_column=np.array("target_next_congestion_score", dtype=object),
    )


def test_generate_synthetic_stress_has_six_scenarios_and_labels(tmp_path):
    base = tmp_path / "base.npz"
    out = tmp_path / "stress.npz"
    _base_npz(base)
    meta = generate_synthetic_stress_windows(base, out, samples_per_scenario=20, seed=123)
    with np.load(out, allow_pickle=True) as data:
        assert set(data["scenario_name"].tolist()) == set(SCENARIOS)
        assert set(data["phase"].tolist()) == {"background", "pre_incident", "incident", "recovery"}
        assert data["is_synthetic"].all()
        assert data["true_alert_label"].sum() > 0
        assert (~data["true_alert_label"]).sum() > 0
        assert data["X_test"].dtype == np.float16
    assert 0.20 <= meta["positive_ratio"] <= 0.40
    labels = pd.read_csv(meta["labels_path"])
    assert set(labels["phase"]) == {"background", "pre_incident", "incident", "recovery"}
    assert set(labels["scenario_name"]) == set(SCENARIOS)


def test_synthetic_generation_reproducible(tmp_path):
    base = tmp_path / "base.npz"
    out1 = tmp_path / "stress1.npz"
    out2 = tmp_path / "stress2.npz"
    _base_npz(base)
    generate_synthetic_stress_windows(base, out1, samples_per_scenario=8, seed=7)
    generate_synthetic_stress_windows(base, out2, samples_per_scenario=8, seed=7)
    with np.load(out1, allow_pickle=True) as a, np.load(out2, allow_pickle=True) as b:
        assert np.array_equal(a["X_test"], b["X_test"])
        assert np.array_equal(a["scenario_name"], b["scenario_name"])
