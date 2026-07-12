import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_pipeline_smoke_offline_passes():
    result = subprocess.run(
        [sys.executable, "scripts/run_data_pipeline.py", "--config", "configs/data/nasa_http_smoke.yaml"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "success" in result.stdout
    manifest_path = Path("outputs/metrics/data_pipeline_manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    windows_path = Path(manifest["output_paths"]["windows"])
    assert windows_path.exists()
    with np.load(windows_path, allow_pickle=True) as data:
        assert data["X_train"].dtype == np.float16
        assert data["y_train"].dtype == np.float16
        assert data["X_train"].shape[1] == 30
        assert data["X_train"].shape[0] > 0
        assert data["X_val"].shape[0] > 0
        assert data["X_test"].shape[0] > 0
