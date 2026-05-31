import json
from pathlib import Path


def test_final_artifact_manifest_groups_and_exists_fields():
    path = Path("outputs/metrics/final_artifact_manifest.json")
    assert path.exists()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    groups = manifest["groups"]
    assert {"models", "real_public_metrics", "calibration", "synthetic", "governance", "dashboard"} <= set(groups)
    for items in groups.values():
        for item in items:
            assert "exists" in item
            assert "path" in item
            assert "purpose" in item
    assert manifest["summary"]["exists_count"] >= 1
