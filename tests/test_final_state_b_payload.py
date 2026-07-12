import json
import math
from pathlib import Path


def _walk_numbers(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_numbers(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_numbers(value)
    elif isinstance(obj, float):
        yield obj


def test_final_state_b_payload_has_real_metrics():
    path = Path("outputs/web/final_state_b_dashboard_payload.json")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_state"] == "STATE_B_FALLBACK_COMPLETE"
    assert payload["governance"]["no_cross_source_claim"] is True
    assert payload["synthetic_stress_result"]["synthetic_not_real_world"] is True
    assert payload["real_public_proxy_result"]["rmse"] == 0.056398649724090116
    assert payload["real_public_proxy_result"]["f1"] == 0.014598540145985401
    assert payload["threshold_calibration"]["calibrated_f1"] == 0.8655956456851901
    assert all(math.isfinite(value) for value in _walk_numbers(payload))
