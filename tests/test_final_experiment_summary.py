import json
from pathlib import Path

from scripts.build_final_experiment_summary import _finite_payload, write_summary


def test_finite_payload_removes_nan():
    payload = _finite_payload({"x": float("nan"), "nested": [float("inf"), 1.0]})
    assert payload["x"] is None
    assert payload["nested"][0] is None


def test_write_summary_creates_json_and_markdown(tmp_path, monkeypatch):
    import scripts.build_final_experiment_summary as summary_module

    monkeypatch.setattr(summary_module, "ROOT", tmp_path)
    summary = {
        "completion_state": "STATE B FALLBACK COMPLETE",
        "state_reason": "no zanbil",
        "source_governance": {"sources": ["nasa_http_1995"]},
        "zanbil_status": {"ready_for_prepare": False},
        "multi_source_status": {"sources": ["nasa_http_1995"], "ready_for_cross_source_claim": False},
        "nasa_only_real_public_proxy_result": {"metrics": {"mae": 0.1, "rmse": 0.2, "r2": 0.3}, "alert_metrics": {"precision": 0.1, "recall": 0.2, "f1": 0.3}},
        "synthetic_stress_result_separate": {"best_synthetic_f1_threshold": {"precision": 0.4, "recall": 0.5, "f1": 0.6}},
        "limitations": ["proxy"],
        "next_action": "place raw",
    }
    paths = write_summary(summary)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["completion_state"] == "STATE B FALLBACK COMPLETE"
