import json

from scripts.build_source_license_manifest import _update_dashboard_schema
from scripts.generate_synthetic_stress_benchmark import write_synthetic_dashboard_payload


def test_dashboard_payload_gets_governance_fields(tmp_path):
    payload_dir = tmp_path / "web" / "full_120_tcn_attention_bilstm"
    payload_dir.mkdir(parents=True)
    payload_path = payload_dir / "model_dashboard_payload.json"
    payload_path.write_text(
        json.dumps({"threshold_info": {"mode": "quantile"}, "warnings": []}),
        encoding="utf-8",
    )
    manifest = {
        "sources": [
            {
                "source_id": "nasa_http_1995",
                "license_name": "redistributable",
                "citation": "citation",
                "pii_handling": ["hash_ip"],
            }
        ]
    }
    _update_dashboard_schema(manifest, tmp_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert "data_governance" in payload
    assert "evaluation_context" in payload
    assert payload["data_governance"]["sources"] == ["nasa_http_1995"]
    assert payload["evaluation_context"]["target_type"] == "proxy"


def test_synthetic_dashboard_payload_has_evaluation_context(tmp_path):
    manifest = {
        "path": "data/processed/synthetic_stress/windows/windows_fp16.npz",
        "labels_path": "data/processed/synthetic_stress/labels/synthetic_stress_labels.csv",
        "positive_ratio": 0.3,
        "negative_cases": 70,
        "positive_cases": 30,
        "phase_counts": {"background": 50, "pre_incident": 15, "incident": 25, "recovery": 10},
        "scenario_counts": {"flash_crowd": 100},
    }
    payload_path = write_synthetic_dashboard_payload(manifest, tmp_path)
    payload = json.loads(open(payload_path, encoding="utf-8").read())
    context = payload["evaluation_context"]
    assert context["result_type"] == "synthetic_stress_test"
    assert context["target_type"] == "synthetic_label"
    assert context["positive_ratio"] == 0.3
    assert "phase_distribution" in context
    assert "scenario_distribution" in context
    assert "synthetic_not_real_world" in payload["warnings"]
