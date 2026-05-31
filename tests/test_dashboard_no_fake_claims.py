from pathlib import Path


def test_dashboard_payload_has_no_fake_claims():
    payload = Path("outputs/web/final_state_b_dashboard_payload.json").read_text(encoding="utf-8")
    assert "0.92" not in payload
    assert "0.89" not in payload
    assert "NASA target is proxy congestion score" in payload
    assert "Zanbil raw is missing" in payload
    assert "Synthetic stress benchmark is controlled simulation" in payload


def test_no_dashboard_html_with_fake_demo_numbers():
    html_files = list(Path(".").rglob("*.html")) + list(Path(".").rglob("*.htm"))
    for path in html_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "TCN-Attention-BiLSTM | Model Test Result Demo" not in text
        assert "R² 0.92" not in text
        assert "F1 0.89" not in text


def test_research_defense_dashboard_has_no_fake_metrics():
    dashboard = Path("outputs/web/research_defense_dashboard.html")
    if not dashboard.exists():
        return
    text = dashboard.read_text(encoding="utf-8")
    # Fake numbers that must NOT appear as results
    assert "0.078" not in text, "Fake RMSE 0.078 found"
    assert "0.92" not in text, "Fake R² 0.92 found"
    assert "0.89" not in text, "Fake F1 0.89 found"
    assert "threshold 0.70" not in text, "Fake threshold 0.70 found"
    assert "best epoch 48" not in text.lower() or "48" not in text, "Fake best epoch 48 found"
    assert "model.keras" not in text, "Fake model.keras found"
    assert "12,480" not in text, "Fake 12,480 samples found"
    assert "giải thích tốt biến thiên" not in text, "Fake claim found"


def test_research_defense_dashboard_has_real_metrics():
    dashboard = Path("outputs/web/research_defense_dashboard.html")
    if not dashboard.exists():
        return
    text = dashboard.read_text(encoding="utf-8")
    # Real numbers that MUST appear
    assert "0.042792" in text, "Real MAE missing"
    assert "0.056399" in text, "Real RMSE missing"
    assert "0.331430" in text, "Real R² missing"
    assert "0.014599" in text, "Real F1 missing"
    assert "0.183838" in text, "Real threshold missing"
    assert "30" in text, "Real best epoch missing"
    assert "best_model.pt" in text, "Real checkpoint missing"


def test_research_defense_dashboard_has_required_sections():
    dashboard = Path("outputs/web/research_defense_dashboard.html")
    if not dashboard.exists():
        return
    text = dashboard.read_text(encoding="utf-8").lower()
    required = [
        "data pipeline",
        "target construction",
        "model training",
        "real public proxy",
        "threshold calibration",
        "synthetic stress",
        "recommendation",
        "minh bạch",
        "next work",
    ]
    for section in required:
        assert section in text, f"Missing section: {section}"


def test_research_defense_dashboard_has_warnings():
    dashboard = Path("outputs/web/research_defense_dashboard.html")
    if not dashboard.exists():
        return
    text = dashboard.read_text(encoding="utf-8").lower()
    assert "proxy" in text, "Missing proxy warning"
    assert "synthetic" in text or "controlled benchmark" in text, "Missing synthetic warning"
    assert "cross-source" in text or "zanbil" in text, "Missing cross-source warning"
