from pathlib import Path


def test_final_state_b_research_summary_transparency():
    text = Path("outputs/reports/final_state_b_research_summary.md").read_text(encoding="utf-8")
    required = [
        "proxy congestion score",
        "synthetic_not_real_world",
        "data/raw/zanbil/access.log",
        "ready_for_cross_source_claim=false",
        "Training a multi-source model in this state would be misleading",
    ]
    for phrase in required:
        assert phrase in text
