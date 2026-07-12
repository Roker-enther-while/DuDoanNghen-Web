"""Run a small hyperparameter sweep for TCN-Attention-BiLSTM."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import load_training_config, train_and_evaluate


def _best_val_rmse(history_path: str | None) -> float | None:
    if not history_path:
        return None
    path = Path(history_path)
    if not path.exists():
        return None
    history = json.loads(path.read_text(encoding="utf-8")).get("history", {})
    val_loss = history.get("val_loss") or []
    if not val_loss:
        return None
    return float(math.sqrt(min(float(x) for x in val_loss)))


def write_tuning_report(results: list[dict], output_dir: str | Path) -> tuple[str, str]:
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics" / "tcn_attention_bilstm_tuning.json"
    report_path = output_dir / "reports" / "tcn_attention_bilstm_tuning.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    successful = [r for r in results if r.get("status") == "success"]
    best = min(successful, key=lambda r: r.get("best_val_rmse") or float("inf")) if successful else None
    payload = {"results": results, "best_option": best["option_name"] if best else None}
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# TCN-Attention-BiLSTM Small Tuning",
        "",
        "| option | status | best_val_rmse | test_rmse | test_mae | model_path |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['option_name']} | {r['status']} | {r.get('best_val_rmse') or 0.0:.6f} | "
            f"{r.get('metrics', {}).get('rmse', 0.0):.6f} | {r.get('metrics', {}).get('mae', 0.0):.6f} | {r.get('model_path')} |"
        )
    lines.extend(["", f"- Best by validation RMSE: {payload['best_option']}", "- Diagnostic tuning only, not full hyperparameter search."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(metrics_path), str(report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "training" / "tcn_attention_bilstm_tuning_small.yaml"))
    parser.add_argument("--max-options", type=int, default=None)
    args = parser.parse_args(argv)
    base_config = load_training_config(args.config)
    options = base_config.get("options", [])
    if args.max_options:
        options = options[: args.max_options]
    results = []
    for option in options:
        option_name = option["name"]
        config = dict(base_config)
        config.update(option)
        config.pop("options", None)
        config["output_dir"] = str(Path(base_config.get("output_dir", "outputs")) / "tuning" / "tcn_attention_bilstm" / option_name)
        result = train_and_evaluate("tcn_attention_bilstm", args.data, config)
        result["option_name"] = option_name
        result["best_val_rmse"] = _best_val_rmse(result.get("history_path"))
        results.append(result)
    metrics_path, report_path = write_tuning_report(results, base_config.get("output_dir", "outputs"))
    print(json.dumps({"metrics": metrics_path, "report": report_path, "options_run": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
