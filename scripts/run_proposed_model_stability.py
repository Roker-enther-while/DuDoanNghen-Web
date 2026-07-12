"""Run repeated-seed stability checks for the proposed TCN-Attention-BiLSTM model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import load_training_config, train_and_evaluate


OPTION_C = {
    "filters": 64,
    "lstm_units": 32,
    "attention_heads": 2,
    "attention_key_dim": 16,
    "dropout": 0.15,
    "learning_rate": 0.0007,
    "dense_units": 64,
    "dilations": [1, 2, 4],
}


def _best_epoch(history_path: str | None) -> int | None:
    if not history_path or not Path(history_path).exists():
        return None
    history = json.loads(Path(history_path).read_text(encoding="utf-8")).get("history", {})
    val_loss = history.get("val_loss") or []
    if not val_loss:
        return None
    return int(np.argmin(val_loss) + 1)


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()) if arr.size else 0.0, "std": float(arr.std(ddof=0)) if arr.size else 0.0}


def write_report(results: list[dict], output_dir: str | Path) -> tuple[str, str]:
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics" / "tcn_attention_bilstm_stability.json"
    report_path = output_dir / "reports" / "tcn_attention_bilstm_stability.md"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    successful = [r for r in results if r.get("status") == "success"]
    payload = {
        "results": results,
        "rmse": _summary([r["metrics"]["rmse"] for r in successful]),
        "f1": _summary([r["alert_metrics"]["f1"] for r in successful]),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# TCN-Attention-BiLSTM Stability",
        "",
        "| seed | status | MAE | RMSE | R2 | F1 | train_time | best_epoch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['seed']} | {r['status']} | {r['metrics']['mae']:.6f} | {r['metrics']['rmse']:.6f} | "
            f"{r['metrics']['r2']:.6f} | {r['alert_metrics']['f1']:.6f} | {r['train_time_seconds']:.3f} | {r.get('best_epoch') or ''} |"
        )
    lines.extend(
        [
            "",
            f"- RMSE mean ± std: {payload['rmse']['mean']:.6f} ± {payload['rmse']['std']:.6f}",
            f"- F1 mean ± std: {payload['f1']['mean']:.6f} ± {payload['f1']['std']:.6f}",
            "- Stability check only; not a full hyperparameter search.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(metrics_path), str(report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "training" / "compare_all_balanced.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args(argv)
    base = load_training_config(args.config)
    seeds = args.seeds[: args.max_seeds] if args.max_seeds else args.seeds
    results = []
    for seed in seeds:
        config = dict(base)
        config.update(OPTION_C)
        config["seed"] = seed
        config["models"] = ["tcn_attention_bilstm"]
        config["output_tag"] = f"stability_seed_{seed}"
        config["output_dir"] = str(Path(base.get("output_dir", "outputs")) / "stability" / f"seed_{seed}")
        result = train_and_evaluate("tcn_attention_bilstm", args.data, config)
        result["seed"] = seed
        result["best_epoch"] = _best_epoch(result.get("history_path"))
        results.append(result)
    metrics_path, report_path = write_report(results, base.get("output_dir", "outputs"))
    print(json.dumps({"metrics": metrics_path, "report": report_path, "seeds": seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
