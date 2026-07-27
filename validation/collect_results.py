#!/usr/bin/env python3
"""
Collect ablation results into a summary JSON and CSV.

Reads results.json from each condition directory, then prints a summary
matching Table 4 and writes:
  - <output_dir>/ablation_summary.json
  - <output_dir>/ablation_results_table.csv

Usage:
    python collect_results.py --results_dir results/ablation
"""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path


def load_json(path):
    return json.load(open(path)) if Path(path).exists() else {}


def compute_perplexity(loss):
    return math.exp(loss) if (loss and loss > 0) else None


def get_train_loss(cond_dir):
    """Mean training loss from the newest checkpoint's trainer_state.json."""
    cond_dir = Path(cond_dir)
    states = sorted(
        glob.glob(str(cond_dir / "checkpoint-*" / "trainer_state.json")),
        key=os.path.getmtime,
    )
    if not states:
        return None
    log_history = json.load(open(states[-1])).get("log_history", [])
    losses = [e["loss"] for e in log_history
              if "loss" in e and "eval_loss" not in e]
    return round(sum(losses) / len(losses), 4) if losses else None


def main():
    parser = argparse.ArgumentParser(
        description="Collect ablation results (Table 4)"
    )
    parser.add_argument(
        "--results_dir", default="results/ablation",
        help="Directory containing condition subdirectories (default: results/ablation)",
    )
    args = parser.parse_args()

    R = Path(args.results_dir)

    rows = [
        {
            "condition": "base",
            "condition_label": "Base model",
            "train_loss": None,
            "eval_loss": load_json(R / "base" / "results.json").get("eval_loss"),
            "eval_perplexity": compute_perplexity(
                load_json(R / "base" / "results.json").get("eval_loss")
            ),
            "note": "reply tokens only",
        },
        {
            "condition": "dialogue_only",
            "condition_label": "Dialogue only",
            "train_loss": get_train_loss(R / "dialogue_only"),
            "eval_loss": load_json(R / "dialogue_only" / "results.json").get("eval_loss"),
            "eval_perplexity": compute_perplexity(
                load_json(R / "dialogue_only" / "results.json").get("eval_loss")
            ),
            "note": "reply tokens only",
        },
        {
            "condition": "dialogue_reasoning",
            "condition_label": "Dialogue + reasoning annotation",
            "train_loss": None,
            "eval_loss": load_json(
                R / "dialogue_only" / "results_with_reasoning.json"
            ).get("eval_loss"),
            "eval_perplexity": compute_perplexity(
                load_json(
                    R / "dialogue_only" / "results_with_reasoning.json"
                ).get("eval_loss")
            ),
            "note": "dialogue_only model; gold reasoning in context at eval",
        },
    ]

    # ---- JSON ----
    summary = {
        "experiment": "CoT_Ablation",
        "eval_target": "physician reply tokens",
        "results": rows,
    }
    summary_path = R / "ablation_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2, ensure_ascii=False)
    print(f"Summary → {summary_path}\n")

    # ---- table ----
    print(f"{'Condition':<34}{'Train Loss':>12}{'Eval Loss':>12}{'Eval PPL':>10}")
    print("-" * 68)
    for r in rows:
        tl = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "—"
        el = f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "N/A"
        ppl = f"{r['eval_perplexity']:.2f}" if r["eval_perplexity"] is not None else "N/A"
        print(f"{r['condition_label']:<34}{tl:>12}{el:>12}{ppl:>10}")
    print("-" * 68)
    print("Note: all PPLs computed over physician reply tokens only.")

    # ---- CSV ----
    csv_path = R / "ablation_results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
