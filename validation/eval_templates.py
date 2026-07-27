#!/usr/bin/env python3
"""
Generate evaluation templates for annotation quality (B2) and translation quality (B3).

Implements the template generation for expert validation described in paper
Technical Validation: Reasoning Annotation Quality and the translation verification step.

Usage:
    python eval_templates.py --input release.jsonl --outdir templates/
"""

import argparse
import csv
import json
import random
from pathlib import Path


def generate_cot_eval_template(dialogues: list, n: int = 50, seed: int = 42) -> list[dict]:
    """Generate B2: CoT annotation quality evaluation template."""
    random.seed(seed)
    sampled = random.sample(dialogues, min(n, len(dialogues)))

    rows = []
    for d in sampled:
        for t in d["turns"]:
            if t["role"] == "physician" and t.get("cot_reasoning_zh"):
                rows.append({
                    "dialogue_id": d["dialogue_id"],
                    "turn_id": t["turn_id"],
                    "department": d["department"],
                    "content_zh": t["content_zh"],
                    "cot_reasoning_zh": t["cot_reasoning_zh"],
                    "coherence": "",        # 1-5 Likert
                    "professionalism": "",  # 1-5 Likert
                    "completeness": "",     # 1-5 Likert
                    "clinical_accuracy": "",  # 1-5 Likert
                    "issues": "",
                    "reviewer": "",
                })
    return rows


def generate_translation_eval_template(dialogues: list, n: int = 50, seed: int = 42) -> list[dict]:
    """Generate B3: Translation quality evaluation template."""
    random.seed(seed + 1)
    sampled = random.sample(dialogues, min(n, len(dialogues)))

    rows = []
    for d in sampled:
        for t in d["turns"]:
            if t["role"] == "physician" and t.get("content_en"):
                row = {
                    "dialogue_id": d["dialogue_id"],
                    "turn_id": t["turn_id"],
                    "content_zh": t["content_zh"],
                    "content_en": t["content_en"],
                    "medical_accuracy": "",  # 1-5 Likert
                    "fluency": "",           # 1-5 Likert
                    "terminology": "",       # 1-5 Likert
                    "cot_reasoning_zh": t.get("cot_reasoning_zh", ""),
                    "cot_reasoning_en": t.get("cot_reasoning_en", ""),
                    "cot_accuracy": "",      # 1-5 Likert
                    "issues": "",
                    "reviewer": "",
                }
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Bilingual JSONL")
    parser.add_argument("--outdir", default="templates")
    parser.add_argument("--n", type=int, default=50, help="Number of dialogues to sample")
    args = parser.parse_args()

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} dialogues")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # B2: CoT quality
    cot_rows = generate_cot_eval_template(data, args.n)
    cot_path = outdir / "b2_cot_quality_eval.csv"
    with open(cot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cot_rows[0].keys())
        writer.writeheader()
        writer.writerows(cot_rows)
    print(f"  B2 CoT template: {cot_path} ({len(cot_rows)} physician turns from {args.n} dialogues)")

    # B3: Translation quality
    tr_rows = generate_translation_eval_template(data, args.n)
    tr_path = outdir / "b3_translation_quality_eval.csv"
    with open(tr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tr_rows[0].keys())
        writer.writeheader()
        writer.writerows(tr_rows)
    print(f"  B3 Translation template: {tr_path} ({len(tr_rows)} physician turns from {args.n} dialogues)")


if __name__ == "__main__":
    main()
