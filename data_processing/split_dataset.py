#!/usr/bin/env python3
"""
Split dataset into train/validation/test by department (stratified).

Implements the stratified splitting described in paper Methods: Sampling Strategy.

Usage:
    python split_dataset.py --input cleaned.jsonl --outdir splits/ --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} dialogues")

    # Group by department in either release schema or intermediate pipeline schema.
    by_dept = {}
    for d in data:
        info = d.get("dialog_analysis_info", {})
        dept = d.get("department") or info.get("department_classification", {}).get("department", "Other")
        by_dept.setdefault(dept, []).append(d)

    train, val, test = [], [], []
    for dept, items in by_dept.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * args.train_ratio))
        n_val = max(1, round(n * args.val_ratio))
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = outdir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for d in split_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(split_data)} dialogues")


if __name__ == "__main__":
    main()
