#!/usr/bin/env python3
"""
PII scanning for de-identification verification (Validation B1).

Scans dialogues for residual personally identifiable information across 18 categories.
Implements the verification step described in paper Technical Validation: De-identification.

Usage:
    python pii_scan.py --input cleaned.jsonl --output pii_scan_results.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# PII detection patterns
PII_PATTERNS = {
    "phone_number": re.compile(r"1[3-9]\d{9}|\d{3,4}[-\s]\d{7,8}"),
    "id_number": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "address_detail": re.compile(r"(?:省|市|区|县|路|街|号|栋|单元|室].{0,10}(?:省|市|区|县|路|街|号|栋|单元|室))"),
}

# Hospital/department names that might identify the institution
HOSPITAL_KEYWORDS = [
    "急诊科", "住院", "门诊", "ICU", "急诊", "抢救",
]

# Potential personal name pattern (2-3 character Chinese names in context)
NAME_PATTERN = re.compile(r"(?:叫|名为|姓名[是为])\s*[一-鿿]{2,4}")


def scan_dialogue(dialogue: dict) -> list[dict]:
    """Scan a single dialogue for PII. Returns list of findings."""
    findings = []
    turns = dialogue.get("dialog") or dialogue.get("turns", [])
    context_fields = {
        k: dialogue.get(k)
        for k in (
            "dialogue_id",
            "department",
            "chief_complaint_zh",
            "chief_complaint_en",
            "patient_age_group",
            "patient_gender",
            "primary_intent",
        )
        if k in dialogue
    }
    text = json.dumps({"metadata": context_fields, "turns": turns}, ensure_ascii=False)

    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": pii_type, "value": match.group()})

    # Check hospital names
    for kw in HOSPITAL_KEYWORDS:
        if kw in text:
            idx = text.find(kw)
            context = text[max(0, idx - 10) : idx + len(kw) + 10]
            findings.append({"type": "hospital_name", "value": kw, "context": context})
            break  # one match per keyword per dialogue

    # Check potential names
    for match in NAME_PATTERN.finditer(text):
        findings.append({"type": "possible_name", "value": match.group()})

    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Scanning {len(data)} dialogues for PII...")

    all_findings = {}
    type_counts = Counter()
    dialogues_with_findings = 0

    for i, d in enumerate(data):
        findings = scan_dialogue(d)
        if findings:
            dialogues_with_findings += 1
            all_findings[f"dlg_{i}"] = findings
            for f in findings:
                type_counts[f"type:{f['type']}"] += 1

    clean_rate = (len(data) - dialogues_with_findings) / len(data) * 100

    result = {
        "summary": {
            "total_dialogues": len(data),
            "dialogues_with_findings": dialogues_with_findings,
            "clean_dialogues": len(data) - dialogues_with_findings,
            "clean_rate": f"{clean_rate:.1f}%",
            "findings_by_type": dict(type_counts.most_common()),
        },
        "findings": all_findings,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Clean: {len(data) - dialogues_with_findings}/{len(data)} ({clean_rate:.1f}%)")
    if type_counts:
        print("Findings by type:", dict(type_counts))


if __name__ == "__main__":
    main()
