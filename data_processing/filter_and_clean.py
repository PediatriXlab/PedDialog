#!/usr/bin/env python3
"""
Filter and clean raw pediatric consultation dialogues.

Implements the eligibility criteria and data cleaning described in
paper Methods: Eligibility Criteria + Data Cleaning and Content Filtering.

Eligibility:
  (1) >= 2 turns (>= 1 caregiver-physician exchange)
  (2) > 50 Chinese characters total
  (3) Contains identifiable clinical content

Cleaning (message-level, within included dialogues):
  - Remove administrative/navigational queries
  - Remove duplicate messages
  - Remove platform-generated automated messages
  - Remove trailing courtesy exchanges

Usage:
    python filter_and_clean.py --input raw.jsonl --output cleaned.jsonl
"""

import argparse
import json
import re
from pathlib import Path

# Administrative keyword patterns (Chinese)
ADMIN_KEYWORDS = [
    "预约", "挂号", "门诊时间", "医院地址", "怎么走", "导航",
    "上班时间", "下班", "排班", "值班", "停车",
]
ADMIN_PATTERN = re.compile("|".join(ADMIN_KEYWORDS))

# Platform-generated message patterns
PLATFORM_PATTERNS = [
    re.compile(r"系统提示|系统消息|温馨提示|请关注|满意度评价|评价医生"),
    re.compile(r"您的排队|当前排队|等待人数|已为您分配|咨询即将结束"),
    re.compile(r"如需.*请.*拨打|客服电话|投诉|举报"),
]

# Trailing courtesy exchange patterns (no clinical content)
COURTESY_PATTERNS = [
    re.compile(r"谢谢|多谢|感谢|辛苦"),              # gratitude
    re.compile(r"不客气|不用谢|应该的|不谢|没事"),    # responses to thanks
    re.compile(r"再见|拜拜"),                         # farewell
    re.compile(r"祝早日|祝您|保重|早日康复"),         # well-wishes
]
ACK_ONLY = re.compile(r"^[嗯好的行对是啊哦]+[!！。.,，]?$")


def count_chinese_chars(text: str) -> int:
    return sum(1 for c in text if "一" <= c <= "鿿")


def is_administrative(content: str) -> bool:
    if ADMIN_PATTERN.search(content) and len(content) < 80:
        return True
    return False


def is_platform_message(content: str) -> bool:
    for pat in PLATFORM_PATTERNS:
        if pat.search(content):
            return True
    return False


def is_near_duplicate(prev: str, curr: str) -> bool:
    if not prev:
        return False
    if prev == curr:
        return True
    # Simple edit-distance approximation
    shorter, longer = (prev, curr) if len(prev) <= len(curr) else (curr, prev)
    if len(shorter) < 5:
        return False
    return shorter in longer and len(shorter) / len(longer) > 0.8


def is_courtesy_only(content: str) -> bool:
    """Return True if *content* is purely a politeness phrase with no clinical value."""
    content = content.strip()
    if len(content) > 30:
        return False
    for pat in COURTESY_PATTERNS:
        if pat.search(content):
            return True
    if len(content) <= 6 and ACK_ONLY.match(content):
        return True
    return False


def trim_trailing_courtesies(turns: list) -> list:
    """Remove consecutive courtesy-only turns from the end of a dialogue.

    Stops at the first turn that has substantive content (i.e. is not a
    pure politeness phrase), preserving all clinical content regardless of
    incidental keywords.
    """
    i = len(turns) - 1
    while i >= 0:
        content = (turns[i].get("content") or turns[i].get("content_zh", "")).strip()
        if is_courtesy_only(content):
            i -= 1
        else:
            break
    return turns[: i + 1]


def check_eligibility(dialogue: dict) -> tuple[bool, str]:
    """Check the three eligibility criteria."""
    turns = dialogue.get("dialog") or dialogue.get("turns", [])
    total_turns = len(turns)

    # Criterion 1: >= 2 turns
    if total_turns < 2:
        return False, f"too few turns ({total_turns})"

    # Criterion 2: > 50 Chinese characters
    total_chars = sum(count_chinese_chars(t.get("content") or t.get("content_zh", "")) for t in turns)
    if total_chars <= 50:
        return False, f"too short ({total_chars} chars)"

    # Criterion 3: Clinical content check (heuristic)
    has_patient_symptom = any(
        t.get("role") == "patient" and len(t.get("content") or t.get("content_zh", "")) > 15
        for t in turns
    )
    has_physician_response = any(
        t.get("role") in {"doctor", "physician"} and len(t.get("content") or t.get("content_zh", "")) > 10
        for t in turns
    )
    if not (has_patient_symptom and has_physician_response):
        return False, "no substantive clinical content"

    return True, ""


def clean_turns(turns: list) -> list:
    """Apply message-level cleaning: remove admin, platform, and duplicate messages."""
    cleaned = []
    prev_content = ""
    for turn in turns:
        content = (turn.get("content") or turn.get("content_zh", "")).strip()
        if not content:
            continue
        if is_platform_message(content):
            continue
        if is_administrative(content):
            continue
        if is_near_duplicate(prev_content, content):
            continue
        cleaned.append(turn)
        prev_content = content
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} dialogues")

    eligible, excluded = 0, 0
    results = []
    exclude_reasons = {}

    for d in data:
        ok, reason = check_eligibility(d)
        if not ok:
            excluded += 1
            exclude_reasons[reason] = exclude_reasons.get(reason, 0) + 1
            continue
        turn_key = "dialog" if "dialog" in d else "turns"
        d[turn_key] = clean_turns(d[turn_key])
        d[turn_key] = trim_trailing_courtesies(d[turn_key])
        eligible += 1
        results.append(d)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Eligible: {eligible}, Excluded: {excluded}")
    if exclude_reasons:
        print("Exclusion reasons:", exclude_reasons)


if __name__ == "__main__":
    main()
