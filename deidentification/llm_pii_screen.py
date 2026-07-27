#!/usr/bin/env python3
"""
LLM-based automated PII screening for pediatric consultation dialogues.

Implements the de-identification protocol Stage 1 described in paper Methods:
De-identification Protocol — Automated Screening. Uses an LLM (default:
Qwen3-30B-A3B-Instruct-250720, deployed locally) to scan each dialogue and
flag explicit personal identifiers for subsequent manual review.

Usage:
    python deidentification/llm_pii_screen.py \
        --input cleaned.jsonl \
        --output screened.jsonl \
        --workers 5 \
        --report pii_report.json
"""

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.llm_client import LLMClient, parse_json


# ---------------------------------------------------------------------------
# PII categories matching the paper's enumeration
# ---------------------------------------------------------------------------
PII_CATEGORIES = [
    "person_name",         # personal names (patient / caregiver / physician)
    "phone_number",        # telephone numbers
    "id_number",           # national identification numbers
    "address",             # residential / workplace addresses
    "specific_date",       # complete dates (birth, visit, surgery)
    "hospital_record_id",  # medical record / admission numbers
    "other_identifier",    # other identifying information
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一个医学数据脱敏专家。"
    "你的任务是识别儿科问诊对话中所有可能识别患者个人身份的信息（PII）。"
    "请逐条输出发现的PII，不要遗漏。"
    "仅输出JSON格式的结果，不包含任何其他文本。"
)

SCAN_PROMPT = """请仔细阅读以下儿科问诊对话，识别所有个人身份信息（PII）。

## PII类别
1. person_name（患者、家属、医护人员的姓名）
2. phone_number（手机号、座机号）
3. id_number（身份证号码）
4. address（家庭住址、单位地址等详细地址）
5. specific_date（出生日期、就诊日期、手术日期等完整日期。注意：仅描述时间长短的相对表达如"三天前""上周"不是PII，不要标记）
6. hospital_record_id（病历号、住院号）
7. other_identifier（其他可能识别身份的信息，如社保号、护照号、车牌号等）

## 对话
{dialogue_text}

## 输出格式
请输出一个JSON数组，每个元素包含以下字段：
- "category": PII类别（使用上述英文名，如 person_name / phone_number / specific_date 等）
- "text": 发现的具体PII文本
- "turn_id": 该PII所在的对话轮次编号（从1开始）
- "role": 说话人角色（patient 或 physician）
- "context": 该PII所在的上下文片段（约20字）

如果对话中未发现任何PII，输出空数组 []。

仅输出JSON数组，不要包含任何其他文字或markdown标记。"""


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def get_turns(dialogue: dict) -> list:
    """Return turns list, handling both intermediate and release schemas."""
    return dialogue.get("dialog") or dialogue.get("turns", [])


def turn_content(turn: dict) -> str:
    return turn.get("content") or turn.get("content_zh", "")


def turn_role(turn: dict) -> str:
    role = turn.get("role", "patient")
    if role in {"doctor", "physician"}:
        return "physician"
    return "patient"


def format_dialogue_text(dialogue: dict) -> str:
    """Render a dialogue as labelled text for the PII-scan prompt."""
    lines = []
    for i, turn in enumerate(get_turns(dialogue)):
        role_label = "患者/家属" if turn_role(turn) == "patient" else "医生"
        content = turn_content(turn)
        lines.append(f"[Turn {i + 1}] {role_label}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------
VALID_CATEGORIES = set(PII_CATEGORIES)
VALID_ROLES = {"patient", "physician"}


def parse_flags(raw: list | None) -> list[dict]:
    """Validate and normalise the LLM output into a list of PII flag dicts.

    Returns an empty list when *raw* is None, not a list, or contains
    entries that cannot be normalised.
    """
    if not isinstance(raw, list):
        return []

    flags = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = item.get("category", "")
        if category not in VALID_CATEGORIES:
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        role = item.get("role", "patient")
        if role not in VALID_ROLES:
            role = "patient"

        flags.append({
            "category": category,
            "text": text,
            "turn_id": item.get("turn_id", 1),
            "role": role,
            "context": (item.get("context") or "").strip(),
        })

    return flags


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def scan_dialogue(client: LLMClient, dialogue: dict) -> tuple[dict, bool]:
    """Scan one dialogue for PII.

    Returns (dialogue, success).  On success the dialogue dict carries
    a ``pii_flags`` key (list of dicts, may be empty).
    """
    text = format_dialogue_text(dialogue)
    prompt = SCAN_PROMPT.format(dialogue_text=text)
    raw = client.chat(SYSTEM_PROMPT, prompt, temperature=0.0)
    result = parse_json(raw)
    flags = parse_flags(result)
    dialogue["pii_flags"] = flags
    return dialogue, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM-based automated PII screening for pediatric dialogues"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL (cleaned dialogues)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL (dialogues with pii_flags field added)"
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N dialogues (0 = all)"
    )
    parser.add_argument(
        "--report", default=None,
        help="Optional path for aggregated JSON summary report"
    )
    args = parser.parse_args()

    # ---- load ----
    client = LLMClient()
    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} dialogues")

    # ---- parallel scan ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(data), "done": 0, "failed": 0, "with_pii": 0}
    all_findings: dict[str, list] = {}
    lock = threading.Lock()

    def _process(d):
        try:
            return scan_dialogue(client, d)
        except Exception:
            return d, False

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, d): i for i, d in enumerate(data)}
            for future in as_completed(futures):
                result, ok = future.result()
                with lock:
                    stats["done"] += 1
                    if not ok:
                        stats["failed"] += 1
                    if ok and result.get("pii_flags"):
                        stats["with_pii"] += 1
                        did = result.get("dialogue_id", "")
                        if did:
                            all_findings[did] = result["pii_flags"]
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if stats["done"] % 100 == 0:
                    print(
                        f"  {stats['done']}/{stats['total']} "
                        f"(failed: {stats['failed']}, pii: {stats['with_pii']})"
                    )

    success = stats["done"] - stats["failed"]
    clean = success - stats["with_pii"]
    clean_rate = (clean / success * 100) if success else 0
    print(
        f"Done: {stats['done']}  success={success}  failed={stats['failed']}  "
        f"with_pii={stats['with_pii']}  clean={clean} ({clean_rate:.1f}%)"
    )

    # ---- optional report ----
    if args.report:
        type_counts = Counter()
        for flags in all_findings.values():
            for f in flags:
                type_counts[f"type:{f['category']}"] += 1

        report = {
            "summary": {
                "total_dialogues": len(data),
                "dialogues_with_findings": stats["with_pii"],
                "clean_dialogues": clean,
                "clean_rate": f"{clean_rate:.1f}%",
                "findings_by_type": dict(type_counts.most_common()),
            },
            "findings": all_findings,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
