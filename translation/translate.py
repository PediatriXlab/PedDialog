#!/usr/bin/env python3
"""
Translate dialogues and CoT annotations from Chinese to English.

Implements the translation process described in paper Methods (English translation section).
Translates per-dialogue to maintain full context. Outputs JSONL with content_en and
cot_reasoning_en or inner_thinking_en fields added to each physician turn.

Usage:
    python translate.py --input dialogues.jsonl --output translated.jsonl --workers 10
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.llm_client import LLMClient, parse_json


def get_turns(dialogue: dict) -> list:
    return dialogue.get("dialog") or dialogue.get("turns", [])


def is_physician_turn(turn: dict) -> bool:
    return turn.get("role") in {"doctor", "physician"}


def turn_content_zh(turn: dict) -> str:
    return turn.get("content") or turn.get("content_zh", "")


def turn_reasoning_zh(turn: dict) -> str:
    return turn.get("inner_thinking") or turn.get("cot_reasoning_zh", "")


SYSTEM_PROMPT = """You are a professional medical translator specializing in pediatric telemedicine. Translate the following Chinese dialogue into English, preserving:
1. Medical accuracy — clinical meanings must be conveyed faithfully.
2. Dialogue naturalness — conversational tone appropriate for medical chat.
3. Cultural context — adapt China-specific terms for international readers.
4. Reasoning structure — maintain logical organization of annotations.
Output ONLY valid JSON. No markdown fences."""

DIALOGUE_PROMPT = """Translate this pediatric telemedicine dialogue from Chinese to English.

## Dialogue (Chinese)
{dialogue_text}

## Instructions
Return a JSON object with this exact structure:
{{
  "turns": [
    {{
      "turn_index": 0,
      "content_en": "English translation",
      "reasoning_en": "English translation of reasoning (only for physician turns)"
    }}
  ]
}}

Rules:
- turn_index must match the input order (0-based).
- Translate every turn's content.
- For physician turns with reasoning annotations, translate them into reasoning_en.
- For patient turns, set reasoning_en to null.
- Preserve medical terminology accurately.
- Output ONLY the JSON object."""


def format_dialogue_text(dialogue: dict) -> str:
    lines = []
    for i, turn in enumerate(get_turns(dialogue)):
        role = "Patient/Parent" if turn["role"] == "patient" else "Physician"
        lines.append(f"[Turn {i}] {role}: {turn_content_zh(turn)}")
        thinking = turn_reasoning_zh(turn)
        if thinking:
            lines.append(f"  [Reasoning]: {thinking}")
    return "\n".join(lines)


def translate_dialogue(client: LLMClient, dialogue: dict) -> tuple[dict, bool]:
    text = format_dialogue_text(dialogue)
    prompt = DIALOGUE_PROMPT.format(dialogue_text=text)
    raw = client.chat(SYSTEM_PROMPT, prompt, temperature=0.1, max_retries=2)
    data = parse_json(raw)

    if not data or "turns" not in data:
        return dialogue, False

    turn_map = {t["turn_index"]: t for t in data["turns"] if "turn_index" in t}
    uses_release_schema = "turns" in dialogue and "dialog" not in dialogue
    for i, turn in enumerate(get_turns(dialogue)):
        tr = turn_map.get(i)
        if tr:
            turn["content_en"] = tr.get("content_en", "")
            if is_physician_turn(turn):
                reasoning_en = tr.get("reasoning_en") or tr.get("inner_thinking_en") or tr.get("cot_reasoning_en") or ""
                if uses_release_schema:
                    turn["cot_reasoning_en"] = reasoning_en
                else:
                    turn["inner_thinking_en"] = reasoning_en
        else:
            turn["content_en"] = ""

    return dialogue, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = LLMClient()

    data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} dialogues")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": len(data), "success": 0, "failed": 0}
    lock = threading.Lock()

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(translate_dialogue, client, d): i for i, d in enumerate(data)}
            for future in as_completed(futures):
                result, ok = future.result()
                with lock:
                    if ok:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    done = stats["success"] + stats["failed"]
                if ok:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if done % 100 == 0:
                    print(f"  {done}/{stats['total']} (failed: {stats['failed']})")

    print(f"Success: {stats['success']}, Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
