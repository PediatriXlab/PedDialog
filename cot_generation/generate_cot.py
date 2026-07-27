#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning annotations for physician turns.

Implements the annotation process described in paper Methods: Reasoning Annotation Process.
Key features:
  - Prospective constraint: only provides dialogue history up to the current turn
  - 5 response types (A-E) with type-specific reasoning structures
  - LLM-assisted generation followed by clinician validation

Usage:
    python generate_cot.py --input cleaned.jsonl --output cot_annotated.jsonl --workers 5
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cot_generation.prompts import CLASSIFY_PROMPT, TYPE_PROMPTS, format_dialog_history
from utils.llm_client import LLMClient, parse_json


def get_turns(dialogue: dict) -> list:
    return dialogue.get("dialog") or dialogue.get("turns", [])


def is_physician_turn(turn: dict) -> bool:
    return turn.get("role") in {"doctor", "physician"}


def turn_content(turn: dict) -> str:
    return turn.get("content") or turn.get("content_zh", "")


def classify_turn(client: LLMClient, dialog_history: str, doctor_response: str) -> str:
    prompt = CLASSIFY_PROMPT.format(dialog_history=dialog_history, doctor_response=doctor_response)
    result = client.chat("You are a medical expert analyzing physician responses.", prompt, temperature=0.1)
    for c in result.upper():
        if c in "ABCDE":
            return c
    return "A"


def generate_cot_for_turn(
    client: LLMClient, dialog_history: str, doctor_response: str, turn_type: str
) -> dict | None:
    prompt_template = TYPE_PROMPTS.get(turn_type, TYPE_PROMPTS["A"])
    prompt = prompt_template.format(dialog_history=dialog_history, doctor_response=doctor_response)
    data = client.chat_json(
        "You are a senior pediatrician generating clinical reasoning annotations.",
        prompt,
        temperature=0.3,
    )
    if data and "inner_thinking" in data:
        return data["inner_thinking"]
    return None


def process_dialogue(client: LLMClient, dialogue: dict) -> dict:
    turns = get_turns(dialogue)
    uses_release_schema = "turns" in dialogue and "dialog" not in dialogue
    for i, turn in enumerate(turns):
        if not is_physician_turn(turn):
            continue
        # Prospective constraint: only history up to this turn
        history = format_dialog_history(turns, i)
        doctor_content = turn_content(turn)

        # Classify and generate
        turn_type = classify_turn(client, history, doctor_content)
        thinking = generate_cot_for_turn(client, history, doctor_content, turn_type)

        if uses_release_schema:
            turn["cot_reasoning_zh"] = thinking or ""
        else:
            turn["inner_thinking"] = thinking or ""
        turn["response_type"] = turn_type

    dialogue["generation_info"] = {
        "generated": True,
        "total_doctor_turns": sum(1 for t in turns if is_physician_turn(t)),
        "generated_turns": sum(1 for t in turns if t.get("inner_thinking") or t.get("cot_reasoning_zh")),
    }
    return dialogue


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
    stats = {"total": len(data), "done": 0, "failed": 0}
    lock = threading.Lock()

    def process(d):
        try:
            return process_dialogue(client, d), True
        except Exception as e:
            return d, False

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, d): i for i, d in enumerate(data)}
            for future in as_completed(futures):
                result, ok = future.result()
                with lock:
                    stats["done"] += 1
                    if not ok:
                        stats["failed"] += 1
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if stats["done"] % 50 == 0:
                    print(f"  {stats['done']}/{stats['total']} (failed: {stats['failed']})")

    print(f"Done: {stats['done']}, Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
