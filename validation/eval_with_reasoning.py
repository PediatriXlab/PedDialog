#!/usr/bin/env python3
"""
Table 4, condition 3: Dialogue + reasoning annotation.

Uses the **same dialogue_only model** as condition 2, but additionally
provides the gold reasoning annotation as input context during evaluation.
Perplexity is computed exclusively over physician reply tokens, directly
comparable with the other two conditions.

Usage:
    python eval_with_reasoning.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter results/ablation/dialogue_only/final_model \
        --data_dir ../data/release
"""

import argparse
import json
import math
import os

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           DataCollatorForSeq2Seq, Trainer,
                           TrainingArguments)
from datasets import Dataset
from peft import PeftModel

import cot_ablation as m


def main():
    parser = argparse.ArgumentParser(
        description="Table 4 condition 3: dialogue_only model + reasoning in context"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base HuggingFace model ID",
    )
    parser.add_argument(
        "--adapter", default="results/ablation/dialogue_only/final_model",
        help="Path to the dialogue_only LoRA adapter",
    )
    parser.add_argument(
        "--data_dir", default="../data/release",
        help="Path to data/release",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for results JSON (default: <adapter>/../results_with_reasoning.json)",
    )
    parser.add_argument("--max_len", type=int, default=2048)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.adapter), "..",
            "results_with_reasoning.json",
        )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.config.pad_token_id = tok.pad_token_id

    # Same dialogue_only model as condition 2, but keeping reasoning in the
    # input context.  Loss still on reply tokens only → comparable PPL.
    _, _, test = m.load_turns(args.data_dir)
    tokenize_fn = m.make_tokenize_fn(tok, args.max_len, "reply",
                                      strip_reasoning=False)
    test_ds = (Dataset.from_list(test)
               .map(tokenize_fn, batched=True, remove_columns=["turns"]))

    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True,
                                      return_tensors="pt")
    training_args = TrainingArguments(
        output_dir="/tmp/_reasoning_eval_tmp",
        per_device_eval_batch_size=4,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, eval_dataset=test_ds,
                      data_collator=collator)
    res = trainer.evaluate()
    loss = res["eval_loss"]
    res["eval_perplexity"] = math.exp(loss)
    res["eval_target"] = "reply_tokens_only; reasoning kept in context"

    print("=" * 64)
    print("  Table 4, condition 3:")
    print("  Dialogue-only model + gold reasoning as input")
    print(f"  eval_loss        = {loss:.4f}")
    print(f"  reply-token PPL   = {res['eval_perplexity']:.2f}")
    print("-" * 64)
    print("  Reference (Table 4, reply-token PPL):")
    print("    Base model                          = 28.95")
    print("    Dialogue only                       = 18.73")
    print("    Dialogue + reasoning annotation     =  8.49")
    print("=" * 64)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
