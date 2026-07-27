#!/usr/bin/env python3
"""
CoT ablation experiment — fine-tune a compact model under 2 conditions.

Implements the downstream utility validation described in the paper
(Technical Validation: Illustrative Downstream Utility, Table 4).

Conditions (Table 4):
  1. base            — unfine-tuned Qwen2.5-0.5B-Instruct, reply-token PPL
  2. dialogue_only    — LoRA fine-tuned on physician replies only

The third row of Table 4 (Dialogue + reasoning annotation) uses the
**same dialogue_only model** with gold reasoning annotations additionally
provided as input context during evaluation.  See ``eval_with_reasoning.py``.

Label masking:
  - User turns and chat-template structural tokens are masked (-100).
  - TRAINING: loss on physician reply tokens only.
  - EVALUATION: ALL conditions — physician reply tokens ONLY
    (reasoning and [推理过程]/[回复] markers are masked), making PPL
    values directly comparable.

Usage:
    python cot_ablation.py --condition base
    python cot_ablation.py --condition dialogue_only
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_turns(data_dir):
    """Parse release-schema dialogues into structured turns.

    Reasoning is always loaded (used by ``eval_with_reasoning.py`` for the
    third Table 4 condition).  Returns train / val / test lists of::

        {"turns": [{"role": "user"|"assistant", "content": str,
                    "reasoning": str|None}, ...]}

    Splits are read from ``metadata.split`` (release schema) with a
    fallback to a top-level ``split`` key.
    """
    train_data, val_data, test_data = [], [], []
    for split_name in ("train", "val", "test"):
        path = Path(data_dir) / "peddialog_zh" / f"{split_name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            dialogues = [json.loads(line) for line in f if line.strip()]

        for d in dialogues:
            meta = d.get("metadata", {})
            sp = meta.get("split", d.get("split", "train"))
            raw_turns = d.get("dialog", d.get("turns", []))
            turns = []
            for t in raw_turns:
                role = t.get("role", "")
                content_zh = t.get("content_zh", t.get("content", ""))
                if role in ("patient", "患者", "家长"):
                    turns.append({"role": "user", "content": content_zh,
                                  "reasoning": None})
                elif role in ("doctor", "医生", "physician"):
                    cot = t.get("cot_reasoning_zh",
                                t.get("inner_thinking", ""))
                    turns.append({"role": "assistant", "content": content_zh,
                                  "reasoning": cot or None})
            if len(turns) >= 2:
                if sp == "train":
                    train_data.append({"turns": turns})
                elif sp == "val":
                    val_data.append({"turns": turns})
                else:
                    test_data.append({"turns": turns})

    print(f"  train={len(train_data)}  val={len(val_data)}  "
          f"test={len(test_data)}")
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def _encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def build_example(turns, tok, max_len, loss_target):
    """Build token ids + masked labels for one Qwen-formatted dialogue.

    ``loss_target``:
      ``"reply"`` — keep only physician reply (+ footer) in the loss;
                    reasoning / markers are masked.
    """
    input_ids, labels = [], []
    for turn in turns:
        role = turn["role"]
        header = _encode(tok, f"<|im_start|>{role}\n")
        footer = _encode(tok, "<|im_end|>\n")
        if role == "user":
            content = _encode(tok, turn["content"])
            input_ids += header + content + footer
            labels += [-100] * (len(header) + len(content) + len(footer))
        else:  # assistant
            reply = _encode(tok, turn["content"])
            reasoning = turn.get("reasoning")
            if reasoning:
                m_open = _encode(tok, "[推理过程]\n")
                r_ids = _encode(tok, reasoning)
                m_close = _encode(tok, "\n[回复]\n")
                input_ids += (header + m_open + r_ids + m_close +
                              reply + footer)
                masked = (len(header) + len(m_open) + len(r_ids) +
                          len(m_close))
                labels += [-100] * masked + reply + footer
            else:
                input_ids += header + reply + footer
                labels += [-100] * len(header) + reply + footer

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def make_tokenize_fn(tok, max_len, loss_target, strip_reasoning=False):
    """Factory for ``datasets.Dataset.map`` batched tokenization.

    ``strip_reasoning`` drops the reasoning field so every condition is
    scored on identical, reply-only target tokens at evaluation time.
    """
    def tokenize_fn(examples):
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for turns in examples["turns"]:
            t = turns
            if strip_reasoning:
                t = [{**x, "reasoning": None} for x in turns]
            ex = build_example(t, tok, max_len, loss_target)
            out["input_ids"].append(ex["input_ids"])
            out["labels"].append(ex["labels"])
            out["attention_mask"].append(ex["attention_mask"])
        return out
    return tokenize_fn


def _eval_to_result(eval_result):
    loss = eval_result.get("eval_loss")
    eval_result["eval_perplexity"] = (math.exp(loss)
                                       if loss and loss > 0
                                       else float("inf"))
    eval_result["eval_target"] = "reply_tokens_only"
    return eval_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CoT ablation — Table 4 downstream utility experiment"
    )
    parser.add_argument(
        "--condition", required=True,
        choices=["base", "dialogue_only"],
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--data_dir", default="../data/release",
        help="Path to data/release (default: ../data/release)",
    )
    parser.add_argument(
        "--output_dir", default="results/ablation",
        help="Output directory (default: results/ablation)",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=2048)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Condition 1: base (no fine-tuning)
    # ------------------------------------------------------------------
    if args.condition == "base":
        print("Base: evaluating reply-token PPL (no fine-tuning).")
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   DataCollatorForSeq2Seq, Trainer,
                                   TrainingArguments)
        from datasets import Dataset

        _, _, test_data = load_turns(args.data_dir)
        if not test_data:
            print("No test data found.")
            return

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        tokenize_fn = make_tokenize_fn(tokenizer, args.max_len, "reply",
                                       strip_reasoning=True)
        test_ds = (Dataset.from_list(test_data)
                   .map(tokenize_fn, batched=True,
                        remove_columns=["turns"]))
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                          return_tensors="pt")

        base_dir = os.path.join(args.output_dir, "base")
        os.makedirs(base_dir, exist_ok=True)

        dummy_args = TrainingArguments(
            output_dir=base_dir, per_device_eval_batch_size=4,
            report_to="none", seed=args.seed,
        )
        trainer = Trainer(model=model, args=dummy_args, eval_dataset=test_ds,
                          data_collator=collator)
        eval_result = _eval_to_result(trainer.evaluate())
        print(f"Base — eval_loss: {eval_result['eval_loss']:.4f}  "
              f"reply-token PPL: {eval_result['eval_perplexity']:.2f}")

        with open(os.path.join(base_dir, "results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
        print(f"Saved → {base_dir}/results.json")
        return

    # ------------------------------------------------------------------
    # Condition 2: dialogue_only (LoRA fine-tuning on physician replies)
    # ------------------------------------------------------------------
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   TrainingArguments, Trainer,
                                   DataCollatorForSeq2Seq)
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch transformers peft datasets")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    train_data, val_data, test_data = load_turns(args.data_dir)
    if not train_data:
        print("Error: no training data loaded")
        sys.exit(1)

    cond_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(cond_dir, exist_ok=True)

    # Save processed data for reproducibility
    with open(os.path.join(cond_dir, "train.json"), "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(cond_dir, "val.json"), "w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # ---- model & LoRA ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- tokenize ----
    # TRAIN: loss on physician reply tokens only.
    # EVAL:  loss on reply tokens only (reasoning stripped → identical target
    #        across conditions 1 and 2; condition 3 keeps reasoning — see
    #        eval_with_reasoning.py).
    train_tok_fn = make_tokenize_fn(tokenizer, args.max_len, "reply",
                                    strip_reasoning=True)
    eval_tok_fn = make_tokenize_fn(tokenizer, args.max_len, "reply",
                                   strip_reasoning=True)

    train_ds = (Dataset.from_list(train_data)
                .map(train_tok_fn, batched=True,
                     remove_columns=["turns"]))
    val_ds = None
    if val_data:
        val_ds = (Dataset.from_list(val_data)
                  .map(eval_tok_fn, batched=True,
                       remove_columns=["turns"]))

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True,
                                      return_tensors="pt")

    # ---- train ----
    training_args = TrainingArguments(
        output_dir=cond_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        eval_strategy="epoch" if val_ds else "no",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(os.path.join(cond_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(cond_dir, "final_model"))

    # ---- evaluate on test ----
    if test_data:
        test_ds = (Dataset.from_list(test_data)
                   .map(eval_tok_fn, batched=True,
                        remove_columns=["turns"]))
        eval_result = _eval_to_result(trainer.evaluate(test_ds))
        print(f"Test reply-token loss: {eval_result['eval_loss']:.4f}  "
              f"PPL: {eval_result['eval_perplexity']:.2f}")
        with open(os.path.join(cond_dir, "results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)

    print(f"Done → {cond_dir}")


if __name__ == "__main__":
    main()
