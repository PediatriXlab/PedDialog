# PedDialog-CoT Code

Code supporting the PedDialog-CoT dataset, as described in the paper *"A Pediatric Consultation Dialogue Dataset with Pediatrician-Validated Reasoning Annotations"*.

## Directory Structure

```
code/
├── data_processing/       # Data filtering, cleaning, and dataset splitting
│   ├── filter_and_clean.py
│   └── split_dataset.py
├── deidentification/      # PII screening + audit
│   ├── llm_pii_screen.py       # Stage 1: LLM-based automated PII screening
│   └── pii_scan.py             # Validation B1: regex-based post-hoc audit
├── cot_generation/        # CoT reasoning annotation generation
│   ├── generate_cot.py
│   └── prompts.py         # Prompt templates (5 response types A-E)
├── translation/           # Dialogue and CoT translation (ZH→EN)
│   └── translate.py
├── validation/            # Downstream experiments and evaluation templates
│   ├── cot_ablation.py          # LoRA fine-tuning: base + dialogue_only (Table 4, rows 1–2)
│   ├── eval_with_reasoning.py   # Table 4, row 3: dialogue_only model + reasoning input
│   ├── collect_results.py       # Aggregate results into summary JSON + CSV
│   └── eval_templates.py        # B2/B3 evaluation template generation
├── utils/
│   └── llm_client.py      # Shared LLM API wrapper
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your LLM API credentials
```

## Usage

The scripts accept the public release schema (`turns`, `physician`,
`cot_reasoning_zh`/`cot_reasoning_en`) and remain compatible with the
intermediate processing schema used during dataset construction (`dialog`,
`doctor`, `inner_thinking`).

### 1. Data Processing

```bash
# Filter and clean raw dialogues
python data_processing/filter_and_clean.py --input raw.jsonl --output cleaned.jsonl

# Split into train/val/test (stratified by department)
python data_processing/split_dataset.py --input cleaned.jsonl --outdir splits/
```

### 2. CoT Annotation Generation

```bash
# Generate reasoning annotations for physician turns
python cot_generation/generate_cot.py --input cleaned.jsonl --output annotated.jsonl --workers 5
```

### 3. Translation

```bash
# Translate dialogues and annotations to English
python translation/translate.py --input annotated.jsonl --output translated.jsonl --workers 10
```

### 4. De-identification

```bash
# Stage 1: LLM-based automated PII screening
# (uses Qwen3-30B-A3B-Instruct-250720 or any OpenAI-compatible LLM)
python deidentification/llm_pii_screen.py \
    --input cleaned.jsonl --output screened.jsonl \
    --workers 5 --report pii_report.json

# Validation B1: regex-based post-hoc audit
python deidentification/pii_scan.py --input cleaned.jsonl --output pii_results.json
```

### 5. Validation

```bash
# Table 4, row 1: evaluate base model on reply-token PPL
python validation/cot_ablation.py --condition base \
    --data_dir ../data/release --output_dir results/ablation

# Table 4, row 2: LoRA fine-tune on dialogue replies (requires GPU)
python validation/cot_ablation.py --condition dialogue_only \
    --data_dir ../data/release --output_dir results/ablation

# Table 4, row 3: same dialogue_only model + reasoning as input at eval
python validation/eval_with_reasoning.py \
    --adapter results/ablation/dialogue_only/final_model \
    --data_dir ../data/release

# Collect all results into summary JSON + CSV
python validation/collect_results.py --results_dir results/ablation

# Generate evaluation templates for expert review
python validation/eval_templates.py --input bilingual.jsonl --outdir templates/
```

## License

[MIT/Apache 2.0]
