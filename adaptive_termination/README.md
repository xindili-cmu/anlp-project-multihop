# Approach A: Classifier-Based Termination — Diagnostic Evaluation

This directory contains the code for Approach A from our final project on adaptive retrieval termination for multi-hop RAG. We train a DeBERTa-base sufficiency classifier and evaluate it diagnostically against real IRCoT pipeline outputs, rather than deploying it end-to-end.

## Overview

Given a question Q, accumulated evidence E, and a candidate answer A, the classifier predicts whether E is sufficient to support A. We:

1. Train the classifier on 271K triples constructed from HotpotQA train (clean validation F1 = 0.998)
2. Extract 2,128 real-distribution triples from baseline IRCoT trajectories on 500 HotpotQA dev questions
3. Evaluate the classifier on these real triples against two proxy labels (SF coverage and EM)
4. Analyze the resulting calibration, failure modes, and parametric-vs-retrieval composition

## Pipeline

### Step 1 — Train the sufficiency classifier

```bash
# Construct training data from HotpotQA train
python construct_training_data.py \
    --input_file /path/to/hotpotqa/train.json \
    --output_file training_data.jsonl

# Fine-tune DeBERTa-base
python train_deberta.py \
    --train_file training_data_train.jsonl \
    --val_file training_data_val.jsonl \
    --output_dir deberta_sufficiency_classifier \
    --epochs 5 --batch_size 16 --learning_rate 2e-5 --max_length 512
```

### Step 2 — Extract real-distribution evaluation triples

```bash
# Parse IRCoT trajectory output
python parse_chains.py \
    --chains_file /path/to/prediction_chains.txt \
    --predictions_file /path/to/prediction.json \
    --output_file parsed_trajectories.jsonl

# Build title-to-paragraph index from HotpotQA distractor data
python build_title_index.py \
    --hotpot_dev /path/to/hotpotqa/dev.json \
    --trajectories parsed_trajectories.jsonl \
    --output_file title_index.json

# Construct (Q, E_t, A_t) triples
python extract_triples.py \
    --trajectories parsed_trajectories.jsonl \
    --title_index title_index.json \
    --output_file triples_raw.jsonl

# Assign dual proxy labels
python add_labels.py \
    --triples triples_raw.jsonl \
    --title_index title_index.json \
    --predictions /path/to/prediction.json \
    --output_file triples_with_labels.jsonl
```

### Step 3 — Run inference with the trained classifier

Inference is done in a separate Colab notebook (A100 GPU) that loads the best_model checkpoint and scores each triple. Output is `triples_with_scores.jsonl`.

### Step 4 — Compute metrics and analyses

```bash
# Aggregate metrics + by-hop + by-miss-rate
python compute_metrics.py --input triples_with_scores.jsonl

# Threshold sensitivity analysis
python threshold_analysis.py --input triples_with_scores.jsonl

# Calibration / reliability diagrams
python calibration_analysis.py --input triples_with_scores.jsonl

# Failure mode breakdown (Mode 1 / 2 / 3 / Successful)
python failure_mode_breakdown.py --input triples_with_scores.jsonl

# Generate all figures
python plot_distributions.py --input triples_with_scores.jsonl
```

## Key results

- **Clean validation F1 = 0.998** (gold E, gold A)
- **Real-distribution F1 = 0.684** (BM25 E, hedged LLM A) — a 0.314 drop
- **AUC = 0.891** preserved despite F1 drop, indicating ranking ability survives but calibration does not
- **37% false positive rate** at high-confidence (score > 0.9) predictions
- **65%** of IRCoT's successful answers come from cases with incomplete retrieved evidence — indicating LLM parametric knowledge, not retrieval, drives a substantial fraction of baseline performance

See the final report (Section 5) for full details.

## Dependencies

- Python 3.8+
- PyTorch 2.x with CUDA
- `transformers >= 4.40`
- `numpy`, `matplotlib`

```bash
pip install torch transformers numpy matplotlib
```

## File reference

| File | Role |
|---|---|
| `construct_training_data.py` | Build 271K training triples from HotpotQA train |
| `train_deberta.py` | Fine-tune DeBERTa-base for binary sufficiency classification |
| `parse_chains.py` | Extract per-hop snapshots from IRCoT trajectory output |
| `build_title_index.py` | Build title→paragraph reverse lookup from HotpotQA distractor data |
| `extract_triples.py` | Construct (Q, E_t, A_t) evaluation triples |
| `add_labels.py` | Assign dual proxy labels (SF coverage, EM) to triples |
| `compute_metrics.py` | Compute F1 / AUC / ECE, including stratified breakdowns |
| `threshold_analysis.py` | Sweep classification thresholds and identify optimal τ |
| `calibration_analysis.py` | Reliability diagram + per-bin ECE decomposition |
| `failure_mode_breakdown.py` | Classifier performance by IRCoT failure mode |
| `plot_distributions.py` | Generate all paper figures |

## Trained model and data artifacts

The trained DeBERTa checkpoint (~500MB) and intermediate evaluation files exceed Github's file size limits. Available on request from the authors.
