# SLM Model Training Guide

This directory contains scripts to train a Small Language Model (SLM) on the resume scoring task using the kaggle_dataset.

## Overview

The training process fine-tunes a pre-trained model (e.g., DistilBERT, Phi-3, Llama 3.2) to predict resume scores given resume text and job descriptions.

## Training Data

- **Source**: `kaggle_dataset/hf_clean_data/resume_score_details.jsonl`
- **Format**: JSONL with `input` (resume + job_description) and `output` (scores)
- **Task**: Regression (predict score 0-100)

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers torch datasets accelerate scikit-learn
```

### 2. Train the Model

```bash
python slm_training/train_slm_model.py
```

This will:
- Load data from `kaggle_dataset/hf_clean_data/resume_score_details.jsonl`
- Split into train/validation/test sets
- Fine-tune DistilBERT (or other model) on scoring task
- Save trained model to `models/trained_slm/`
- Display training metrics (accuracy, RMSE, MAE, R²)

### 3. Evaluate the Trained Model

```bash
python slm_training/evaluate_trained_model.py
```

This calculates:
- Accuracy (±10 points)
- Precision/Recall for high scores (≥80)
- RMSE, MAE, R², Correlation

## Model Configuration

Edit `train_slm_model.py` to customize:

- **Model**: Change `model_name` (default: `distilbert-base-uncased`)
  - Options: `microsoft/Phi-3-mini-4k-instruct`, `microsoft/DialoGPT-small`, etc.
- **Epochs**: `num_epochs` (default: 5)
- **Batch Size**: `batch_size` (default: 8)
- **Learning Rate**: `learning_rate` (default: 2e-5)

## Output

Trained model saved to:
- `models/trained_slm/` - Model weights and tokenizer
- `models/trained_slm/training_results.json` - Training metrics
- `models/trained_slm/evaluation_results.json` - Test set metrics

## Using Trained Model

The trained model is automatically used by `HybridScoringAgent` if available:

```python
from agents.hybrid_scoring_agent import HybridScoringAgent

agent = HybridScoringAgent(use_trained_slm=True)  # Uses trained model
# Falls back to rule-based if model not found
```

## Training Metrics

After training, you'll see:
- **Training Loss**: Loss during training
- **Validation Metrics**: RMSE, MAE, R², Accuracy
- **Test Metrics**: Final performance on held-out test set

Expected performance (example):
- Accuracy (±10 points): 85-95%
- RMSE: 8-15 points
- R²: 0.75-0.90

