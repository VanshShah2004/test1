"""
Simplified SLM Training Script - Avoids TensorFlow/Keras conflicts.
Uses only PyTorch and transformers with proper environment setup.
"""

import os
import sys

# CRITICAL: Set these BEFORE any transformers imports
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Now import everything else
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Import transformers (will skip TensorFlow now)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)


class ResumeScoringDataset(Dataset):
    """Dataset for resume scoring task."""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


def load_training_data(dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl") -> Tuple[List[str], List[float]]:
    """Load and prepare training data from kaggle_dataset."""
    texts = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return [], []
    
    print(f"📂 Loading training data from {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                
                input_data = record.get("input", {})
                resume_text = input_data.get("resume", "")
                job_description = input_data.get("job_description", "")
                
                output_data = record.get("output", {})
                scores = output_data.get("scores", {})
                aggregated = scores.get("aggregated_scores", {})
                total_score = aggregated.get("macro_scores", 0.0)
                
                # Normalize to 0-100 scale
                if total_score > 10:
                    total_score_normalized = total_score
                else:
                    total_score_normalized = total_score * 10
                
                # Format input text
                formatted_text = f"""RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

TASK: Score this resume against the job description on a scale of 0-100."""
                
                texts.append(formatted_text)
                labels.append(total_score_normalized)
                
            except Exception as e:
                if line_num <= 5:
                    print(f"⚠️  Skipping line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(texts)} training samples")
    return texts, labels


def prepare_datasets(texts: List[str], labels: List[float], test_size=0.2, val_size=0.1):
    """Split data into train/validation/test sets."""
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, 
        test_size=val_size/(1-test_size), 
        random_state=42
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"  • Training: {len(train_texts)} samples")
    print(f"  • Validation: {len(val_texts)} samples")
    print(f"  • Test: {len(test_texts)} samples")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def main():
    """Main training function."""
    print("="*70)
    print("SLM MODEL TRAINING (Simplified - PyTorch Only)")
    print("="*70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load data
    texts, labels = load_training_data()
    
    if len(texts) == 0:
        print("❌ No training data available. Cannot train model.")
        return
    
    # Prepare datasets
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_datasets(
        texts, labels
    )
    
    # Load model (DistilBERT - fast and efficient)
    model_name = "distilbert-base-uncased"
    print(f"\n📥 Loading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Prepare datasets
    print("\n🔄 Preparing datasets...")
    train_dataset = ResumeScoringDataset(train_texts, train_labels, tokenizer)
    val_dataset = ResumeScoringDataset(val_texts, val_labels, tokenizer)
    test_dataset = ResumeScoringDataset(test_texts, test_labels, tokenizer)
    
    # Create output directory
    output_dir = "models/trained_slm"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,  # Disable FP16 to avoid issues
        report_to="none"
    )
    
    # Metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        within_tolerance = np.sum(np.abs(predictions - labels) <= 10)
        accuracy = within_tolerance / len(labels)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'accuracy': float(accuracy)
        }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n🚀 Starting training...")
    try:
        train_result = trainer.train()
        print("\n✅ Training completed!")
        
        # Evaluate
        print("\n📊 Evaluating on validation set...")
        val_results = trainer.evaluate()
        print(f"   Validation RMSE: {val_results['eval_rmse']:.2f}")
        print(f"   Validation MAE: {val_results['eval_mae']:.2f}")
        print(f"   Validation R²: {val_results['eval_r2']:.3f}")
        print(f"   Validation Accuracy (±10): {val_results['eval_accuracy']:.1%}")
        
        print("\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"   Test RMSE: {test_results['eval_rmse']:.2f}")
        print(f"   Test MAE: {test_results['eval_mae']:.2f}")
        print(f"   Test R²: {test_results['eval_r2']:.3f}")
        print(f"   Test Accuracy (±10): {test_results['eval_accuracy']:.1%}")
        
        # Save
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save results
        results = {
            'validation': {
                'rmse': val_results['eval_rmse'],
                'mae': val_results['eval_mae'],
                'r2': val_results['eval_r2'],
                'accuracy': val_results['eval_accuracy']
            },
            'test': {
                'rmse': test_results['eval_rmse'],
                'mae': test_results['eval_mae'],
                'r2': test_results['eval_r2'],
                'accuracy': test_results['eval_accuracy']
            }
        }
        
        with open(f"{output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Model saved to {output_dir}")
        print(f"   Results saved to {output_dir}/training_results.json")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

