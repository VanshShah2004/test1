"""
Train SLM (Small Language Model) on resume scoring task using kaggle_dataset.
Fine-tunes a small model to predict resume scores given resume text and job description.
"""

import json
import os
import sys

# Set environment to avoid TensorFlow/Keras conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import transformers after setting environment variables
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
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


def load_training_data(dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl") -> Tuple[List[Dict], List[float]]:
    """
    Load and prepare training data from kaggle_dataset.
    
    Returns:
        texts: List of formatted text (resume + job_description)
        labels: List of total scores (0-100)
    """
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
                
                # Extract input
                input_data = record.get("input", {})
                resume_text = input_data.get("resume", "")
                job_description = input_data.get("job_description", "")
                
                # Extract ground truth score
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
                formatted_text = f"""
RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

TASK: Score this resume against the job description on a scale of 0-100.
"""
                texts.append(formatted_text)
                labels.append(total_score_normalized)
                
            except Exception as e:
                if line_num <= 5:  # Only print first few errors
                    print(f"⚠️  Skipping line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(texts)} training samples")
    return texts, labels


def prepare_datasets(texts: List[str], labels: List[float], test_size=0.2, val_size=0.1):
    """
    Split data into train/validation/test sets.
    
    Returns:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
    """
    # First split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    # Second split: train vs val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_size/(1-test_size), random_state=42
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"  • Training: {len(train_texts)} samples")
    print(f"  • Validation: {len(val_texts)} samples")
    print(f"  • Test: {len(test_texts)} samples")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def train_slm_model(
    model_name: str = "distilbert-base-uncased",
    dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
    output_dir: str = "models/trained_slm",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """
    Train SLM model on resume scoring task.
    
    Args:
        model_name: Base model to fine-tune (e.g., Phi-3, Llama 3.2, Gemma)
        dataset_path: Path to training dataset
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    print("="*70)
    print("SLM MODEL TRAINING")
    print("="*70)
    
    # Load data
    texts, labels = load_training_data(dataset_path)
    
    if len(texts) == 0:
        print("❌ No training data available. Cannot train model.")
        return None
    
    # Prepare datasets
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_datasets(
        texts, labels
    )
    
    # Load tokenizer and model (PyTorch only, no TensorFlow)
    print(f"\n📥 Loading base model: {model_name}...")
    try:
        # Force PyTorch backend
        from transformers import set_seed
        set_seed(42)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        # Add padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # For regression task, use sequence classification head with 1 output
        # Use torch_dtype to ensure PyTorch model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression: single output (score 0-100)
            problem_type="regression",
            torch_dtype=torch.float32
        )
        
        # Resize token embeddings if we added a new pad token
        if tokenizer.pad_token_id is not None and model.config.vocab_size < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        
        print("✅ Model loaded successfully (PyTorch)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"   Trying alternative: Using distilbert-base-uncased")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    print("\n🔄 Preparing datasets...")
    train_dataset = ResumeScoringDataset(train_texts, train_labels, tokenizer)
    val_dataset = ResumeScoringDataset(val_texts, val_labels, tokenizer)
    test_dataset = ResumeScoringDataset(test_texts, test_labels, tokenizer)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
        report_to="none",
        remove_unused_columns=False
    )
    
    # Metric function for evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        # R² score
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Accuracy (within ±10 points)
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
    
    # Train model
    print("\n🚀 Starting training...")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    try:
        train_result = trainer.train()
        print("\n✅ Training completed!")
        print(f"   Training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on validation set
        print("\n📊 Evaluating on validation set...")
        val_results = trainer.evaluate()
        print(f"   Validation RMSE: {val_results['eval_rmse']:.2f}")
        print(f"   Validation MAE: {val_results['eval_mae']:.2f}")
        print(f"   Validation R²: {val_results['eval_r2']:.3f}")
        print(f"   Validation Accuracy (±10): {val_results['eval_accuracy']:.1%}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"   Test RMSE: {test_results['eval_rmse']:.2f}")
        print(f"   Test MAE: {test_results['eval_mae']:.2f}")
        print(f"   Test R²: {test_results['eval_r2']:.3f}")
        print(f"   Test Accuracy (±10): {test_results['eval_accuracy']:.1%}")
        
        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training results
        results = {
            'training_loss': train_result.training_loss,
            'validation_metrics': {
                'rmse': val_results['eval_rmse'],
                'mae': val_results['eval_mae'],
                'r2': val_results['eval_r2'],
                'accuracy': val_results['eval_accuracy']
            },
            'test_metrics': {
                'rmse': test_results['eval_rmse'],
                'mae': test_results['eval_mae'],
                'r2': test_results['eval_r2'],
                'accuracy': test_results['eval_accuracy']
            },
            'model_name': model_name,
            'training_samples': len(train_texts),
            'validation_samples': len(val_texts),
            'test_samples': len(test_texts)
        }
        
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Model and results saved!")
        print(f"   Results: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training function."""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Train model
    results = train_slm_model(
        model_name="distilbert-base-uncased",  # Start with smaller model
        dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
        output_dir="models/trained_slm",
        num_epochs=5,
        batch_size=8,
        learning_rate=2e-5
    )
    
    if results:
        print("\n" + "="*70)
        print("TRAINING COMPLETE - FINAL METRICS")
        print("="*70)
        print(f"\n📊 Test Set Performance:")
        print(f"  • Accuracy (±10 points): {results['test_metrics']['accuracy']:.1%}")
        print(f"  • Precision (High Scores): See detailed evaluation")
        print(f"  • RMSE: {results['test_metrics']['rmse']:.2f} points")
        print(f"  • MAE: {results['test_metrics']['mae']:.2f} points")
        print(f"  • R² Score: {results['test_metrics']['r2']:.3f}")
    else:
        print("\n❌ Training failed. Check errors above.")


if __name__ == "__main__":
    main()

