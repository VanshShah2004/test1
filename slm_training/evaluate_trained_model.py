"""
Evaluate trained SLM model and calculate accuracy and precision metrics.
"""

import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Any


class TrainedSLMPredictor:
    """Load and use trained SLM model for predictions."""
    
    def __init__(self, model_path: str = "models/trained_slm"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load trained model and tokenizer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train model first.")
        
        print(f"📥 Loading trained model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")
    
    def predict(self, resume_text: str, job_description: str) -> float:
        """
        Predict score for resume-job pair.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text content
            
        Returns:
            Predicted score (0-100)
        """
        # Format input
        formatted_text = f"""
RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

TASK: Score this resume against the job description on a scale of 0-100.
"""
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            predicted_score = logits.item()
        
        # Clamp to 0-100 range
        predicted_score = max(0.0, min(100.0, predicted_score))
        
        return predicted_score
    
    def predict_batch(self, resume_texts: List[str], job_descriptions: List[str]) -> List[float]:
        """Predict scores for multiple resume-job pairs."""
        predictions = []
        for resume_text, job_desc in zip(resume_texts, job_descriptions):
            try:
                pred = self.predict(resume_text, job_desc)
                predictions.append(pred)
            except Exception as e:
                print(f"⚠️  Prediction failed: {e}")
                predictions.append(0.0)
        return predictions


def evaluate_trained_model(
    model_path: str = "models/trained_slm",
    dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
    test_size: int = None
) -> Dict[str, Any]:
    """
    Evaluate trained model on dataset and calculate accuracy/precision.
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to evaluation dataset
        test_size: Number of samples to evaluate (None = all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*70)
    print("EVALUATING TRAINED SLM MODEL")
    print("="*70)
    
    # Load model
    predictor = TrainedSLMPredictor(model_path)
    
    # Load test data
    print(f"\n📂 Loading evaluation data from {dataset_path}...")
    texts = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if test_size and len(texts) >= test_size:
                break
                
            try:
                record = json.loads(line.strip())
                
                input_data = record.get("input", {})
                resume_text = input_data.get("resume", "")
                job_description = input_data.get("job_description", "")
                
                output_data = record.get("output", {})
                scores = output_data.get("scores", {})
                aggregated = scores.get("aggregated_scores", {})
                total_score = aggregated.get("macro_scores", 0.0)
                
                # Normalize to 0-100
                if total_score > 10:
                    total_score_normalized = total_score
                else:
                    total_score_normalized = total_score * 10
                
                texts.append((resume_text, job_description))
                labels.append(total_score_normalized)
                
            except Exception as e:
                continue
    
    print(f"✅ Loaded {len(texts)} evaluation samples")
    
    # Predict
    print("\n🔄 Running predictions...")
    predictions = []
    for i, (resume_text, job_desc) in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(texts)}...")
        pred = predictor.predict(resume_text, job_desc)
        predictions.append(pred)
    
    print(f"✅ Completed predictions for {len(predictions)} samples")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Regression metrics
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Correlation
    correlation = np.corrcoef(labels, predictions)[0, 1] if len(predictions) > 1 else 0
    
    # Accuracy (within tolerance)
    tolerance = 10  # ±10 points
    within_tolerance = np.sum(np.abs(predictions - labels) <= tolerance)
    accuracy = within_tolerance / len(predictions)
    
    # Precision/Recall for high scores (classification metrics)
    high_threshold = 80
    pred_high = (predictions >= high_threshold).astype(int)
    true_high = (labels >= high_threshold).astype(int)
    
    precision_high = precision_score(true_high, pred_high, zero_division=0) if len(np.unique(pred_high)) > 1 else 0.0
    recall_high = recall_score(true_high, pred_high, zero_division=0) if len(np.unique(true_high)) > 1 else 0.0
    f1_high = f1_score(true_high, pred_high, zero_division=0) if (precision_high + recall_high) > 0 else 0.0
    
    # Per-criterion metrics (if available)
    criterion_metrics = {}
    # Try to extract per-criterion scores
    with open(dataset_path, 'r', encoding='utf-8') as f:
        criterion_predictions = {}
        criterion_ground_truths = {}
        
        for line_num, line in enumerate(f, 1):
            if line_num > len(predictions):
                break
            
            try:
                record = json.loads(line.strip())
                output_data = record.get("output", {})
                scores = output_data.get("scores", {})
                macro_scores = scores.get("macro_scores", [])
                micro_scores = scores.get("micro_scores", [])
                
                # For now, we'll aggregate per-criterion from ground truth
                # (Note: Model predicts total score, not per-criterion)
                for crit_data in macro_scores + micro_scores:
                    criterion_name = crit_data.get("criteria", "")
                    gt_score = crit_data.get("score", 0.0)
                    
                    if gt_score > 10:
                        gt_score_normalized = gt_score
                    else:
                        gt_score_normalized = gt_score * 10
                    
                    if criterion_name not in criterion_ground_truths:
                        criterion_ground_truths[criterion_name] = []
                    criterion_ground_truths[criterion_name].append(gt_score_normalized)
            except:
                continue
    
    results = {
        'sample_size': len(predictions),
        'overall_metrics': {
            'accuracy': round(accuracy, 3),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2_score': round(r2_score, 3),
            'correlation': round(correlation, 3),
            'mean_predicted': round(float(np.mean(predictions)), 2),
            'mean_ground_truth': round(float(np.mean(labels)), 2),
            'std_predicted': round(float(np.std(predictions)), 2),
            'std_ground_truth': round(float(np.std(labels)), 2)
        },
        'classification_metrics': {
            'precision_high_scores': round(precision_high, 3),
            'recall_high_scores': round(recall_high, 3),
            'f1_high_scores': round(f1_high, 3),
            'high_score_threshold': high_threshold
        },
        'criterion_metrics': criterion_metrics
    }
    
    return results


def display_evaluation_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted way."""
    if results is None:
        print("❌ No results to display")
        return
    
    print("\n" + "="*70)
    print("TRAINED SLM MODEL EVALUATION RESULTS")
    print("="*70)
    
    overall = results.get('overall_metrics', {})
    classification = results.get('classification_metrics', {})
    
    print(f"\n📊 OVERALL METRICS ({results.get('sample_size', 0)} samples):")
    print(f"  • Accuracy (±10 points): {overall.get('accuracy', 0):.1%}")
    print(f"  • Mean Absolute Error (MAE): {overall.get('mae', 0):.2f} points")
    print(f"  • Root Mean Squared Error (RMSE): {overall.get('rmse', 0):.2f} points")
    print(f"  • R² Score: {overall.get('r2_score', 0):.3f}")
    print(f"  • Correlation: {overall.get('correlation', 0):.3f}")
    print(f"  • Mean Predicted: {overall.get('mean_predicted', 0):.2f}/100")
    print(f"  • Mean Ground Truth: {overall.get('mean_ground_truth', 0):.2f}/100")
    
    print(f"\n🎯 HIGH SCORE PREDICTION (≥{classification.get('high_score_threshold', 80)} points):")
    print(f"  • Precision: {classification.get('precision_high_scores', 0):.1%}")
    print(f"  • Recall: {classification.get('recall_high_scores', 0):.1%}")
    print(f"  • F1 Score: {classification.get('f1_high_scores', 0):.3f}")


def main():
    """Main evaluation function."""
    # Evaluate trained model
    results = evaluate_trained_model(
        model_path="models/trained_slm",
        dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
        test_size=200  # Evaluate on 200 samples for speed
    )
    
    if results:
        display_evaluation_results(results)
        
        # Save results
        output_path = "models/trained_slm/evaluation_results.json"
        os.makedirs("models/trained_slm", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")
    else:
        print("❌ Evaluation failed")


if __name__ == "__main__":
    main()

