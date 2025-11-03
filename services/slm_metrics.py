"""
SLM Metrics Calculator - Calculates accuracy and precision for SLM scoring.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


class SLMMetricsCalculator:
    """Calculate accuracy and precision metrics for SLM scoring against ground truth."""
    
    def __init__(self, dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl"):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset() if os.path.exists(dataset_path) else []
    
    def _load_dataset(self) -> List[Dict]:
        """Load ground truth dataset."""
        dataset = []
        if not os.path.exists(self.dataset_path):
            return dataset
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        dataset.append(data)
                    except:
                        continue
        except Exception as e:
            print(f"⚠️  Warning: Could not load dataset for metrics: {e}")
        
        return dataset
    
    def find_matching_record(self, resume_text: str, job_description: str) -> Optional[Dict]:
        """
        Find matching record in dataset by comparing text similarity.
        Returns ground truth record if found.
        """
        if not self.dataset:
            return None
        
        # Simple matching: check if resume text and job description match
        resume_text_lower = resume_text.lower()[:500]  # Use first 500 chars for matching
        job_desc_lower = job_description.lower()[:500]
        
        best_match = None
        best_score = 0
        
        for record in self.dataset:
            input_data = record.get("input", {})
            record_resume = input_data.get("resume", "").lower()[:500]
            record_jd = input_data.get("job_description", "").lower()[:500]
            
            # Calculate simple similarity (word overlap)
            resume_words = set(resume_text_lower.split())
            record_resume_words = set(record_resume.split())
            
            # Remove very common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            resume_words = resume_words - common_words
            record_resume_words = record_resume_words - common_words
            
            resume_match = len(resume_words & record_resume_words) / max(1, min(len(resume_words), len(record_resume_words)))
            
            jd_words = set(job_desc_lower.split())
            record_jd_words = set(record_jd.split())
            jd_words = jd_words - common_words
            record_jd_words = record_jd_words - common_words
            jd_match = len(jd_words & record_jd_words) / max(1, min(len(jd_words), len(record_jd_words)))
            
            # Combined score (weighted: resume more important)
            similarity = 0.7 * resume_match + 0.3 * jd_match
            
            # Lower threshold to 15% for better matching
            if similarity > best_score and similarity > 0.15:
                best_score = similarity
                best_match = record
        
        return best_match
    
    def calculate_accuracy_precision(
        self,
        slm_predictions: Dict[str, Dict[str, Any]],
        resume_texts: Dict[str, str],
        job_description_text: str,
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Calculate accuracy and precision metrics for SLM predictions.
        
        Args:
            slm_predictions: Dict mapping resume_path -> SLM scoring result
            resume_texts: Dict mapping resume_path -> resume text
            job_description_text: Job description text
            criteria_requirements: Criteria dictionary
            
        Returns:
            Dictionary with accuracy and precision metrics
        """
        if not self.dataset:
            return {
                "available": False,
                "message": "Ground truth dataset not available"
            }
        
        predictions = []
        ground_truths = []
        criterion_predictions = {}
        criterion_ground_truths = {}
        matched_count = 0
        
        # For each resume, try to find matching ground truth
        for resume_path, slm_result in slm_predictions.items():
            if slm_result is None:
                continue
            
            resume_text = resume_texts.get(resume_path, "")
            if not resume_text:
                continue
            
            # Find matching ground truth record
            gt_record = self.find_matching_record(resume_text, job_description_text)
            
            if gt_record is None:
                continue  # No ground truth match found
            
            matched_count += 1
            
            # Extract ground truth scores
            output_data = gt_record.get("output", {})
            gt_scores = output_data.get("scores", {})
            gt_aggregated = gt_scores.get("aggregated_scores", {})
            gt_total = gt_aggregated.get("macro_scores", 0.0)
            
            # Normalize ground truth to 0-100 scale
            if gt_total > 10:
                gt_total_normalized = gt_total
            else:
                gt_total_normalized = gt_total * 10  # Convert 0-10 to 0-100
            
            # Get SLM predicted total score
            pred_total = slm_result.get("total_score", 0.0)
            
            predictions.append(pred_total)
            ground_truths.append(gt_total_normalized)
            
            # Per-criterion comparison
            macro_scores_gt = gt_scores.get("macro_scores", [])
            micro_scores_gt = gt_scores.get("micro_scores", [])
            
            for criterion_data in macro_scores_gt + micro_scores_gt:
                criterion_name = criterion_data.get("criteria", "").lower()
                gt_score = criterion_data.get("score", 0.0)
                
                # Normalize to 0-100
                if gt_score > 10:
                    gt_score_normalized = gt_score
                else:
                    gt_score_normalized = gt_score * 10
                
                # Try to match criterion name with SLM result
                pred_data = None
                for key in slm_result.keys():
                    if key.lower() in criterion_name or criterion_name in key.lower():
                        if isinstance(slm_result[key], dict):
                            pred_data = slm_result[key]
                            break
                
                if pred_data:
                    pred_score = pred_data.get("raw_score", 0.0)
                    
                    if criterion_name not in criterion_predictions:
                        criterion_predictions[criterion_name] = []
                        criterion_ground_truths[criterion_name] = []
                    
                    criterion_predictions[criterion_name].append(pred_score)
                    criterion_ground_truths[criterion_name].append(gt_score_normalized)
        
        if len(predictions) == 0:
            return {
                "available": True,
                "matched_samples": 0,
                "total_samples": len(slm_predictions),
                "message": "No matching ground truth records found"
            }
        
        # Calculate overall metrics
        mae = mean_absolute_error(ground_truths, predictions)
        mse = mean_squared_error(ground_truths, predictions)
        rmse = np.sqrt(mse)
        
        # Accuracy (within tolerance bands)
        tolerance = 10  # ±10 points
        within_tolerance = sum(1 for p, g in zip(predictions, ground_truths) if abs(p - g) <= tolerance)
        accuracy = within_tolerance / len(predictions) if predictions else 0
        
        # Precision for high scores (predictions > 80)
        high_score_threshold = 80
        pred_high = [p for p in predictions if p >= high_score_threshold]
        gt_high = [g for g in ground_truths if g >= high_score_threshold]
        
        if pred_high:
            # Precision: Of predicted high scores, how many were actually high?
            true_positives = sum(1 for i, p in enumerate(predictions) 
                               if p >= high_score_threshold and ground_truths[i] >= high_score_threshold)
            precision_high = true_positives / len(pred_high) if pred_high else 0
        else:
            precision_high = None
        
        # Recall for high scores
        if gt_high:
            true_positives = sum(1 for i, g in enumerate(ground_truths)
                               if g >= high_score_threshold and predictions[i] >= high_score_threshold)
            recall_high = true_positives / len(gt_high) if gt_high else 0
        else:
            recall_high = None
        
        # F1 score
        if precision_high is not None and recall_high is not None:
            if (precision_high + recall_high) > 0:
                f1_high = 2 * (precision_high * recall_high) / (precision_high + recall_high)
            else:
                f1_high = 0.0
        else:
            f1_high = None
        
        # R² score
        ss_res = np.sum((np.array(ground_truths) - np.array(predictions)) ** 2)
        ss_tot = np.sum((np.array(ground_truths) - np.mean(ground_truths)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(ground_truths, predictions)[0, 1] if len(predictions) > 1 else 0
        
        # Per-criterion metrics
        criterion_metrics = {}
        for criterion_name in criterion_predictions:
            preds = criterion_predictions[criterion_name]
            gts = criterion_ground_truths[criterion_name]
            
            if len(preds) > 0:
                criterion_mae = mean_absolute_error(gts, preds)
                criterion_rmse = np.sqrt(mean_squared_error(gts, preds))
                
                # Criterion accuracy (within tolerance)
                criterion_within_tol = sum(1 for p, g in zip(preds, gts) if abs(p - g) <= tolerance)
                criterion_accuracy = criterion_within_tol / len(preds)
                
                criterion_metrics[criterion_name] = {
                    "mae": round(criterion_mae, 2),
                    "rmse": round(criterion_rmse, 2),
                    "accuracy": round(criterion_accuracy, 3),
                    "samples": len(preds)
                }
        
        return {
            "available": True,
            "matched_samples": matched_count,
            "total_samples": len(slm_predictions),
            "overall_metrics": {
                "accuracy": round(accuracy, 3),  # Accuracy within ±10 points
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2_score": round(r2_score, 3),
                "correlation": round(correlation, 3),
                "precision_high_scores": round(precision_high, 3) if precision_high is not None else None,
                "recall_high_scores": round(recall_high, 3) if recall_high is not None else None,
                "f1_high_scores": round(f1_high, 3) if f1_high is not None else None,
                "mean_predicted": round(np.mean(predictions), 2),
                "mean_ground_truth": round(np.mean(ground_truths), 2)
            },
            "criterion_metrics": criterion_metrics
        }

