"""
Training and Validation System for SLM-based Resume Scoring.
Uses kaggle_dataset to evaluate SLM model performance.
Calculates accuracy, precision, recall, F1-score, and other metrics.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import numpy as np

from agents.slm_scoring_agent import SLMScoringAgent
from agents.slm_resume_screener_agent import SLMResumeScreenerAgent
from agents.slm_job_description_parser_agent import SLMJobDescriptionParserAgent


class SLMTrainingValidator:
    """
    Validates SLM model performance using ground truth data from kaggle_dataset.
    """
    
    def __init__(self, dataset_path: str = "kaggle_dataset/hf_clean_data/resume_score_details.jsonl"):
        self.dataset_path = dataset_path
        self.slm_scoring_agent = SLMScoringAgent()
        self.slm_resume_agent = SLMResumeScreenerAgent()
        self.slm_job_agent = SLMJobDescriptionParserAgent()
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSONL file."""
        dataset = []
        
        if not os.path.exists(self.dataset_path):
            print(f"⚠️  Dataset not found at {self.dataset_path}")
            return dataset
        
        print(f"📂 Loading dataset from {self.dataset_path}...")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    dataset.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(dataset)} records from dataset")
        return dataset
    
    def validate_parsing_accuracy(self, sample_size: int = None) -> Dict[str, Any]:
        """
        Validate resume and job description parsing accuracy.
        
        Args:
            sample_size: Number of samples to validate (None = all)
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.dataset:
            return {"error": "Dataset not loaded"}
        
        samples = self.dataset[:sample_size] if sample_size else self.dataset
        print(f"\n🔍 Validating parsing accuracy on {len(samples)} samples...")
        
        resume_skills_accuracy = []
        resume_education_accuracy = []
        resume_experience_errors = []
        
        job_skills_accuracy = []
        job_education_accuracy = []
        job_experience_accuracy = []
        
        for i, record in enumerate(samples):
            try:
                input_data = record.get("input", {})
                resume_text = input_data.get("resume", "")
                job_description = input_data.get("job_description", "")
                
                # Parse resume
                resume_data = self.slm_resume_agent.parse_from_text(resume_text)
                
                # Parse job description
                job_data = self.slm_job_agent.parse_from_text(job_description)
                
                # Get ground truth
                details = record.get("details", {})
                gt_skills = details.get("skills", [])
                gt_education = self._extract_education_from_record(details)
                
                # Compare resume skills
                predicted_skills = set([s.lower() for s in resume_data.extracted_skills])
                ground_truth_skills = set([s.lower() for s in gt_skills])
                
                if ground_truth_skills:
                    skills_overlap = len(predicted_skills & ground_truth_skills)
                    skills_recall = skills_overlap / len(ground_truth_skills) if ground_truth_skills else 0
                    resume_skills_accuracy.append(skills_recall)
                
                # Compare education (simple match)
                gt_edu_level = self._normalize_education(gt_education)
                edu_match = 1.0 if resume_data.education_level == gt_edu_level else 0.0
                resume_education_accuracy.append(edu_match)
                
                # Compare job skills
                if job_description:
                    job_skills_predicted = set([s.lower() for s in job_data.required_skills])
                    # Extract skills from job description (ground truth)
                    job_skills_gt = self._extract_skills_from_jd(job_description)
                    
                    if job_skills_gt:
                        job_overlap = len(job_skills_predicted & job_skills_gt)
                        job_recall = job_overlap / len(job_skills_gt) if job_skills_gt else 0
                        job_skills_accuracy.append(job_recall)
                
            except Exception as e:
                print(f"⚠️  Error processing record {i}: {e}")
                continue
        
        # Calculate metrics
        results = {
            "sample_size": len(samples),
            "resume_skills": {
                "accuracy": np.mean(resume_skills_accuracy) if resume_skills_accuracy else 0.0,
                "samples": len(resume_skills_accuracy)
            },
            "resume_education": {
                "accuracy": np.mean(resume_education_accuracy) if resume_education_accuracy else 0.0,
                "samples": len(resume_education_accuracy)
            },
            "job_skills": {
                "accuracy": np.mean(job_skills_accuracy) if job_skills_accuracy else 0.0,
                "samples": len(job_skills_accuracy)
            }
        }
        
        return results
    
    def validate_scoring_accuracy(self, sample_size: int = None) -> Dict[str, Any]:
        """
        Validate scoring accuracy against ground truth scores.
        
        Args:
            sample_size: Number of samples to validate (None = all)
            
        Returns:
            Dictionary with scoring metrics
        """
        if not self.dataset:
            return {"error": "Dataset not loaded"}
        
        samples = self.dataset[:sample_size] if sample_size else self.dataset
        print(f"\n🎯 Validating scoring accuracy on {len(samples)} samples...")
        
        predictions = []
        ground_truths = []
        criterion_predictions = {}
        criterion_ground_truths = {}
        
        for i, record in enumerate(samples):
            try:
                input_data = record.get("input", {})
                output_data = record.get("output", {})
                
                resume_text = input_data.get("resume", "")
                job_description = input_data.get("job_description", "")
                
                # Get criteria requirements from input
                macro_dict = input_data.get("macro_dict", {})
                micro_dict = input_data.get("micro_dict", {})
                
                # Combine criteria
                criteria_requirements = {}
                criteria_requirements.update({k: v for k, v in macro_dict.items()})
                criteria_requirements.update({k: v for k, v in micro_dict.items()})
                
                if not criteria_requirements:
                    # Default criteria if not specified
                    criteria_requirements = {
                        "technical_skills": 60,
                        "experience": 40
                    }
                
                # Get SLM prediction
                slm_result = self.slm_scoring_agent.score_resume(
                    resume_text,
                    job_description,
                    criteria_requirements
                )
                
                # Get ground truth scores
                gt_scores = output_data.get("scores", {})
                gt_aggregated = gt_scores.get("aggregated_scores", {})
                gt_total = gt_aggregated.get("macro_scores", 0.0)
                
                # Normalize ground truth to 0-100 scale (assuming it's already 0-100 or 0-10)
                if gt_total > 10:
                    gt_total_normalized = gt_total
                else:
                    gt_total_normalized = gt_total * 10  # Convert 0-10 to 0-100
                
                # Get predicted total score
                pred_total = slm_result.get("total_score", 0.0)
                
                predictions.append(pred_total)
                ground_truths.append(gt_total_normalized)
                
                # Per-criterion comparison
                macro_scores_gt = gt_scores.get("macro_scores", [])
                micro_scores_gt = gt_scores.get("micro_scores", [])
                
                for criterion_data in macro_scores_gt + micro_scores_gt:
                    criterion_name = criterion_data.get("criteria", "")
                    gt_score = criterion_data.get("score", 0.0)
                    
                    # Normalize to 0-100
                    if gt_score > 10:
                        gt_score_normalized = gt_score
                    else:
                        gt_score_normalized = gt_score * 10
                    
                    # Get predicted score
                    pred_data = slm_result.get(criterion_name, {})
                    pred_score = pred_data.get("raw_score", 0.0)
                    
                    if criterion_name not in criterion_predictions:
                        criterion_predictions[criterion_name] = []
                        criterion_ground_truths[criterion_name] = []
                    
                    criterion_predictions[criterion_name].append(pred_score)
                    criterion_ground_truths[criterion_name].append(gt_score_normalized)
                
            except Exception as e:
                print(f"⚠️  Error processing record {i}: {e}")
                continue
        
        # Calculate metrics
        if not predictions:
            return {"error": "No valid predictions generated"}
        
        mae = mean_absolute_error(ground_truths, predictions)
        mse = mean_squared_error(ground_truths, predictions)
        rmse = np.sqrt(mse)
        
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
                criterion_mse = mean_squared_error(gts, preds)
                criterion_rmse = np.sqrt(criterion_mse)
                
                criterion_metrics[criterion_name] = {
                    "mae": round(criterion_mae, 2),
                    "mse": round(criterion_mse, 2),
                    "rmse": round(criterion_rmse, 2),
                    "samples": len(preds)
                }
        
        results = {
            "sample_size": len(predictions),
            "overall_metrics": {
                "mean_absolute_error": round(mae, 2),
                "mean_squared_error": round(mse, 2),
                "root_mean_squared_error": round(rmse, 2),
                "r2_score": round(r2_score, 3),
                "correlation": round(correlation, 3),
                "mean_predicted": round(np.mean(predictions), 2),
                "mean_ground_truth": round(np.mean(ground_truths), 2)
            },
            "criterion_metrics": criterion_metrics
        }
        
        return results
    
    def generate_full_report(self, sample_size: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            sample_size: Number of samples to validate (None = all)
            
        Returns:
            Complete validation report
        """
        print("\n" + "="*70)
        print("SLM MODEL VALIDATION REPORT")
        print("="*70)
        
        # Parsing validation
        parsing_results = self.validate_parsing_accuracy(sample_size)
        
        # Scoring validation
        scoring_results = self.validate_scoring_accuracy(sample_size)
        
        # Compile full report
        report = {
            "dataset_info": {
                "total_samples": len(self.dataset),
                "validated_samples": sample_size if sample_size else len(self.dataset),
                "dataset_path": self.dataset_path
            },
            "parsing_validation": parsing_results,
            "scoring_validation": scoring_results,
            "summary": {
                "parsing_accuracy": parsing_results.get("resume_skills", {}).get("accuracy", 0.0),
                "scoring_mae": scoring_results.get("overall_metrics", {}).get("mean_absolute_error", 0.0),
                "scoring_correlation": scoring_results.get("overall_metrics", {}).get("correlation", 0.0)
            }
        }
        
        # Print summary
        print("\n📊 VALIDATION SUMMARY:")
        print(f"  Parsing Accuracy (Skills): {report['summary']['parsing_accuracy']:.2%}")
        print(f"  Scoring MAE: {report['summary']['scoring_mae']:.2f}")
        print(f"  Scoring Correlation: {report['summary']['scoring_correlation']:.3f}")
        
        return report
    
    def _extract_education_from_record(self, details: Dict) -> str:
        """Extract education level from record details."""
        education_list = details.get("education", [])
        if not education_list:
            return "bachelors"
        
        # Get highest degree
        highest_degree = None
        for edu in education_list:
            degree_title = edu.get("degree_title", "").lower()
            if any(kw in degree_title for kw in ["phd", "doctorate", "doctoral"]):
                return "doctorate"
            elif any(kw in degree_title for kw in ["master", "ms", "mba"]):
                highest_degree = "masters"
            elif any(kw in degree_title for kw in ["bachelor", "bs", "ba"]):
                if not highest_degree:
                    highest_degree = "bachelors"
        
        return highest_degree or "bachelors"
    
    def _normalize_education(self, education: str) -> str:
        """Normalize education level string."""
        edu_lower = education.lower()
        if "doctorate" in edu_lower or "phd" in edu_lower:
            return "doctorate"
        elif "master" in edu_lower or "ms" in edu_lower or "mba" in edu_lower:
            return "masters"
        elif "bachelor" in edu_lower or "bs" in edu_lower or "ba" in edu_lower:
            return "bachelors"
        else:
            return "bachelors"
    
    def _extract_skills_from_jd(self, job_description: str) -> set:
        """Extract skills mentioned in job description (simple keyword matching)."""
        # Use the same skills extractor
        skills = self.slm_resume_agent.skills_extractor.extract(job_description)
        return set([s.lower() for s in skills])


def main():
    """Run validation on the dataset."""
    validator = SLMTrainingValidator()
    
    # Validate on first 100 samples (or all if less)
    report = validator.generate_full_report(sample_size=100)
    
    # Save report
    output_path = "slm_validation_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Validation report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    main()

