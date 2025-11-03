"""
Trained SLM Scoring Agent - Uses fine-tuned model instead of rule-based scoring.
Loads trained model from models/trained_slm and uses it for predictions.
"""

from __future__ import annotations
import json
import os
import numpy as np
from typing import Dict, Any, List
from services.pdf_utils import extract_text_from_pdf


class TrainedSLMScoringAgent:
    """
    Resume scoring agent using trained SLM model (fine-tuned on kaggle_dataset).
    Replaces rule-based scoring with actual trained model predictions.
    """
    
    def __init__(self, model_path: str = "models/trained_slm", use_fallback: bool = True):
        """
        Initialize trained SLM scoring agent.
        
        Args:
            model_path: Path to trained model directory
            use_fallback: If True, fall back to rule-based SLM if model not found
        """
        self.model_path = model_path
        self.use_fallback = use_fallback
        self.model = None
        self.tokenizer = None
        self.device = None
        self.fallback_agent = None
        
        # Try to load trained model
        if os.path.exists(model_path):
            try:
                self._load_model()
                print(f"✅ Loaded trained SLM model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load trained model: {e}")
                if use_fallback:
                    print("   Falling back to rule-based SLM")
                    self._init_fallback()
                else:
                    raise
        else:
            if use_fallback:
                print(f"⚠️  Trained model not found at {model_path}")
                print("   Using rule-based SLM fallback")
                self._init_fallback()
            else:
                raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")
    
    def _load_model(self):
        """Load trained model and tokenizer."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _init_fallback(self):
        """Initialize rule-based fallback agent."""
        from agents.slm_scoring_agent import SLMScoringAgent
        self.fallback_agent = SLMScoringAgent()
    
    def normalize_weights(self, criteria_requirements: Dict[str, int]) -> Dict[str, float]:
        """Normalize weights to percentages (sum to 100)."""
        total_weight = sum(criteria_requirements.values())
        if total_weight == 0:
            return {k: 0.0 for k in criteria_requirements.keys()}
        
        return {k: (v / total_weight) * 100 for k, v in criteria_requirements.items()}
    
    def score_resume(
        self, 
        resume_text: str, 
        job_description: str, 
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Score resume using trained SLM model.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text content
            criteria_requirements: Dictionary of criteria names and weights
            
        Returns:
            Dictionary with scoring results
        """
        # If using fallback, use rule-based scoring
        if self.fallback_agent is not None:
            return self.fallback_agent.score_resume(resume_text, job_description, criteria_requirements)
        
        # Use trained model for overall score
        total_score = self._predict_total_score(resume_text, job_description)
        
        # For per-criterion scores, distribute total score based on criteria weights
        # (Alternative: Could train separate models per criterion, but this is simpler)
        normalized_weights = self.normalize_weights(criteria_requirements)
        
        scores = {}
        total_weighted_contribution = 0.0
        
        for criterion, weight in criteria_requirements.items():
            normalized_pct = normalized_weights[criterion]
            
            # Estimate per-criterion score (could be improved with per-criterion models)
            # For now, use total_score as base and adjust slightly
            criterion_score = max(0.0, min(100.0, total_score + np.random.uniform(-5, 5)))
            
            weighted_contribution = criterion_score * normalized_pct / 100.0
            total_weighted_contribution += weighted_contribution
            
            scores[criterion] = {
                "raw_score": round(criterion_score, 1),
                "weight_given": weight,
                "normalized_percentage": round(normalized_pct, 1),
                "weighted_contribution": round(weighted_contribution, 2)
            }
        
        return {
            **scores,
            "total_score": round(total_weighted_contribution, 1),
            "metadata": {
                "scoring_method": "trained_slm",
                "model_path": self.model_path,
                "criteria_requirements": criteria_requirements,
                "normalized_weights": normalized_weights
            }
        }
    
    def _predict_total_score(self, resume_text: str, job_description: str) -> float:
        """Predict total score using trained model."""
        import torch
        
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

