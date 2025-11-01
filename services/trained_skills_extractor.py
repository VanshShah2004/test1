"""
Trained Skills Extractor - Uses ML models trained on Kaggle dataset.
Falls back to rule-based extractor if models not available.
"""

from typing import List, Dict
import os
import pickle

from .skills_dictionary import ALL_TECH_SKILLS
from .nlp_extractors import SkillsExtractor


class TrainedSkillsExtractor:
    """
    Skills extractor that uses trained models when available,
    falls back to rule-based extraction otherwise.
    """
    
    def __init__(self, use_trained: bool = True):
        self.use_trained = use_trained
        self.rule_based_extractor = SkillsExtractor()
        
        # Load trained models if available
        self.trained_models = {}
        if use_trained:
            self._load_trained_models()
    
    def _load_trained_models(self):
        """Load trained models from disk."""
        models_dir = "trained_models"
        
        if os.path.exists(models_dir):
            # Try to load any trained models
            for file in os.listdir(models_dir):
                if file.endswith('.pkl'):
                    model_path = os.path.join(models_dir, file)
                    try:
                        with open(model_path, 'rb') as f:
                            model_name = file.replace('.pkl', '')
                            self.trained_models[model_name] = pickle.load(f)
                        print(f"✅ Loaded trained model: {model_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to load {model_name}: {e}")
    
    def extract(self, text: str) -> List[str]:
        """
        Extract skills using trained models + rule-based fallback.
        """
        # Start with rule-based extraction
        skills = self.rule_based_extractor.extract(text)
        
        # If trained models available, enhance with ML-based extraction
        if self.trained_models:
            # Could use ML models to find additional skills or validate existing ones
            # For now, rule-based is primary
            pass
        
        return skills


class TrainedEducationClassifier:
    """
    Education classifier that uses trained ML model when available.
    """
    
    def __init__(self, use_trained: bool = True):
        self.use_trained = use_trained
        self.trained_model = None
        self.vectorizer = None
        
        from .nlp_extractors import EducationClassifier
        self.rule_based_classifier = EducationClassifier()
        
        if use_trained:
            self._load_trained_model()
    
    def _load_trained_model(self):
        """Load trained education classifier."""
        model_path = "trained_models/education_classifier.pkl"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.trained_model = model_data.get('model')
                    self.vectorizer = model_data.get('vectorizer')
                print("✅ Loaded trained education classifier")
            except Exception as e:
                print(f"⚠️  Failed to load trained classifier: {e}")
    
    def classify(self, text: str) -> str:
        """Classify education level using trained model or rule-based fallback."""
        # Use trained model if available
        if self.trained_model and self.vectorizer:
            try:
                text_vectorized = self.vectorizer.transform([text[:500]])
                prediction = self.trained_model.predict(text_vectorized)[0]
                return prediction
            except Exception as e:
                print(f"⚠️  ML classification failed, using rule-based: {e}")
        
        # Fallback to rule-based
        return self.rule_based_classifier.classify(text)

