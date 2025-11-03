"""
SLM-based Resume Scoring Agent (NLP-based, NO LLMs).
Uses rule-based scoring algorithms instead of language models.
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List
from services.pdf_utils import extract_text_from_pdf
from services.nlp_extractors import SkillsExtractor, ExperienceExtractor, EducationClassifier
from services.llm import LLMService


class SLMScoringAgent:
    """
    Resume scoring agent using pure NLP techniques (NO LLMs for scoring).
    Calculates scores based on rule-based matching and quantitative analysis.
    """
    
    def __init__(self):
        self.skills_extractor = SkillsExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.education_classifier = EducationClassifier()
    
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
        Score resume against job description using rule-based NLP matching.
        
        Args:
            resume_text: Resume text content
            job_description: Job description text content
            criteria_requirements: Dictionary of criteria names and weights
            
        Returns:
            Dictionary with scoring results matching StructuredScoringAgent format
        """
        # Normalize weights
        normalized_weights = self.normalize_weights(criteria_requirements)
        
        # Extract resume data
        resume_skills = self.skills_extractor.extract(resume_text)
        resume_experience = self.experience_extractor.extract_experience(resume_text)
        resume_education = self.education_classifier.classify(resume_text)
        
        # Extract job requirements
        job_skills_extractor = SkillsExtractor()
        job_skills = job_skills_extractor.extract(job_description)
        
        # Extract minimum experience requirement (using helper method)
        job_experience = self._extract_min_experience_from_text(job_description)
        job_education = self.education_classifier.classify(job_description)
        
        # Calculate scores for each criterion
        scores = {}
        total_weighted_contribution = 0.0
        
        for criterion, weight in criteria_requirements.items():
            normalized_pct = normalized_weights[criterion]
            
            # Calculate raw score based on criterion type
            raw_score = self._calculate_criterion_score(
                criterion,
                resume_skills, resume_experience, resume_education,
                job_skills, job_experience, job_education,
                resume_text, job_description
            )
            
            # Calculate weighted contribution
            weighted_contribution = raw_score * normalized_pct / 100.0
            total_weighted_contribution += weighted_contribution
            
            scores[criterion] = {
                "raw_score": round(raw_score, 1),
                "weight_given": weight,
                "normalized_percentage": round(normalized_pct, 1),
                "weighted_contribution": round(weighted_contribution, 2)
            }
        
        return {
            **scores,
            "total_score": round(total_weighted_contribution, 1),
            "metadata": {
                "scoring_method": "slm_nlp",
                "criteria_requirements": criteria_requirements,
                "normalized_weights": normalized_weights
            }
        }
    
    def _calculate_criterion_score(
        self,
        criterion: str,
        resume_skills: List[str],
        resume_experience: float,
        resume_education: str,
        job_skills: List[str],
        job_experience: int,
        job_education: str,
        resume_text: str,
        job_description: str
    ) -> float:
        """
        Calculate score for a specific criterion (0-100 scale).
        Uses rule-based matching algorithms.
        """
        criterion_lower = criterion.lower()
        
        # Technical skills scoring
        if 'skill' in criterion_lower or 'technical' in criterion_lower:
            return self._score_skills_match(resume_skills, job_skills)
        
        # Experience scoring
        elif 'experience' in criterion_lower or 'exp' in criterion_lower:
            return self._score_experience_match(resume_experience, job_experience)
        
        # Education scoring
        elif 'education' in criterion_lower or 'edu' in criterion_lower:
            return self._score_education_match(resume_education, job_education)
        
        # Presentation scoring
        elif 'presentation' in criterion_lower or 'format' in criterion_lower:
            return self._score_presentation(resume_text)
        
        # Certifications scoring
        elif 'certification' in criterion_lower or 'cert' in criterion_lower:
            return self._score_certifications(resume_text, job_description)
        
        # Projects scoring
        elif 'project' in criterion_lower:
            return self._score_projects(resume_text)
        
        # Soft skills scoring
        elif 'soft' in criterion_lower:
            return self._score_soft_skills(resume_text)
        
        # Career progression scoring
        elif 'progression' in criterion_lower or 'career' in criterion_lower:
            return self._score_career_progression(resume_text)
        
        # Marketability scoring
        elif 'marketability' in criterion_lower or 'marketable' in criterion_lower:
            return self._score_marketability(resume_text, resume_skills, resume_experience)
        
        # Industry knowledge scoring
        elif 'industry' in criterion_lower:
            return self._score_industry_knowledge(resume_text, job_description)
        
        # Default: general match score
        else:
            return self._score_general_match(
                resume_skills, resume_experience, resume_education,
                job_skills, job_experience, job_education
            )
    
    def _score_skills_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Score based on skills overlap."""
        if not job_skills:
            return 50.0  # Neutral if no job skills specified
        
        resume_skills_set = set([s.lower() for s in resume_skills])
        job_skills_set = set([s.lower() for s in job_skills])
        
        # Calculate match percentage
        if job_skills_set:
            match_count = len(resume_skills_set & job_skills_set)
            match_ratio = match_count / len(job_skills_set)
            return min(100.0, match_ratio * 100)
        
        return 0.0
    
    def _score_experience_match(self, resume_exp: float, job_req_exp: int) -> float:
        """Score based on experience match."""
        if job_req_exp == 0:
            return 70.0  # Neutral if no requirement
        
        if resume_exp >= job_req_exp:
            # More or equal experience = high score
            excess = resume_exp - job_req_exp
            if excess >= 5:
                return 100.0
            elif excess >= 2:
                return 95.0
            else:
                return 90.0
        else:
            # Less experience = proportional score
            ratio = resume_exp / job_req_exp if job_req_exp > 0 else 0
            return max(20.0, ratio * 80)  # Minimum 20, max 80 for less experience
    
    def _score_education_match(self, resume_edu: str, job_edu: str) -> float:
        """Score based on education level match."""
        education_hierarchy = {
            'high_school': 1,
            'bachelors': 2,
            'masters': 3,
            'doctorate': 4
        }
        
        resume_level = education_hierarchy.get(resume_edu, 2)
        job_level = education_hierarchy.get(job_edu, 2)
        
        if resume_level >= job_level:
            return 100.0
        else:
            # Partial credit for lower education
            gap = job_level - resume_level
            return max(40.0, 100.0 - (gap * 20))
    
    def _score_presentation(self, resume_text: str) -> float:
        """Score based on resume presentation/formatting."""
        score = 70.0  # Base score
        
        # Check for sections
        sections = ['experience', 'education', 'skills', 'project']
        found_sections = sum(1 for sec in sections if sec in resume_text.lower())
        score += (found_sections / len(sections)) * 20
        
        # Penalize for poor formatting indicators
        if len(resume_text.split('\n')) < 10:
            score -= 20  # Too short
        if resume_text.count('\n\n\n') > 5:
            score -= 10  # Too many blank lines
        
        return max(0.0, min(100.0, score))
    
    def _score_certifications(self, resume_text: str, job_description: str) -> float:
        """Score based on certifications mentioned."""
        cert_keywords = ['certified', 'certification', 'certificate', 'license', 'credential']
        
        resume_certs = sum(1 for kw in cert_keywords if kw in resume_text.lower())
        job_mentions_certs = any(kw in job_description.lower() for kw in cert_keywords)
        
        if resume_certs > 0:
            return min(100.0, 60.0 + (resume_certs * 10))
        elif not job_mentions_certs:
            return 50.0  # Neutral if not mentioned in JD
        else:
            return 20.0  # Low if required but not present
    
    def _score_projects(self, resume_text: str) -> float:
        """Score based on projects mentioned."""
        project_keywords = ['project', 'portfolio', 'github', 'built', 'developed', 'created']
        
        mentions = sum(1 for kw in project_keywords if kw in resume_text.lower())
        
        if mentions >= 3:
            return 90.0
        elif mentions >= 2:
            return 70.0
        elif mentions >= 1:
            return 50.0
        else:
            return 30.0
    
    def _score_soft_skills(self, resume_text: str) -> float:
        """Score based on soft skills mentioned."""
        soft_skill_keywords = [
            'communication', 'leadership', 'teamwork', 'collaboration',
            'problem solving', 'critical thinking', 'management'
        ]
        
        mentions = sum(1 for kw in soft_skill_keywords if kw in resume_text.lower())
        
        return min(100.0, 40.0 + (mentions * 10))
    
    def _score_career_progression(self, resume_text: str) -> float:
        """Score based on career progression indicators."""
        progression_keywords = ['senior', 'lead', 'manager', 'director', 'promoted', 'advanced']
        
        mentions = sum(1 for kw in progression_keywords if kw in resume_text.lower())
        
        return min(100.0, 50.0 + (mentions * 15))
    
    def _score_marketability(
        self, 
        resume_text: str, 
        resume_skills: List[str],
        resume_experience: float
    ) -> float:
        """Score based on overall marketability."""
        score = 50.0
        
        # Skills diversity
        if len(resume_skills) >= 10:
            score += 20
        elif len(resume_skills) >= 5:
            score += 10
        
        # Experience level
        if resume_experience >= 5:
            score += 20
        elif resume_experience >= 2:
            score += 10
        
        # Quantified achievements
        if any(char.isdigit() for char in resume_text):
            score += 10
        
        return min(100.0, score)
    
    def _score_industry_knowledge(self, resume_text: str, job_description: str) -> float:
        """Score based on industry-specific knowledge."""
        # Simple keyword overlap for industry terms
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        # Common industry terms (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        resume_words -= common_words
        job_words -= common_words
        
        if len(job_words) > 0:
            overlap = len(resume_words & job_words)
            ratio = overlap / len(job_words)
            return min(100.0, ratio * 100)
        
        return 50.0
    
    def _score_general_match(
        self,
        resume_skills: List[str],
        resume_experience: float,
        resume_education: str,
        job_skills: List[str],
        job_experience: int,
        job_education: str
    ) -> float:
        """General matching score combining multiple factors."""
        skills_score = self._score_skills_match(resume_skills, job_skills)
        exp_score = self._score_experience_match(resume_experience, job_experience)
        edu_score = self._score_education_match(resume_education, job_education)
        
        # Weighted average
        return (skills_score * 0.5 + exp_score * 0.3 + edu_score * 0.2)


    def _extract_min_experience_from_text(self, text: str) -> int:
        """Helper to extract minimum experience requirement from text."""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
        ]
        
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        return 0

