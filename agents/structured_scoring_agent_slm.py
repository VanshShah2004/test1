"""
SLM-based Structured Scoring Agent using pure NLP (NO LLMs).
Uses rule-based scoring algorithms based on extracted data.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List
from services.pdf_utils import extract_text_from_pdf
from services.nlp_service import NLPService
from common.models import ResumeData, JobCriteria


class StructuredScoringAgentSLM:
    """
    Resume scoring agent using pure NLP techniques (SLM approach).
    Scores resumes based on extracted data matching against job criteria.
    No LLM API calls - deterministic, fast, local processing.
    """
    
    def __init__(self, nlp_service: NLPService | None = None):
        self._nlp = nlp_service or NLPService()
    
    def normalize_weights(self, criteria_requirements: Dict[str, int]) -> Dict[str, float]:
        """Normalize weights to percentages (sum to 100)"""
        total_weight = sum(criteria_requirements.values())
        if total_weight == 0:
            return {k: 0.0 for k in criteria_requirements.keys()}
        
        return {k: (v / total_weight) * 100 for k, v in criteria_requirements.items()}
    
    def score_resume(
        self,
        resume_path: str,
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Score a single resume against job description using NLP-based matching.
        """
        try:
            # Extract text from PDFs
            resume_text = extract_text_from_pdf(resume_path)
            job_description_text = extract_text_from_pdf(job_description_path)
            
            # Parse using NLP
            resume_data = self._nlp.parse_resume(resume_text)
            job_criteria = self._nlp.parse_job_description(job_description_text)
            
            # Score based on criteria
            scores = self._calculate_scores(resume_data, job_criteria, criteria_requirements)
            
            # Normalize weights
            normalized_weights = self.normalize_weights(criteria_requirements)
            
            # Calculate weighted contributions
            result = {}
            total_score = 0.0
            
            for criterion, score_data in scores.items():
                weight = criteria_requirements.get(criterion, 0)
                normalized_pct = normalized_weights.get(criterion, 0.0)
                weighted_contrib = score_data['raw_score'] * normalized_pct / 100.0
                total_score += weighted_contrib
                
                result[criterion] = {
                    'raw_score': score_data['raw_score'],
                    'weight_given': weight,
                    'normalized_percentage': normalized_pct,
                    'weighted_contribution': round(weighted_contrib, 2)
                }
            
            result['total_score'] = round(total_score, 1)
            result['metadata'] = {
                'resume_path': resume_path,
                'job_description_path': job_description_path,
                'criteria_requirements': criteria_requirements,
                'normalized_weights': normalized_weights,
                'scoring_method': 'nlp_slm'
            }
            
            return {
                'success': True,
                'scoring_result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'NLP scoring error: {str(e)}',
                'resume_path': resume_path,
                'job_description_path': job_description_path
            }
    
    def score_resumes_batch(
        self,
        resume_paths: List[str],
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Score multiple resumes in batch using NLP-based matching.
        """
        results = []
        
        for resume_path in resume_paths:
            result = self.score_resume(resume_path, job_description_path, criteria_requirements)
            results.append(result)
        
        return results
    
    def _calculate_scores(
        self,
        resume_data: Dict,
        job_criteria: Dict,
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Dict]:
        """
        Calculate scores for each criterion using rule-based matching.
        """
        scores = {}
        
        # Parse criteria into resume and job data
        resume_skills = set([s.lower() for s in resume_data.get('extracted_skills', [])])
        job_required_skills = set([s.lower() for s in job_criteria.get('required_skills', [])])
        job_preferred_skills = set([s.lower() for s in job_criteria.get('preferred_skills', [])])
        resume_exp = resume_data.get('total_experience_years', 0.0)
        job_min_exp = job_criteria.get('min_experience_years', 0)
        resume_edu = resume_data.get('education_level', 'bachelors')
        job_edu = job_criteria.get('education_level', 'bachelors')
        
        # Score each criterion
        for criterion in criteria_requirements.keys():
            if criterion == 'technical_skills' or criterion == 'skills':
                score = self._score_technical_skills(resume_skills, job_required_skills, job_preferred_skills)
            elif criterion == 'experience':
                score = self._score_experience(resume_exp, job_min_exp)
            elif criterion == 'education':
                score = self._score_education(resume_edu, job_edu)
            elif criterion == 'presentation':
                score = self._score_presentation(resume_data.get('raw_text', ''))
            elif criterion == 'certifications':
                score = self._score_certifications(resume_data.get('raw_text', ''))
            elif criterion == 'projects':
                score = self._score_projects(resume_data.get('raw_text', ''))
            elif criterion == 'soft_skills':
                score = self._score_soft_skills(resume_data.get('raw_text', ''))
            elif criterion == 'industry_knowledge':
                score = self._score_industry_knowledge(resume_data.get('raw_text', ''), job_criteria)
            else:
                # Default score for unknown criteria
                score = {'raw_score': 50, 'reasoning': 'Criterion not implemented'}
            
            scores[criterion] = score
        
        return scores
    
    def _score_technical_skills(
        self,
        resume_skills: set,
        job_required: set,
        job_preferred: set
    ) -> Dict:
        """Score technical skills match."""
        if not job_required and not job_preferred:
            return {'raw_score': 50, 'reasoning': 'No skills specified in job'}
        
        # Required skills match (70% weight)
        required_match = 0.0
        if job_required:
            matched_required = len(resume_skills & job_required)
            required_match = (matched_required / len(job_required)) * 70.0
        
        # Preferred skills match (30% weight)
        preferred_match = 0.0
        if job_preferred:
            matched_preferred = len(resume_skills & job_preferred)
            preferred_match = (matched_preferred / len(job_preferred)) * 30.0
        
        score = required_match + preferred_match
        return {'raw_score': min(100, int(score)), 'reasoning': f'Matched {len(resume_skills & job_required)}/{len(job_required)} required, {len(resume_skills & job_preferred)}/{len(job_preferred)} preferred'}
    
    def _score_experience(self, resume_exp: float, job_min_exp: int) -> Dict:
        """Score experience match."""
        if job_min_exp == 0:
            return {'raw_score': 70, 'reasoning': 'No experience requirement'}
        
        if resume_exp >= job_min_exp:
            # Bonus for exceeding
            excess = resume_exp - job_min_exp
            bonus = min(20, excess * 2)  # Up to 20 points bonus
            score = 80 + bonus
        elif resume_exp > 0:
            # Partial match
            ratio = resume_exp / job_min_exp
            score = 40 + (ratio * 40)  # 40-80 range
        else:
            # No experience
            score = 20
        
        return {'raw_score': min(100, int(score)), 'reasoning': f'{resume_exp} years vs {job_min_exp} required'}
    
    def _score_education(self, resume_edu: str, job_edu: str) -> Dict:
        """Score education match."""
        edu_hierarchy = {'high_school': 1, 'bachelors': 2, 'masters': 3, 'doctorate': 4}
        
        resume_level = edu_hierarchy.get(resume_edu, 2)
        job_level = edu_hierarchy.get(job_edu, 2)
        
        if resume_level >= job_level:
            # Meets or exceeds
            if resume_level > job_level:
                score = 90  # Exceeds
            else:
                score = 85  # Meets exactly
        else:
            # Below requirement
            gap = job_level - resume_level
            score = 70 - (gap * 10)  # Penalty for each level below
        
        return {'raw_score': max(30, min(100, score)), 'reasoning': f'{resume_edu} vs {job_edu} required'}
    
    def _score_presentation(self, text: str) -> Dict:
        """Score resume presentation quality."""
        if not text:
            return {'raw_score': 30, 'reasoning': 'Empty resume'}
        
        score = 70  # Base score
        
        # Check for sections
        sections = ['experience', 'education', 'skills', 'projects']
        found_sections = sum(1 for section in sections if section.lower() in text.lower())
        score += found_sections * 5  # +5 per section found
        
        # Check length (not too short, not too long)
        word_count = len(text.split())
        if 200 <= word_count <= 800:
            score += 5
        elif word_count < 100:
            score -= 15
        
        # Check formatting indicators
        if '\n\n' in text:  # Proper paragraph breaks
            score += 5
        
        return {'raw_score': min(100, max(30, score)), 'reasoning': f'{found_sections}/4 sections found'}
    
    def _score_certifications(self, text: str) -> Dict:
        """Score certifications."""
        cert_keywords = ['certified', 'certification', 'certificate', 'license', 'accreditation']
        text_lower = text.lower()
        
        found_certs = sum(1 for keyword in cert_keywords if keyword in text_lower)
        
        if found_certs > 0:
            score = 60 + min(30, found_certs * 10)
        else:
            score = 30
        
        return {'raw_score': min(100, score), 'reasoning': f'{found_certs} certification mentions'}
    
    def _score_projects(self, text: str) -> Dict:
        """Score projects."""
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented', 'designed']
        text_lower = text.lower()
        
        found_projects = sum(1 for keyword in project_keywords if keyword in text_lower)
        
        if found_projects >= 3:
            score = 75
        elif found_projects >= 1:
            score = 55
        else:
            score = 30
        
        return {'raw_score': score, 'reasoning': f'{found_projects} project indicators'}
    
    def _score_soft_skills(self, text: str) -> Dict:
        """Score soft skills."""
        soft_skill_keywords = ['communication', 'leadership', 'teamwork', 'collaboration', 'problem solving', 'analytical']
        text_lower = text.lower()
        
        found_skills = sum(1 for keyword in soft_skill_keywords if keyword in text_lower)
        
        if found_skills >= 3:
            score = 70
        elif found_skills >= 1:
            score = 50
        else:
            score = 35
        
        return {'raw_score': score, 'reasoning': f'{found_skills} soft skill mentions'}
    
    def _score_industry_knowledge(self, text: str, job_criteria: Dict) -> Dict:
        """Score industry knowledge."""
        industry = job_criteria.get('industry', 'general')
        if industry == 'general':
            return {'raw_score': 50, 'reasoning': 'No specific industry'}
        
        industry_keywords = {
            'fintech': ['finance', 'banking', 'payment', 'transaction'],
            'healthcare': ['medical', 'health', 'patient', 'clinical'],
            'saas': ['saas', 'software', 'cloud', 'subscription'],
            'ecommerce': ['ecommerce', 'retail', 'shopping', 'payment']
        }
        
        text_lower = text.lower()
        keywords = industry_keywords.get(industry, [])
        
        if keywords:
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = 40 + (matches * 15)
        else:
            score = 50
        
        return {'raw_score': min(100, score), 'reasoning': f'{industry} industry match'}

