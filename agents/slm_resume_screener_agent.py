"""
SLM-based Resume Screener Agent (NLP-based, NO LLMs).
Uses rule-based extraction instead of language models.
"""

from __future__ import annotations
from typing import List
from services.pdf_utils import extract_text_from_pdf
from services.nlp_extractors import SkillsExtractor, ExperienceExtractor, EducationClassifier
from common.models import ResumeData


class SLMResumeScreenerAgent:
    """
    Resume parser using pure NLP techniques (NO LLMs).
    Uses dictionary-based skills extraction, date parsing, and rule-based classification.
    """
    
    def __init__(self):
        self.skills_extractor = SkillsExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.education_classifier = EducationClassifier()
    
    def parse(self, pdf_path: str) -> ResumeData:
        """
        Parse resume PDF and extract structured data using NLP.
        
        Args:
            pdf_path: Path to resume PDF file
            
        Returns:
            ResumeData object with extracted information
        """
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Extract skills using NLP
        extracted_skills = self.skills_extractor.extract(text)
        
        # Extract education level
        education_level = self.education_classifier.classify(text)
        
        # Extract experience years
        total_experience_years = self.experience_extractor.extract_experience(text)
        
        return ResumeData(
            file_path=pdf_path,
            raw_text=text,
            extracted_skills=extracted_skills,
            education_level=education_level,
            total_experience_years=total_experience_years,
        )
    
    def parse_from_text(self, text: str, file_path: str = "text_input") -> ResumeData:
        """
        Parse resume from text directly (without PDF extraction).
        
        Args:
            text: Resume text content
            file_path: Optional file path identifier
            
        Returns:
            ResumeData object
        """
        # Extract skills using NLP
        extracted_skills = self.skills_extractor.extract(text)
        
        # Extract education level
        education_level = self.education_classifier.classify(text)
        
        # Extract experience years
        total_experience_years = self.experience_extractor.extract_experience(text)
        
        return ResumeData(
            file_path=file_path,
            raw_text=text,
            extracted_skills=extracted_skills,
            education_level=education_level,
            total_experience_years=total_experience_years,
        )

