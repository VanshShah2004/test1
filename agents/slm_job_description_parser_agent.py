"""
SLM-based Job Description Parser Agent (NLP-based, NO LLMs).
Uses rule-based extraction instead of language models.
"""

from __future__ import annotations
import json
from typing import Dict
from services.pdf_utils import extract_text_from_pdf
from services.nlp_extractors import SkillsExtractor, EducationClassifier, JobDescriptionParser
from common.models import JobCriteria


class SLMJobDescriptionParserAgent:
    """
    Job description parser using pure NLP techniques (NO LLMs).
    Uses pattern matching, keyword extraction, and rule-based parsing.
    """
    
    def __init__(self):
        self.skills_extractor = SkillsExtractor()
        self.education_classifier = EducationClassifier()
        self.job_parser = JobDescriptionParser(
            self.skills_extractor,
            self.education_classifier
        )
    
    def parse(self, pdf_path: str) -> JobCriteria:
        """
        Parse job description PDF and extract structured criteria using NLP.
        
        Args:
            pdf_path: Path to job description PDF file
            
        Returns:
            JobCriteria object with extracted information
        """
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Parse using NLP
        data = self.job_parser.parse(text)
        
        return JobCriteria(**data)
    
    def parse_from_text(self, text: str) -> JobCriteria:
        """
        Parse job description from text directly.
        
        Args:
            text: Job description text content
            
        Returns:
            JobCriteria object
        """
        # Parse using NLP
        data = self.job_parser.parse(text)
        
        return JobCriteria(**data)

