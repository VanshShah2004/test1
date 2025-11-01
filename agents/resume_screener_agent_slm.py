"""
SLM-based Resume Screener Agent using pure NLP (NO LLMs).
Uses rule-based parsing, regex patterns, and keyword extraction.
"""

from __future__ import annotations

from services.pdf_utils import extract_text_from_pdf
from services.nlp_service import NLPService
from common.models import ResumeData


class ResumeScreenerAgentSLM:
    """
    Resume parser using pure NLP techniques (SLM approach).
    No LLM API calls - deterministic, fast, local processing.
    """
    
    def __init__(self, nlp_service: NLPService | None = None):
        self._nlp = nlp_service or NLPService()
    
    def parse(self, pdf_path: str) -> ResumeData:
        """
        Parse resume PDF and extract structured data.
        Uses NLP-based extraction (no LLM).
        """
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Use NLP service to parse resume
        parsed_data = self._nlp.parse_resume(text)
        
        # Return ResumeData model
        return ResumeData(
            file_path=pdf_path,
            raw_text=text,
            extracted_skills=parsed_data.get('extracted_skills', []),
            education_level=parsed_data.get('education_level', 'bachelors'),
            total_experience_years=parsed_data.get('total_experience_years', 0.0),
        )

