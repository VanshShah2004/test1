"""
SLM-based Job Description Parser Agent using pure NLP (NO LLMs).
Uses rule-based parsing, regex patterns, and keyword extraction.
"""

from __future__ import annotations

import json
from typing import Dict
from services.pdf_utils import extract_text_from_pdf
from services.nlp_service import NLPService
from common.models import JobCriteria


class JobDescriptionParserAgentSLM:
    """
    Job description parser using pure NLP techniques (SLM approach).
    No LLM API calls - deterministic, fast, local processing.
    """
    
    def __init__(self, nlp_service: NLPService | None = None):
        self._nlp = nlp_service or NLPService()
    
    def parse(self, pdf_path: str) -> JobCriteria:
        """
        Parse job description PDF and extract criteria.
        Uses NLP-based extraction (no LLM).
        """
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Use NLP service to parse job description
        parsed_data = self._nlp.parse_job_description(text)
        
        # Return JobCriteria model
        return JobCriteria(**parsed_data)

