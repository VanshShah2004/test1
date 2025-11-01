from __future__ import annotations

from typing import List
from services.pdf_utils import extract_text_from_pdf
from services.llm import LLMService
from common.models import ResumeData


class ResumeScreenerAgent:
    def __init__(self, llm: LLMService | None = None):
        self._llm = llm or LLMService()

    def parse(self, pdf_path: str) -> ResumeData:
        text = extract_text_from_pdf(pdf_path)
        system = "You are an expert resume parser. Return concise, correct outputs."
        prompt = f"""
        From this resume text, extract:
        1. Key technical skills (lowercase, deduplicated list)
        2. Highest education level (one of: high_school, bachelors, masters, doctorate)
        3. Total years of professional work experience (calculate from employment dates, internships count at 0.5x rate)
        
        Return ONLY JSON with keys: extracted_skills, education_level, total_experience_years.
        
        For total_experience_years:
        - Sum all full-time work experience years
        - Count internships as 0.5x their duration (e.g., 6 months = 0.25 years)
        - Only count professional, relevant work experience
        - Return as a decimal number (e.g., 2.5 for 2 years 6 months, 0.0 if no experience)
        
        Text:\n{text}
        """
        content = self._llm.generate(system, prompt).strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        try:
            import json
            data = json.loads(content)
        except Exception:
            data = {"extracted_skills": [], "education_level": "bachelors", "total_experience_years": 0.0}
        
        # Safely extract experience years, ensuring it's a float
        experience_years = 0.0
        try:
            exp_value = data.get("total_experience_years", 0.0)
            if exp_value is None:
                experience_years = 0.0
            else:
                experience_years = float(exp_value)
                # Ensure non-negative
                experience_years = max(0.0, experience_years)
        except (ValueError, TypeError):
            experience_years = 0.0
        
        return ResumeData(
            file_path=pdf_path,
            raw_text=text,
            extracted_skills=data.get("extracted_skills", []),
            education_level=data.get("education_level", "bachelors"),
            total_experience_years=experience_years,
        )


