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
        From this resume text, list key technical skills (lowercase, deduplicated) and highest
        education level (one of: high_school, bachelors, masters, doctorate). Return ONLY JSON
        with keys: extracted_skills, education_level.
        Text:\n{text}
        """
        content = self._llm.generate(system, prompt).strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        try:
            import json
            data = json.loads(content)
        except Exception:
            data = {"extracted_skills": [], "education_level": "bachelors"}
        return ResumeData(
            file_path=pdf_path,
            raw_text=text,
            extracted_skills=data.get("extracted_skills", []),
            education_level=data.get("education_level", "bachelors"),
            total_experience_years=0.0,
        )


