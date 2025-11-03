from __future__ import annotations

import json
from typing import Dict
from services.llm import LLMService
from services.pdf_utils import extract_text_from_pdf
from common.models import JobCriteria


class JobDescriptionParserAgent:
    def __init__(self, llm: LLMService | None = None):
        self._llm = llm or LLMService()

    def parse(self, pdf_path: str) -> JobCriteria:
        text = extract_text_from_pdf(pdf_path)
        system = "You are an expert HR analyst. Return compact, valid JSON only."
        prompt = f"""
        Extract job criteria and return ONLY JSON with keys:
        position, required_skills, preferred_skills, min_experience_years,
        education_level, industry, company_size, remote_work.
        Text:\n{text}
        """
        content = self._llm.generate(system, prompt).strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        data: Dict = json.loads(content)
        return JobCriteria(**data)


