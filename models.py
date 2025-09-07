from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class JobCriteria(BaseModel):
	position: str = "Unknown Position"
	required_skills: List[str] = Field(default_factory=list)
	preferred_skills: List[str] = Field(default_factory=list)
	min_experience_years: int = 0
	education_level: str = "bachelors"
	industry: str = "general"
	company_size: str = "medium"
	remote_work: bool = False


class ResumeData(BaseModel):
	file_path: str
	raw_text: str
	extracted_skills: List[str] = Field(default_factory=list)
	education_level: str = "bachelors"
	total_experience_years: float = 0.0


class MatchScores(BaseModel):
	skills_match: int = 0
	experience_match: int = 0
	education_match: int = 0
	overall_score: int = 0


class MatchResult(BaseModel):
	job: JobCriteria
	resume: ResumeData
	scores: MatchScores
	analysis_notes: Dict[str, Any] = Field(default_factory=dict)


