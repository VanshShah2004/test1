from __future__ import annotations

from common.models import JobCriteria, ResumeData, MatchScores, MatchResult


class MatchmakerAgent:
    def score(self, resume: ResumeData, job: JobCriteria) -> MatchResult:
        # Skills
        job_skills = {s.lower() for s in job.required_skills}
        res_skills = {s.lower() for s in resume.extracted_skills}
        if job_skills:
            skill_match = int(100 * len(job_skills & res_skills) / max(1, len(job_skills)))
        else:
            skill_match = 50

        # Experience heuristic (no parsed years yet)
        exp_match = 60 if job.min_experience_years <= 0 else 40

        # Education
        order = {"high_school": 0, "bachelors": 1, "masters": 2, "doctorate": 3}
        edu_match = 100 if order.get(resume.education_level, 1) >= order.get(job.education_level, 1) else 70

        overall = int(0.6 * skill_match + 0.25 * exp_match + 0.15 * edu_match)

        scores = MatchScores(
            skills_match=skill_match,
            experience_match=exp_match,
            education_match=edu_match,
            overall_score=overall,
        )

        return MatchResult(job=job, resume=resume, scores=scores)


