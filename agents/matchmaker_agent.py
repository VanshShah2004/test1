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

        # Experience matching based on parsed years
        resume_exp = resume.total_experience_years or 0.0
        job_required_exp = job.min_experience_years or 0
        
        if job_required_exp <= 0:
            # Job doesn't require experience (e.g., internship/entry-level)
            exp_match = 100
        elif resume_exp >= job_required_exp:
            # Candidate meets or exceeds requirement
            exp_match = 100
        elif resume_exp > 0:
            # Candidate has some experience but less than required
            # Score based on percentage: 40-90 range
            percentage = resume_exp / job_required_exp
            exp_match = int(40 + (percentage * 50))  # 40-90 range
            exp_match = min(90, max(40, exp_match))  # Clamp between 40-90
        else:
            # Candidate has no experience but job requires some
            exp_match = 30

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


