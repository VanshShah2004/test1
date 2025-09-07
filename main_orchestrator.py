from __future__ import annotations

from agents.job_description_parser_agent import JobDescriptionParserAgent
from agents.resume_screener_agent import ResumeScreenerAgent
from agents.matchmaker_agent import MatchmakerAgent


def run(job_pdf: str, resume_pdfs: list[str]):
    """Run end-to-end scoring for one job description against many resumes.

    Returns list of MatchResult in same order as resume_pdfs.
    """
    jd_agent = JobDescriptionParserAgent()
    resume_agent = ResumeScreenerAgent()
    matchmaker = MatchmakerAgent()

    job = jd_agent.parse(job_pdf)
    results = []
    for resume_pdf in resume_pdfs:
        resume = resume_agent.parse(resume_pdf)
        results.append(matchmaker.score(resume, job))

    return results


