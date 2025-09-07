from __future__ import annotations

from agents.job_description_parser_agent import JobDescriptionParserAgent
from agents.resume_screener_agent import ResumeScreenerAgent
from agents.matchmaker_agent import MatchmakerAgent


def run(job_pdf: str, resume_pdf: str):
    jd_agent = JobDescriptionParserAgent()
    resume_agent = ResumeScreenerAgent()
    matchmaker = MatchmakerAgent()

    job = jd_agent.parse(job_pdf)
    resume = resume_agent.parse(resume_pdf)
    result = matchmaker.score(resume, job)

    # Return structured result; printing is handled by callers if needed
    return result


