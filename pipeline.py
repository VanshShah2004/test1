from __future__ import annotations

from main_orchestrator import run as orchestrate
from resumeScreenerAgent import main as screener_main


def run_pipeline(job_pdf: str, resume_pdf: str):
    # Orchestrate to get structured result if needed
    _ = orchestrate(job_pdf, resume_pdf)
    # For CLI output consistency, reuse existing formatted screener output
    screener_main(job_description_pdf_path=job_pdf, resume_file_path=resume_pdf)


if __name__ == "__main__":
    # Example usage
    run_pipeline(
        job_pdf="documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf",
        resume_pdf="documents/Nirmit_Jain_Resume_Final.pdf",
    )


