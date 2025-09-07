from __future__ import annotations

from main_orchestrator import run as orchestrate
from resumeScreenerAgent import main as screener_main


def run_pipeline(job_pdf: str, resume_pdfs: list[str]):
	# Run structured flow (useful for programmatic consumption)
	_ = orchestrate(job_pdf, resume_pdfs)

	# Top banner once
	print("=" * 70)
	print("LLM-POWERED RESUME SCREENER AGENT")
	print("=" * 70)

	results = []
	for rp in resume_pdfs:
		print("*" * 70)
		out = screener_main(job_description_pdf_path=job_pdf, resume_file_path=rp, show_header=False, show_footer=False)
		results.append(out)

	# Footer once
	print("=" * 70)
	print("All analyses complete")
	print("=" * 70)

	# Ranking
	ranked = []
	for item in results:
		scores = item.get("scores") or {}
		overall = None
		if isinstance(scores, dict):
			overall = scores.get("overall_score")
		if overall is None and hasattr(scores, 'overall_score'):
			overall = getattr(scores, 'overall_score')
		ranked.append((item.get("resume_path"), overall or 0))

	ranked.sort(key=lambda x: x[1], reverse=True)
	print("\nFinal Ranking (highest score first):")
	for i, (path, score) in enumerate(ranked, 1):
		print(f"{i}. {path} - {score}")


if __name__ == "__main__":
	run_pipeline(
		job_pdf="documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf",
		resume_pdfs=[
			"documents/final5resume.pdf",
			"documents/Nirmit_Jain_Resume_Final.pdf",
			"documents/final4resume.pdf"
		],
	)


