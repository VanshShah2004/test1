"""
Example usage of Hybrid Scoring Agent (LLM + SLM).
Demonstrates batch scoring with weighted consensus.
"""

from agents.hybrid_scoring_agent import HybridScoringAgent
import json
from pathlib import Path


def main():
    """Example: Score multiple resumes using hybrid LLM + SLM approach."""
    
    # Initialize hybrid scoring agent
    hybrid_agent = HybridScoringAgent()
    
    # Define criteria with weights
    criteria_requirements = {
        "technical_skills": 60,
        "experience": 40,
        "education": 15,
        "presentation": 10,
        "certifications": 5,
        "projects": 5
    }
    
    # Example: Get resume paths (update with your actual paths)
    resume_paths = [
        "documents/final4resume.pdf",
        "documents/final5resume.pdf",
        "documents/Nirmit_Jain_Resume_Final.pdf",
    ]
    
    # Filter to only existing files
    resume_paths = [path for path in resume_paths if Path(path).exists()]
    
    if not resume_paths:
        print("❌ No resume files found. Please update resume_paths in the script.")
        return
    
    job_description_path = "documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf"
    
    if not Path(job_description_path).exists():
        print(f"❌ Job description not found: {job_description_path}")
        return
    
    print(f"📄 Scoring {len(resume_paths)} resumes against job description...")
    print(f"📋 Criteria: {criteria_requirements}")
    print()
    
    # Score all resumes using hybrid approach
    results = hybrid_agent.score_resumes_batch(
        resume_paths=resume_paths,
        job_description_path=job_description_path,
        criteria_requirements=criteria_requirements
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for i, result in enumerate(results):
        if not result.get("success"):
            print(f"\n❌ Resume {i+1}: {result.get('error')}")
            continue
        
        scoring_result = result.get("scoring_result", {})
        metadata = scoring_result.get("metadata", {})
        method = metadata.get("scoring_method", "unknown")
        
        resume_name = Path(resume_paths[i]).name
        total_score = scoring_result.get("total_score", 0.0)
        
        print(f"\n📄 Resume {i+1}: {resume_name}")
        print(f"   Method: {method}")
        print(f"   Total Score: {total_score:.1f}/100")
        
        if method == "hybrid_llm_slm":
            llm_score = metadata.get("llm_total_score", 0.0)
            slm_score = metadata.get("slm_total_score", 0.0)
            print(f"   LLM Score: {llm_score:.1f} (weight: 0.65)")
            print(f"   SLM Score: {slm_score:.1f} (weight: 0.35)")
            print(f"   Hybrid Score: {total_score:.1f} (weighted mean)")
        elif method == "slm_fallback":
            slm_score = metadata.get("slm_total_score", 0.0)
            print(f"   SLM Score: {slm_score:.1f} (fallback - LLM unavailable)")
            print(f"   Reason: {metadata.get('llm_error', 'Unknown')}")
    
    # Save results to JSON
    output_file = "outputs/hybrid_scoring_results.json"
    Path("outputs").mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Display detailed results for first successful resume
    print("\n" + "="*70)
    print("DETAILED RESULTS (First Resume)")
    print("="*70)
    
    for result in results:
        if result.get("success"):
            print(hybrid_agent.format_results(result))
            break


if __name__ == "__main__":
    main()

