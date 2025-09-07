#!/usr/bin/env python3
"""
Main Pipeline - Uses main_orchestrator and agents with structured scoring
"""

import json
from typing import Dict, List, Any
from main_orchestrator import run as orchestrate
from structured_scoring_agent import StructuredScoringAgent

def load_criteria_from_file(criteria_file: str = None) -> Dict[str, int]:
    """Load criteria from JSON file and convert to simple weight format"""
    if criteria_file is None:
        criteria_file = "criteria_requirements.json"
    
    try:
        with open(criteria_file, 'r') as f:
            criteria = json.load(f)
        
        # Convert complex structure to simple weights
        simple_criteria = {}
        
        # Add scoring criteria
        scoring_criteria = criteria.get("scoring_criteria", {})
        for key, info in scoring_criteria.items():
            simple_criteria[key] = info.get("weight", 10)
        
        # Add additional criteria
        additional_criteria = criteria.get("additional_criteria", {})
        for key, info in additional_criteria.items():
            simple_criteria[key] = info.get("weight", 10)
        
        return simple_criteria
        
    except FileNotFoundError:
        print(f"❌ Criteria file not found: {criteria_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading criteria: {e}")
        return None

def run_pipeline(job_pdf: str, resume_pdfs: List[str], 
                criteria_requirements: Dict[str, int] = None):
    """Run the main pipeline using orchestrator and structured scoring"""
    
    # Step 1: Run orchestrator flow (agents)
    print("=" * 70)
    print("LLM-POWERED RESUME SCREENER AGENT")
    print("=" * 70)
    print("🔄 Running orchestrator flow...")
    
    try:
        orchestrate_results = orchestrate(job_pdf, resume_pdfs)
        print("✅ Orchestrator flow completed")
    except Exception as e:
        print(f"⚠️  Orchestrator flow failed: {e}")
        orchestrate_results = []
    
    # Step 2: Run structured scoring
    print("\n" + "=" * 70)
    print("STRUCTURED SCORING ANALYSIS")
    print("=" * 70)
    
    # Load criteria if not provided
    if criteria_requirements is None:
        criteria_requirements = load_criteria_from_file()
        if criteria_requirements is None:
            # Fallback to default criteria
            criteria_requirements = {
                "technical_skills": 30,
                "experience": 25,
                "education": 15,
                "presentation": 10,
                "career_progression": 10,
                "marketability": 10
            }
    
    # Initialize structured scoring agent
    agent = StructuredScoringAgent()
    
    # Show criteria summary
    total_weight = sum(criteria_requirements.values())
    print(f"📋 CRITERIA (Total Weight: {total_weight}):")
    for criterion, weight in criteria_requirements.items():
        percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
        print(f"  • {criterion}: {weight} ({percentage:.1f}%)")
    print()
    
    # Process each resume with structured scoring
    results = []
    for i, resume_pdf in enumerate(resume_pdfs, 1):
        print("*" * 70)
        print(f"PROCESSING RESUME {i}/{len(resume_pdfs)}: {resume_pdf}")
        print("*" * 70)
        
        # Score resume
        result = agent.score_resume(
            resume_path=resume_pdf,
            job_description_path=job_pdf,
            criteria_requirements=criteria_requirements
        )
        
        # Display results
        print(agent.format_results(result))
        results.append(result)
    
    # Final ranking
    print("\n" + "=" * 70)
    print("FINAL RANKING")
    print("=" * 70)
    
    # Extract scores for ranking
    ranked_results = []
    for i, result in enumerate(results):
        if result.get("success"):
            scoring = result["scoring_result"]
            total_score = scoring.get("total_score", 0)
            resume_path = result["scoring_result"]["metadata"]["resume_path"]
            ranked_results.append((resume_path, total_score))
        else:
            ranked_results.append((resume_pdfs[i], 0))
    
    # Sort by score (highest first)
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Ranked by Total Score:")
    for i, (resume_path, score) in enumerate(ranked_results, 1):
        print(f"{i}. {resume_path} - {score:.1f}/100")
    
    return results

def create_custom_criteria_example():
    """Create an example custom criteria file"""
    custom_criteria = {
        "technical_skills": 70,
        "experience": 50,
        "education": 30,
        "certifications": 20,
        "projects": 10
    }
    
    with open("custom_criteria_simple.json", "w") as f:
        json.dump(custom_criteria, f, indent=2)
    
    print("✅ Created custom_criteria_simple.json")
    return custom_criteria

if __name__ == "__main__":
    # Load criteria from file
    criteria_requirements = load_criteria_from_file()
    
    # Run main pipeline
    run_pipeline(
        job_pdf="documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf",
        resume_pdfs=[
            "documents/final5resume.pdf",
            "documents/Nirmit_Jain_Resume_Final.pdf",
            "documents/final4resume.pdf"
        ],
        criteria_requirements=criteria_requirements
    )
