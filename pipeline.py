#!/usr/bin/env python3
"""
Main Pipeline - Uses main_orchestrator and agents with structured scoring
"""

import json
import os
from typing import Dict, List, Any
from main_orchestrator import run as orchestrate
from structured_scoring_agent import StructuredScoringAgent
from datetime import datetime
import time

def _ensure_output_dir(path: str = "outputs") -> str:
    """Ensure outputs directory exists and return its path."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def _flatten_scoring_result(scoring_result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single scoring_result dict for CSV/DF friendliness."""
    flat: Dict[str, Any] = {}
    metadata = scoring_result.get("metadata", {})
    flat["resume_path"] = metadata.get("resume_path", "")
    flat["job_description_path"] = metadata.get("job_description_path", "")
    flat["total_score"] = scoring_result.get("total_score", 0)

    # Per-criterion fields
    for criterion, data in scoring_result.items():
        if criterion in ("total_score", "metadata"):
            continue
        raw_score = data.get("raw_score", None)
        weight_given = data.get("weight_given", None)
        normalized_percentage = data.get("normalized_percentage", None)
        weighted_contribution = data.get("weighted_contribution", None)
        flat[f"{criterion}__raw_score"] = raw_score
        flat[f"{criterion}__weight_given"] = weight_given
        flat[f"{criterion}__normalized_percentage"] = normalized_percentage
        flat[f"{criterion}__weighted_contribution"] = weighted_contribution

    # Optional gap analysis if present
    gap = scoring_result.get("gap_analysis", {})
    if gap:
        flat["missing_skills"] = ", ".join(gap.get("missing_skills", []))
        flat["experience_gap_years"] = gap.get("experience_gap_years", None)
        flat["education_gap"] = gap.get("education_gap", "")

    return flat

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
    # Map resume path -> orchestrated match result (for gap analysis)
    orchestrated_by_resume: Dict[str, Any] = {}
    try:
        for mr in orchestrate_results:
            try:
                resume_path = mr.resume.file_path
            except Exception:
                resume_path = None
            if resume_path:
                orchestrated_by_resume[resume_path] = mr
    except Exception:
        pass

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
        
        # Enrich with simple gap analysis using orchestrated data
        try:
            if result.get("success"):
                scoring = result["scoring_result"]
                mr = orchestrated_by_resume.get(resume_pdf)
                missing_skills: List[str] = []
                experience_gap_years = None
                education_gap = None
                if mr is not None:
                    job_required = set(getattr(mr.job, "required_skills", []) or [])
                    resume_skills = set(getattr(mr.resume, "extracted_skills", []) or [])
                    missing_skills = sorted(list(job_required - resume_skills))
                    try:
                        min_years = getattr(mr.job, "min_experience_years", 0) or 0
                        got_years = getattr(mr.resume, "total_experience_years", 0.0) or 0.0
                        experience_gap_years = max(0.0, float(min_years) - float(got_years))
                    except Exception:
                        experience_gap_years = None
                    try:
                        job_edu = (getattr(mr.job, "education_level", "") or "").lower()
                        resume_edu = (getattr(mr.resume, "education_level", "") or "").lower()
                        education_gap = None if resume_edu >= job_edu else f"Requires {job_edu}"
                    except Exception:
                        education_gap = None
                scoring["gap_analysis"] = {
                    "missing_skills": missing_skills,
                    "experience_gap_years": experience_gap_years,
                    "education_gap": education_gap
                }
        except Exception:
            pass

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
    
    # Persist structured results for dashboard consumption
    out_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, "scoring_results.json")
    json_snap_path = os.path.join(out_dir, f"scoring_results_{timestamp}.json")
    flat_csv_path = os.path.join(out_dir, "scoring_results_flat.csv")

    try:
        serializable = results
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        with open(json_snap_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved JSON results → {json_path}")
    except Exception as e:
        print(f"⚠️  Failed to save JSON results: {e}")

    # Also write a flattened CSV for quick analysis
    try:
        import csv
        flattened_rows: List[Dict[str, Any]] = []
        for r in results:
            if r.get("success"):
                flattened_rows.append(_flatten_scoring_result(r["scoring_result"]))
        # Collect all keys to stabilize CSV header
        all_keys = set()
        for row in flattened_rows:
            all_keys.update(row.keys())
        fieldnames = sorted(list(all_keys))
        with open(flat_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flattened_rows:
                writer.writerow(row)
        print(f"💾 Saved CSV results → {flat_csv_path}")
    except Exception as e:
        print(f"⚠️  Failed to save CSV results: {e}")

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
