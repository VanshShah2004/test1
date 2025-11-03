"""
Hybrid Resume Scoring Agent combining LLM (Gemini) and SLM (NLP-based) results.
Uses weighted consensus: 0.65 LLM, 0.35 SLM
Falls back to 100% SLM if LLM fails.
"""

from __future__ import annotations
import json
import traceback
import os
from typing import Dict, Any, List
from services.pdf_utils import extract_text_from_pdf
from services.llm import LLMService
from agents.slm_scoring_agent import SLMScoringAgent
from structured_scoring_agent import StructuredScoringAgent


class HybridScoringAgent:
    """
    Hybrid scoring agent that combines LLM and SLM results with weighted consensus.
    - LLM weight: 0.65 (better at nuanced evaluation)
    - SLM weight: 0.35 (deterministic, rule-based)
    - Fallback: 100% SLM if LLM fails
    """
    
    def __init__(self, llm_model: str = "gemini-2.0-flash-exp"):
        self.llm_scoring_agent = StructuredScoringAgent(model=llm_model)
        self.slm_scoring_agent = SLMScoringAgent()
        self.llm_weight = 0.65
        self.slm_weight = 0.35
        self._slm_metrics = None  # Store SLM metrics after calculation
    
    def score_resume(
        self, 
        resume_path: str, 
        job_description_path: str, 
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Score resume using hybrid LLM + SLM approach with weighted consensus.
        
        Args:
            resume_path: Path to resume PDF
            job_description_path: Path to job description PDF
            criteria_requirements: Dictionary of criteria names and weights
            
        Returns:
            Dictionary with hybrid scoring results and metadata
        """
        # Extract text from PDFs
        try:
            resume_text = extract_text_from_pdf(resume_path)
            job_description_text = extract_text_from_pdf(job_description_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error extracting text: {e}",
                "resume_path": resume_path,
                "job_description_path": job_description_path
            }
        
        # Get SLM score (always available, deterministic)
        slm_result = self.slm_scoring_agent.score_resume(
            resume_text,
            job_description_text,
            criteria_requirements
        )
        
        # Try to get LLM score (may fail)
        llm_result = None
        llm_error = None
        
        try:
            llm_response = self.llm_scoring_agent.score_resume(
                resume_path,
                job_description_path,
                criteria_requirements
            )
            
            if llm_response.get("success"):
                llm_result = llm_response.get("scoring_result")
            else:
                llm_error = llm_response.get("error", "Unknown LLM error")
                
        except Exception as e:
            # LLM failed (network, rate limit, API error, etc.)
            llm_error = str(e)
            print(f"⚠️  LLM scoring failed: {llm_error}")
            print(f"   Falling back to 100% SLM scoring")
        
        # Combine results based on availability
        if llm_result is None:
            # Fallback: 100% SLM
            return self._create_fallback_result(slm_result, resume_path, job_description_path, criteria_requirements, llm_error)
        else:
            # Hybrid: Weighted consensus
            return self._create_hybrid_result(
                llm_result, 
                slm_result, 
                resume_path, 
                job_description_path, 
                criteria_requirements
            )
    
    def score_resumes_batch(
        self,
        resume_paths: List[str],
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Score multiple resumes using hybrid approach.
        Process:
        1. Score ALL resumes with LLM (batch)
        2. Score ALL resumes with SLM (sequential)
        3. Calculate weighted mean for each resume
        
        Args:
            resume_paths: List of paths to resume PDFs
            job_description_path: Path to job description PDF
            criteria_requirements: Dictionary of criteria names and weights
            
        Returns:
            List of scoring results with hybrid scores
        """
        print("="*70)
        print("HYBRID SCORING: LLM + SLM Batch Processing")
        print("="*70)
        
        # Step 1: Score ALL resumes with LLM first
        print("\n📊 STEP 1: Scoring all resumes with LLM (Gemini)...")
        llm_results = None
        llm_error = None
        
        try:
            llm_batch_results = self.llm_scoring_agent.score_resumes_batch(
                resume_paths,
                job_description_path,
                criteria_requirements
            )
            
            # Validate LLM results
            if llm_batch_results and all(r.get("success", False) for r in llm_batch_results):
                llm_results = {}
                for i, result in enumerate(llm_batch_results):
                    resume_path = resume_paths[i]
                    if result.get("success"):
                        llm_results[resume_path] = result.get("scoring_result", {})
                    else:
                        print(f"⚠️  LLM scoring failed for {resume_path}: {result.get('error')}")
                print(f"✅ LLM scoring completed for {len(llm_results)}/{len(resume_paths)} resumes")
            else:
                llm_error = "LLM batch scoring failed or incomplete"
                print(f"❌ {llm_error}")
                
        except Exception as e:
            llm_error = str(e)
            print(f"❌ LLM scoring failed: {llm_error}")
            print(f"   Reason: {type(e).__name__}")
            traceback.print_exc()
        
        # Step 2: Score ALL resumes with SLM
        print("\n📊 STEP 2: Scoring all resumes with SLM (NLP-based)...")
        slm_results = {}
        resume_texts = {}  # Store resume texts for metrics calculation
        
        try:
            from services.pdf_utils import extract_text_from_pdf
            job_description_text = extract_text_from_pdf(job_description_path)
            
            for i, resume_path in enumerate(resume_paths):
                try:
                    resume_text = extract_text_from_pdf(resume_path)
                    resume_texts[resume_path] = resume_text
                    
                    slm_result = self.slm_scoring_agent.score_resume(
                        resume_text,
                        job_description_text,
                        criteria_requirements
                    )
                    
                    slm_results[resume_path] = slm_result
                    print(f"  [{i+1}/{len(resume_paths)}] SLM scored: {os.path.basename(resume_path)}")
                    
                except Exception as e:
                    print(f"⚠️  SLM scoring failed for {resume_path}: {e}")
                    slm_results[resume_path] = None
            
            successful_count = len([r for r in slm_results.values() if r is not None])
            print(f"✅ SLM scoring completed for {successful_count}/{len(resume_paths)} resumes")
            
            # Calculate SLM accuracy and precision metrics
            print("\n📈 Calculating SLM Accuracy & Precision Metrics...")
            slm_metrics_result = None
            try:
                from services.slm_metrics import SLMMetricsCalculator
                metrics_calc = SLMMetricsCalculator()
                
                metrics = metrics_calc.calculate_accuracy_precision(
                    slm_predictions=slm_results,
                    resume_texts=resume_texts,
                    job_description_text=job_description_text,
                    criteria_requirements=criteria_requirements
                )
                
                slm_metrics_result = metrics  # Store for later use
                
                if metrics.get("available"):
                    if metrics.get("matched_samples", 0) > 0:
                        self._display_slm_metrics(metrics)
                    else:
                        print(f"  ℹ️  No matching ground truth records found in dataset")
                        print(f"     Total samples processed: {metrics.get('total_samples', 0)}")
                else:
                    print(f"  ℹ️  Ground truth dataset not available for metrics calculation")
                    
            except Exception as e:
                print(f"  ⚠️  Could not calculate SLM metrics: {e}")
                import traceback
                traceback.print_exc()
            
            # Store metrics for later reference
            self._slm_metrics = slm_metrics_result
            
        except Exception as e:
            print(f"❌ SLM batch scoring failed: {e}")
            return self._create_error_results(resume_paths, job_description_path, str(e))
        
        # Step 3: Calculate weighted mean for each resume
        print("\n📊 STEP 3: Calculating weighted mean (LLM: 0.65, SLM: 0.35)...")
        
        hybrid_results = []
        
        for resume_path in resume_paths:
            slm_result = slm_results.get(resume_path)
            
            if slm_result is None:
                # SLM failed for this resume
                hybrid_results.append({
                    "success": False,
                    "error": "Both LLM and SLM scoring failed",
                    "resume_path": resume_path
                })
                continue
            
            llm_result = llm_results.get(resume_path) if llm_results else None
            
            if llm_result is None:
                # LLM failed - use 100% SLM (fallback)
                print(f"  ⚠️  Using SLM fallback for {os.path.basename(resume_path)}")
                result = self._create_fallback_result(
                    slm_result,
                    resume_path,
                    job_description_path,
                    criteria_requirements,
                    llm_error
                )
            else:
                # Hybrid: Weighted consensus
                result = self._create_hybrid_result(
                    llm_result,
                    slm_result,
                    resume_path,
                    job_description_path,
                    criteria_requirements
                )
                print(f"  ✅ Hybrid score calculated for {os.path.basename(resume_path)}")
            
            hybrid_results.append(result)
        
        print(f"\n✅ Hybrid scoring completed for {len(hybrid_results)} resumes")
        
        # Print summary
        hybrid_count = sum(1 for r in hybrid_results if r.get("success") and 
                          r.get("scoring_result", {}).get("metadata", {}).get("scoring_method") == "hybrid_llm_slm")
        fallback_count = sum(1 for r in hybrid_results if r.get("success") and 
                            r.get("scoring_result", {}).get("metadata", {}).get("scoring_method") == "slm_fallback")
        
        print(f"\n📈 Summary:")
        print(f"  • Hybrid (LLM + SLM): {hybrid_count} resumes")
        print(f"  • Fallback (SLM only): {fallback_count} resumes")
        print(f"  • Failed: {len(hybrid_results) - hybrid_count - fallback_count} resumes")
        
        # Store results and metrics for later access
        self._hybrid_results = hybrid_results
        self._slm_results = slm_results
        self._resume_texts = resume_texts
        self._job_description_text = job_description_text if 'job_description_text' in locals() else None
        self._criteria_requirements = criteria_requirements
        
        return hybrid_results
    
    def get_slm_metrics(self) -> Dict[str, Any]:
        """Get stored SLM metrics. Returns None if not calculated yet."""
        return self._slm_metrics
    
    def calculate_and_display_slm_metrics(self):
        """Calculate and display SLM metrics (can be called after batch scoring)."""
        if not hasattr(self, '_slm_results') or not self._slm_results:
            print("\n⚠️  Cannot calculate SLM metrics: No SLM results available")
            return None
        
        print("\n📈 Calculating SLM Accuracy & Precision Metrics...")
        try:
            from services.slm_metrics import SLMMetricsCalculator
            metrics_calc = SLMMetricsCalculator()
            
            metrics = metrics_calc.calculate_accuracy_precision(
                slm_predictions=self._slm_results,
                resume_texts=self._resume_texts if hasattr(self, '_resume_texts') else {},
                job_description_text=self._job_description_text or "",
                criteria_requirements=self._criteria_requirements if hasattr(self, '_criteria_requirements') else {}
            )
            
            self._slm_metrics = metrics
            
            if metrics.get("available"):
                if metrics.get("matched_samples", 0) > 0:
                    self._display_slm_metrics(metrics)
                else:
                    print(f"  ℹ️  No matching ground truth records found in dataset")
                    print(f"     Total samples processed: {metrics.get('total_samples', 0)}")
                    print(f"     💡 Tip: Metrics require matching records in kaggle_dataset/hf_clean_data/")
            else:
                print(f"  ℹ️  Ground truth dataset not available for metrics calculation")
            
            return metrics
                    
        except Exception as e:
            print(f"  ⚠️  Could not calculate SLM metrics: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_error_results(
        self,
        resume_paths: List[str],
        job_description_path: str,
        error: str
    ) -> List[Dict[str, Any]]:
        """Create error results for all resumes."""
        return [{
            "success": False,
            "error": error,
            "resume_path": resume_path,
            "job_description_path": job_description_path
        } for resume_path in resume_paths]
    
    def _display_slm_metrics(self, metrics: Dict[str, Any]):
        """Display SLM accuracy and precision metrics."""
        overall = metrics.get("overall_metrics", {})
        matched = metrics.get("matched_samples", 0)
        total = metrics.get("total_samples", 0)
        
        print(f"  ✅ Matched {matched}/{total} resumes with ground truth dataset")
        print()
        print("  📊 OVERALL METRICS:")
        print(f"     • Accuracy (±10 points): {overall.get('accuracy', 0):.1%}")
        print(f"     • Mean Absolute Error (MAE): {overall.get('mae', 0):.2f} points")
        print(f"     • Root Mean Squared Error (RMSE): {overall.get('rmse', 0):.2f} points")
        print(f"     • R² Score: {overall.get('r2_score', 0):.3f}")
        print(f"     • Correlation: {overall.get('correlation', 0):.3f}")
        
        if overall.get('precision_high_scores') is not None:
            print()
            print("  🎯 HIGH SCORE PREDICTION (≥80 points):")
            print(f"     • Precision: {overall.get('precision_high_scores', 0):.1%}")
            print(f"     • Recall: {overall.get('recall_high_scores', 0):.1%}")
            print(f"     • F1 Score: {overall.get('f1_high_scores', 0):.3f}")
        
        criterion_metrics = metrics.get("criterion_metrics", {})
        if criterion_metrics:
            print()
            print("  📋 PER-CRITERION METRICS:")
            for criterion, crit_metrics in list(criterion_metrics.items())[:5]:  # Show top 5
                print(f"     • {criterion}:")
                print(f"       - Accuracy: {crit_metrics.get('accuracy', 0):.1%}")
                print(f"       - MAE: {crit_metrics.get('mae', 0):.2f} points")
                print(f"       - Samples: {crit_metrics.get('samples', 0)}")
    
    def _create_hybrid_result(
        self,
        llm_result: Dict[str, Any],
        slm_result: Dict[str, Any],
        resume_path: str,
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Create hybrid scoring result by combining LLM and SLM scores.
        """
        # Get all criteria from requirements
        criteria_keys = list(criteria_requirements.keys())
        
        # Combine scores for each criterion
        hybrid_scores = {}
        total_weighted_contribution = 0.0
        
        for criterion in criteria_keys:
            # Get LLM score
            llm_criterion_data = llm_result.get(criterion, {})
            llm_raw_score = llm_criterion_data.get("raw_score", 0.0)
            llm_weighted = llm_criterion_data.get("weighted_contribution", 0.0)
            
            # Get SLM score
            slm_criterion_data = slm_result.get(criterion, {})
            slm_raw_score = slm_criterion_data.get("raw_score", 0.0)
            slm_weighted = slm_criterion_data.get("weighted_contribution", 0.0)
            
            # Weighted consensus
            hybrid_raw_score = (self.llm_weight * llm_raw_score) + (self.slm_weight * slm_raw_score)
            hybrid_weighted = (self.llm_weight * llm_weighted) + (self.slm_weight * slm_weighted)
            
            total_weighted_contribution += hybrid_weighted
            
            # Get weight info
            weight_given = criteria_requirements[criterion]
            normalized_pct = slm_criterion_data.get("normalized_percentage", 0.0)
            
            hybrid_scores[criterion] = {
                "raw_score": round(hybrid_raw_score, 1),
                "weight_given": weight_given,
                "normalized_percentage": round(normalized_pct, 1),
                "weighted_contribution": round(hybrid_weighted, 2),
                "llm_score": round(llm_raw_score, 1),
                "slm_score": round(slm_raw_score, 1),
                "discrepancy": round(abs(llm_raw_score - slm_raw_score), 1)
            }
        
        # Calculate metadata
        normalized_weights = self.slm_scoring_agent.normalize_weights(criteria_requirements)
        
        return {
            "success": True,
            "scoring_result": {
                **hybrid_scores,
                "total_score": round(total_weighted_contribution, 1),
                "metadata": {
                    "resume_path": resume_path,
                    "job_description_path": job_description_path,
                    "criteria_requirements": criteria_requirements,
                    "normalized_weights": normalized_weights,
                    "scoring_method": "hybrid_llm_slm",
                    "llm_weight": self.llm_weight,
                    "slm_weight": self.slm_weight,
                    "llm_total_score": llm_result.get("total_score", 0.0),
                    "slm_total_score": slm_result.get("total_score", 0.0),
                    "hybrid_total_score": round(total_weighted_contribution, 1)
                }
            }
        }
    
    def _create_fallback_result(
        self,
        slm_result: Dict[str, Any],
        resume_path: str,
        job_description_path: str,
        criteria_requirements: Dict[str, int],
        llm_error: str
    ) -> Dict[str, Any]:
        """
        Create result using 100% SLM when LLM fails.
        """
        # Add metadata about fallback
        normalized_weights = self.slm_scoring_agent.normalize_weights(criteria_requirements)
        
        # Mark all scores as SLM-only
        fallback_scores = {}
        for criterion, data in slm_result.items():
            if criterion != "total_score" and criterion != "metadata":
                fallback_scores[criterion] = {
                    **data,
                    "llm_score": None,
                    "slm_score": data.get("raw_score", 0.0),
                    "discrepancy": None
                }
        
        return {
            "success": True,
            "scoring_result": {
                **fallback_scores,
                "total_score": slm_result.get("total_score", 0.0),
                "metadata": {
                    "resume_path": resume_path,
                    "job_description_path": job_description_path,
                    "criteria_requirements": criteria_requirements,
                    "normalized_weights": normalized_weights,
                    "scoring_method": "slm_fallback",
                    "llm_weight": 0.0,
                    "slm_weight": 1.0,
                    "llm_total_score": None,
                    "slm_total_score": slm_result.get("total_score", 0.0),
                    "hybrid_total_score": slm_result.get("total_score", 0.0),
                    "llm_error": llm_error,
                    "fallback_reason": "LLM unavailable or failed"
                }
            }
        }
    
    def format_results(self, result: Dict[str, Any]) -> str:
        """Format hybrid results for display."""
        if not result.get("success"):
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        scoring = result["scoring_result"]
        metadata = scoring.get("metadata", {})
        
        output = []
        output.append("=" * 70)
        output.append("HYBRID RESUME SCORING RESULTS (LLM + SLM)")
        output.append("=" * 70)
        
        # Show scoring method
        scoring_method = metadata.get("scoring_method", "unknown")
        output.append(f"\n📊 Scoring Method: {scoring_method}")
        
        if scoring_method == "hybrid_llm_slm":
            output.append(f"   LLM Weight: {metadata.get('llm_weight', 0.65)}")
            output.append(f"   SLM Weight: {metadata.get('slm_weight', 0.35)}")
        elif scoring_method == "slm_fallback":
            output.append(f"   ⚠️  Using SLM Fallback (LLM unavailable)")
            output.append(f"   Reason: {metadata.get('llm_error', 'Unknown')}")
        
        # Show criteria and scores
        output.append(f"\n📋 CRITERIA & WEIGHTS:")
        for criterion, weight in metadata.get("criteria_requirements", {}).items():
            normalized = metadata.get("normalized_weights", {}).get(criterion, 0.0)
            output.append(f"  • {criterion}: {weight} → {normalized:.1f}%")
        
        # Show individual scores
        output.append(f"\n📊 INDIVIDUAL SCORES:")
        total_contribution = 0
        
        for criterion, data in scoring.items():
            if criterion in ["total_score", "metadata"]:
                continue
            
            raw_score = data.get("raw_score", 0)
            weighted_contribution = data.get("weighted_contribution", 0)
            total_contribution += weighted_contribution
            
            if scoring_method == "hybrid_llm_slm":
                llm_score = data.get("llm_score", 0)
                slm_score = data.get("slm_score", 0)
                discrepancy = data.get("discrepancy", 0)
                output.append(
                    f"  • {criterion}: {raw_score}/100 "
                    f"(LLM: {llm_score}, SLM: {slm_score}, Δ: {discrepancy}) "
                    f"→ {weighted_contribution:.1f} points"
                )
            else:
                output.append(
                    f"  • {criterion}: {raw_score}/100 (SLM only) "
                    f"→ {weighted_contribution:.1f} points"
                )
        
        # Show total score
        total_score = scoring.get("total_score", total_contribution)
        output.append(f"\n🎯 TOTAL SCORE: {total_score:.1f}/100")
        
        if scoring_method == "hybrid_llm_slm":
            llm_total = metadata.get("llm_total_score", 0.0)
            slm_total = metadata.get("slm_total_score", 0.0)
            output.append(f"\n📈 COMPONENT SCORES:")
            output.append(f"  • LLM Score: {llm_total:.1f}/100")
            output.append(f"  • SLM Score: {slm_total:.1f}/100")
            output.append(f"  • Hybrid Score: {total_score:.1f}/100")
        
        return "\n".join(output)

