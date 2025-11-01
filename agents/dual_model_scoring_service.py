"""
Dual-Model Scoring Service combining LLM (Gemini) and SLM (NLP-based).
Uses weighted consensus: 0.65 LLM + 0.35 SLM.
Falls back to 100% SLM if LLM fails.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List
from services.llm import LLMService
from structured_scoring_agent import StructuredScoringAgent
from agents.structured_scoring_agent_slm import StructuredScoringAgentSLM
from services.pdf_utils import extract_text_from_pdf


class DualModelScoringService:
    """
    Combines LLM and SLM scoring with weighted consensus.
    Weight: 0.65 LLM + 0.35 SLM
    Fallback: 100% SLM if LLM fails.
    """
    
    def __init__(self, llm_model: str = "gemini-2.0-flash-exp"):
        self.llm_weight = 0.65
        self.slm_weight = 0.35
        
        # Initialize both agents
        self.llm_agent = StructuredScoringAgent(model=llm_model)
        self.slm_agent = StructuredScoringAgentSLM()
    
    def score_resume(
        self,
        resume_path: str,
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Score resume using both LLM and SLM, then combine with weighted average.
        Falls back to SLM-only if LLM fails.
        """
        # Try LLM scoring
        llm_result = None
        llm_error = None
        
        try:
            llm_result = self.llm_agent.score_resume(
                resume_path, job_description_path, criteria_requirements
            )
            if not llm_result.get('success'):
                llm_error = llm_result.get('error', 'LLM scoring failed')
                llm_result = None
        except Exception as e:
            llm_error = f"LLM error: {str(e)}"
            llm_result = None
        
        # Always try SLM scoring (local, should not fail)
        slm_result = None
        slm_error = None
        
        try:
            slm_result = self.slm_agent.score_resume(
                resume_path, job_description_path, criteria_requirements
            )
            if not slm_result.get('success'):
                slm_error = slm_result.get('error', 'SLM scoring failed')
                slm_result = None
        except Exception as e:
            slm_error = f"SLM error: {str(e)}"
            slm_result = None
        
        # Determine scoring strategy
        if llm_result and slm_result:
            # Both successful - use weighted consensus
            return self._combine_scores(
                llm_result['scoring_result'],
                slm_result['scoring_result'],
                criteria_requirements,
                use_fallback=False
            )
        elif slm_result:
            # Only SLM successful - use 100% SLM (fallback)
            print(f"⚠️  LLM scoring failed ({llm_error}). Using 100% SLM fallback.")
            return self._create_fallback_result(
                slm_result['scoring_result'],
                criteria_requirements,
                llm_error
            )
        elif llm_result:
            # Only LLM successful (unlikely but handle it)
            print(f"⚠️  SLM scoring failed ({slm_error}). Using 100% LLM.")
            return llm_result
        else:
            # Both failed
            return {
                'success': False,
                'error': f'Both LLM and SLM failed. LLM: {llm_error}, SLM: {slm_error}',
                'resume_path': resume_path,
                'job_description_path': job_description_path
            }
    
    def score_resumes_batch(
        self,
        resume_paths: List[str],
        job_description_path: str,
        criteria_requirements: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Score multiple resumes in batch using dual-model approach.
        
        Process:
        1. Get ALL LLM scores for ALL resumes (one batch call)
        2. Get ALL SLM scores for ALL resumes (one batch call)
        3. Combine each resume's LLM + SLM scores with weighted average (done in code)
        """
        print(f"🔄 Step 1: Getting LLM scores for all {len(resume_paths)} resumes (batch)...")
        
        # Step 1: Get all LLM scores in one batch call
        llm_results = None
        llm_error = None
        try:
            llm_results = self.llm_agent.score_resumes_batch(
                resume_paths=resume_paths,
                job_description_path=job_description_path,
                criteria_requirements=criteria_requirements
            )
            # Check if any failed
            if any(not r.get('success') for r in llm_results):
                llm_error = "Some LLM batch results failed"
                llm_results = None
        except Exception as e:
            llm_error = f"LLM batch scoring error: {str(e)}"
            llm_results = None
        
        print(f"🔄 Step 2: Getting SLM scores for all {len(resume_paths)} resumes (batch)...")
        
        # Step 2: Get all SLM scores in one batch call
        slm_results = None
        slm_error = None
        try:
            slm_results = self.slm_agent.score_resumes_batch(
                resume_paths=resume_paths,
                job_description_path=job_description_path,
                criteria_requirements=criteria_requirements
            )
            # Check if any failed
            if any(not r.get('success') for r in slm_results):
                slm_error = "Some SLM batch results failed"
                slm_results = None
        except Exception as e:
            slm_error = f"SLM batch scoring error: {str(e)}"
            slm_results = None
        
        print(f"🔄 Step 3: Combining LLM + SLM scores with weighted average (0.65 LLM + 0.35 SLM)...")
        
        # Step 3: Combine LLM and SLM results for each resume
        combined_results = []
        
        for i, resume_path in enumerate(resume_paths):
            llm_result = llm_results[i] if llm_results and i < len(llm_results) else None
            slm_result = slm_results[i] if slm_results and i < len(slm_results) else None
            
            # Determine combination strategy
            if llm_result and slm_result and llm_result.get('success') and slm_result.get('success'):
                # Both successful - combine with weighted average
                combined = self._combine_scores(
                    llm_result['scoring_result'],
                    slm_result['scoring_result'],
                    criteria_requirements,
                    use_fallback=False
                )
                combined_results.append(combined)
            elif slm_result and slm_result.get('success'):
                # Only SLM successful - use 100% SLM (fallback)
                print(f"⚠️  Resume {i+1}: LLM failed ({llm_error}). Using 100% SLM fallback.")
                combined = self._create_fallback_result(
                    slm_result['scoring_result'],
                    criteria_requirements,
                    llm_error or "LLM batch scoring failed"
                )
                combined_results.append(combined)
            elif llm_result and llm_result.get('success'):
                # Only LLM successful (unlikely but handle it)
                print(f"⚠️  Resume {i+1}: SLM failed ({slm_error}). Using 100% LLM.")
                combined_results.append(llm_result)
            else:
                # Both failed
                combined_results.append({
                    'success': False,
                    'error': f'Both LLM and SLM failed. LLM: {llm_error}, SLM: {slm_error}',
                    'resume_path': resume_path,
                    'job_description_path': job_description_path
                })
        
        return combined_results
    
    def _combine_scores(
        self,
        llm_scores: Dict[str, Any],
        slm_scores: Dict[str, Any],
        criteria_requirements: Dict[str, int],
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Combine LLM and SLM scores using weighted average.
        Weight: 0.65 LLM + 0.35 SLM (or 1.0 SLM if fallback)
        
        IMPORTANT: All calculations done in code, NOT by LLM.
        - Takes raw scores from LLM and SLM
        - Calculates weighted average: (0.65 * llm_score) + (0.35 * slm_score)
        - Calculates weighted contributions based on criteria weights
        - All math done programmatically, no AI inference
        """
        if use_fallback:
            llm_weight = 0.0
            slm_weight = 1.0
        else:
            llm_weight = self.llm_weight
            slm_weight = self.slm_weight
        
        combined = {}
        total_score = 0.0
        normalized_weights = self.llm_agent.normalize_weights(criteria_requirements)
        
        # Combine each criterion
        for criterion in criteria_requirements.keys():
            llm_criterion = llm_scores.get(criterion, {})
            slm_criterion = slm_scores.get(criterion, {})
            
            llm_raw = llm_criterion.get('raw_score', 0)
            slm_raw = slm_criterion.get('raw_score', 0)
            
            # Weighted average of raw scores
            combined_raw = (llm_weight * llm_raw) + (slm_weight * slm_raw)
            combined_raw = round(combined_raw)
            
            # Calculate weighted contribution
            weight = criteria_requirements.get(criterion, 0)
            normalized_pct = normalized_weights.get(criterion, 0.0)
            weighted_contrib = combined_raw * normalized_pct / 100.0
            total_score += weighted_contrib
            
            combined[criterion] = {
                'raw_score': combined_raw,
                'llm_score': llm_raw,
                'slm_score': slm_raw,
                'weight_given': weight,
                'normalized_percentage': normalized_pct,
                'weighted_contribution': round(weighted_contrib, 2),
                'discrepancy': abs(llm_raw - slm_raw),
                'flagged': abs(llm_raw - slm_raw) > 20  # Flag large discrepancies
            }
        
        combined['total_score'] = round(total_score, 1)
        combined['llm_total'] = llm_scores.get('total_score', 0)
        combined['slm_total'] = slm_scores.get('total_score', 0)
        
        # Metadata
        combined['metadata'] = {
            'scoring_method': 'dual_model_consensus',
            'llm_weight': llm_weight,
            'slm_weight': slm_weight,
            'use_fallback': use_fallback,
            'criteria_requirements': criteria_requirements,
            'normalized_weights': normalized_weights
        }
        
        # Add metadata from original results if available
        if 'metadata' in llm_scores:
            llm_meta = llm_scores['metadata']
            combined['metadata']['llm_metadata'] = llm_meta
            # Extract resume/job paths from LLM metadata
            combined['metadata']['resume_path'] = llm_meta.get('resume_path', '')
            combined['metadata']['job_description_path'] = llm_meta.get('job_description_path', '')
        if 'metadata' in slm_scores:
            slm_meta = slm_scores['metadata']
            combined['metadata']['slm_metadata'] = slm_meta
            # Use SLM paths if LLM paths not available
            if 'resume_path' not in combined['metadata']:
                combined['metadata']['resume_path'] = slm_meta.get('resume_path', '')
            if 'job_description_path' not in combined['metadata']:
                combined['metadata']['job_description_path'] = slm_meta.get('job_description_path', '')
        
        return {
            'success': True,
            'scoring_result': combined
        }
    
    def _create_fallback_result(
        self,
        slm_scores: Dict[str, Any],
        criteria_requirements: Dict[str, int],
        llm_error: str
    ) -> Dict[str, Any]:
        """
        Create result when using SLM fallback (100% SLM).
        """
        result = slm_scores.copy()
        result['metadata'] = result.get('metadata', {})
        result['metadata'].update({
            'scoring_method': 'slm_fallback',
            'llm_weight': 0.0,
            'slm_weight': 1.0,
            'use_fallback': True,
            'llm_error': llm_error,
            'fallback_reason': 'LLM unavailable or failed',
            'criteria_requirements': criteria_requirements,
            'normalized_weights': self.slm_agent.normalize_weights(criteria_requirements)
        })
        
        return {
            'success': True,
            'scoring_result': result,
            'fallback_used': True,
            'fallback_reason': llm_error
        }

