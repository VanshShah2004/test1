#!/usr/bin/env python3
"""
Structured Resume Scoring Agent with JSON-based criteria and weighted scoring
"""

import json
import re
from typing import Dict, Any, List
from services.llm import LLMService
from services.pdf_utils import extract_text_from_pdf

class StructuredScoringAgent:
    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        self.llm = LLMService(model=model)
    
    def normalize_weights(self, criteria_requirements: Dict[str, int]) -> Dict[str, float]:
        """Normalize weights to percentages (sum to 100)"""
        total_weight = sum(criteria_requirements.values())
        if total_weight == 0:
            return {k: 0.0 for k in criteria_requirements.keys()}
        
        return {k: (v / total_weight) * 100 for k, v in criteria_requirements.items()}
    
    def create_scoring_prompt(self, resume_text: str, job_description: str, 
                            criteria_requirements: Dict[str, int]) -> str:
        """Create the structured scoring prompt"""
        
        normalized_weights = self.normalize_weights(criteria_requirements)
        
        criteria_list = []
        for criterion, weight in criteria_requirements.items():
            normalized = normalized_weights[criterion]
            criteria_list.append(f"  - {criterion}: weight={weight}, normalized={normalized:.1f}%")
        
        criteria_text = "\n".join(criteria_list)
        
        prompt = f"""
You are an Expert Resume Screening Assistant with 10+ years of HR experience.
You will evaluate a candidate's resume against a job description based on given weighted criteria.

SCORING GUIDELINES:
- Be STRICT and DISCRIMINATING in your evaluation
- Look for CONCRETE EVIDENCE, not just keywords
- Penalize POOR FORMATTING, MISSING INFORMATION, and VAGUE DESCRIPTIONS
- Reward QUANTIFIED ACHIEVEMENTS, SPECIFIC TECHNOLOGIES, and CLEAR EXPERIENCE

SCORING SCALE (0-100):
- 90-100: EXCEPTIONAL - Exceeds requirements with clear evidence
- 80-89:  STRONG - Meets requirements well with good evidence  
- 70-79:  GOOD - Meets basic requirements adequately
- 60-69:  AVERAGE - Partially meets requirements, some gaps
- 50-59:  BELOW AVERAGE - Significant gaps, limited evidence
- 40-49:  POOR - Major deficiencies, weak evidence
- 30-39:  VERY POOR - Severe gaps, minimal relevant experience
- 20-29:  INSUFFICIENT - Almost no relevant qualifications
- 10-19:  INADEQUATE - Virtually no match to requirements
- 0-9:    UNACCEPTABLE - No relevant qualifications

EVALUATION CRITERIA:
1. TECHNICAL SKILLS: Look for specific technologies, frameworks, tools mentioned with context
2. EXPERIENCE: Assess relevance, duration, progression, and impact of work experience
3. EDUCATION: Consider degree relevance, institution quality, academic achievements
4. PRESENTATION: Evaluate formatting, clarity, organization, professional appearance
5. CAREER PROGRESSION: Look for growth, increasing responsibilities, leadership roles
6. MARKETABILITY: Assess overall competitiveness, unique value proposition
7. CERTIFICATIONS: Evaluate relevance, credibility, and recency of certifications
8. PROJECTS: Assess complexity, relevance, outcomes, and technical depth
9. SOFT SKILLS: Look for evidence of communication, teamwork, leadership
10. INDUSTRY KNOWLEDGE: Assess domain expertise and industry-specific experience

BE CRITICAL:
- Poor formatting = automatic deduction (max 40 points)
- Vague descriptions = significant penalty (max 50 points)
- Missing key information = major deduction (max 30 points)
- No quantified achievements = substantial penalty (max 50 points)
- Irrelevant experience = low scores (max 40 points)
- Generic skills without context = low scores (max 50 points)
- Typos and grammatical errors = severe penalty (max 20 points)
- Inconsistent formatting = major penalty (max 30 points)
- Missing contact information = critical flaw (max 20 points)
- No clear career progression = significant penalty (max 40 points)

EXAMPLES OF POOR SCORES:
- "Worked on various projects" = 20-30 points (too vague)
- "Good communication skills" = 30-40 points (no evidence)
- "Experience with programming" = 40-50 points (too generic)
- Poor formatting, typos = 20-40 points (unprofessional)
- No quantified achievements = 30-50 points (no impact shown)
- Missing contact information = 10-20 points (incomplete)
- Inconsistent formatting = 20-30 points (unprofessional)
- No relevant experience = 10-30 points (major gap)
- Generic objective statements = 30-40 points (no value)
- Skills listed without context = 40-50 points (unsubstantiated)

EXAMPLES OF HIGH SCORES:
- "Led team of 5 developers, reduced deployment time by 40%" = 85-95 points
- "Built React application serving 10,000+ users" = 80-90 points
- "AWS Certified Solutions Architect, 3 years cloud experience" = 85-90 points
- "Bachelor's in Computer Science, 3.8 GPA, Dean's List" = 80-85 points

RESUME DETAILS:
{resume_text}

JOB DESCRIPTION:
{job_description}

CRITERIA REQUIREMENTS:
{criteria_text}

Return ONLY valid JSON in this exact format:
{{
  "skills": {{
    "raw_score": 85,
    "weight_given": 70,
    "normalized_percentage": 35.0,
    "weighted_contribution": 29.75
  }},
  "experience": {{
    "raw_score": 80,
    "weight_given": 50,
    "normalized_percentage": 25.0,
    "weighted_contribution": 20.0
  }},
  "total_score": 74.0
}}
"""
        return prompt
    
    def create_batch_scoring_prompt(self, resumes_data: List[Dict[str, str]], 
                                   job_description: str, 
                                   criteria_requirements: Dict[str, int]) -> str:
        """Create a comparative batch scoring prompt that evaluates all resumes together"""
        
        normalized_weights = self.normalize_weights(criteria_requirements)
        
        criteria_list = []
        for criterion, weight in criteria_requirements.items():
            normalized = normalized_weights[criterion]
            criteria_list.append(f"  - {criterion}: weight={weight}, normalized={normalized:.1f}%")
        
        criteria_text = "\n".join(criteria_list)
        
        # Format resumes for comparison
        resumes_section = ""
        for i, resume_data in enumerate(resumes_data, 1):
            resumes_section += f"""
{'='*80}
RESUME #{i} - File: {resume_data['resume_path']}
{'='*80}
{resume_data['resume_text']}

"""
        
        # Generate example JSON structure based on actual criteria (for first resume only as example)
        example_json_parts = []
        if resumes_data:
            resume_path_example = resumes_data[0]['resume_path']
            example_json_parts.append(f'  "{resume_path_example}": {{')
            crit_examples = []
            total_weighted_sum = 0.0
            for crit_key in criteria_requirements.keys():
                weight = criteria_requirements[crit_key]
                normalized_pct = normalized_weights[crit_key]
                weighted_contrib = 85.0 * normalized_pct / 100.0
                total_weighted_sum += weighted_contrib
                crit_examples.append(f'    "{crit_key}": {{\n      "raw_score": 85,\n      "weight_given": {weight},\n      "normalized_percentage": {normalized_pct:.1f},\n      "weighted_contribution": {weighted_contrib:.2f}\n    }}')
            example_json_parts.append(',\n'.join(crit_examples))
            example_json_parts.append(f'    "total_score": {total_weighted_sum:.1f}')
            example_json_parts.append('  }')
        example_json = '{\n' + '\n'.join(example_json_parts) + '\n}'
        
        prompt = f"""
You are an Expert Resume Screening Assistant with 10+ years of HR experience.
You will evaluate MULTIPLE candidates' resumes against a job description using COMPARATIVE ANALYSIS.
This is a BATCH EVALUATION where you can see all candidates together for RELATIVE COMPARISON.

CRITICAL: COMPARATIVE SCORING APPROACH
- You are evaluating {len(resumes_data)} candidates TOGETHER for direct comparison
- Use RELATIVE SCORING to ensure consistency across all candidates
- The BEST candidate in this batch should get the highest scores
- The WEAKEST candidate should get the lowest scores
- Scores should reflect RELATIVE QUALITY within this batch
- This reduces hallucination and ensures fair comparison

SCORING GUIDELINES:
- Be STRICT and DISCRIMINATING in your evaluation
- Look for CONCRETE EVIDENCE, not just keywords
- Penalize POOR FORMATTING, MISSING INFORMATION, and VAGUE DESCRIPTIONS
- Reward QUANTIFIED ACHIEVEMENTS, SPECIFIC TECHNOLOGIES, and CLEAR EXPERIENCE
- COMPARE candidates to each other - better candidates should score higher

SCORING SCALE (0-100) - Use RELATIVELY within this batch:
- 90-100: EXCEPTIONAL - Top candidate(s) in this batch, exceeds requirements with clear evidence
- 80-89:  STRONG - Above average in this batch, meets requirements well
- 70-79:  GOOD - Average in this batch, meets basic requirements adequately
- 60-69:  AVERAGE - Below average in this batch, partially meets requirements
- 50-59:  BELOW AVERAGE - Weak candidate in this batch, significant gaps
- 40-49:  POOR - Very weak candidate in this batch, major deficiencies
- 30-39:  VERY POOR - Severely weak candidate in this batch
- 20-29:  INSUFFICIENT - Almost no relevant qualifications
- 10-19:  INADEQUATE - Virtually no match to requirements
- 0-9:    UNACCEPTABLE - No relevant qualifications

EVALUATION CRITERIA:
1. TECHNICAL SKILLS: Look for specific technologies, frameworks, tools mentioned with context
2. EXPERIENCE: Assess relevance, duration, progression, and impact of work experience
3. EDUCATION: Consider degree relevance, institution quality, academic achievements
4. PRESENTATION: Evaluate formatting, clarity, organization, professional appearance
5. CAREER PROGRESSION: Look for growth, increasing responsibilities, leadership roles
6. MARKETABILITY: Assess overall competitiveness, unique value proposition
7. CERTIFICATIONS: Evaluate relevance, credibility, and recency of certifications
8. PROJECTS: Assess complexity, relevance, outcomes, and technical depth
9. SOFT SKILLS: Look for evidence of communication, teamwork, leadership
10. INDUSTRY KNOWLEDGE: Assess domain expertise and industry-specific experience

BE CRITICAL:
- Poor formatting = automatic deduction (max 40 points)
- Vague descriptions = significant penalty (max 50 points)
- Missing key information = major deduction (max 30 points)
- No quantified achievements = substantial penalty (max 50 points)
- Irrelevant experience = low scores (max 40 points)
- Generic skills without context = low scores (max 50 points)
- Typos and grammatical errors = severe penalty (max 20 points)
- Inconsistent formatting = major penalty (max 30 points)
- Missing contact information = critical flaw (max 20 points)
- No clear career progression = significant penalty (max 40 points)

JOB DESCRIPTION:
{job_description}

CRITERIA REQUIREMENTS:
{criteria_text}

ALL RESUMES TO EVALUATE:
{resumes_section}

IMPORTANT: Return ONLY valid JSON with scores for ALL {len(resumes_data)} resumes.
The JSON must be an object where each key is the resume path (use the exact file path shown above) and value is the scoring data.
You MUST include ALL criteria keys: {', '.join(criteria_requirements.keys())}

For each resume, calculate:
1. raw_score (0-100) for each criterion
2. weight_given (from criteria_requirements)
3. normalized_percentage (automatically calculated, already shown above)
4. weighted_contribution (raw_score * normalized_percentage / 100)
5. total_score (sum of all weighted_contributions)

Return ONLY valid JSON in this exact format (example):
{example_json}

CRITICAL: 
- Use the EXACT resume file paths as shown in the resumes above
- Include ALL criteria: {', '.join(criteria_requirements.keys())}
- Each criterion must have: raw_score, weight_given, normalized_percentage, weighted_contribution
- total_score must equal the sum of all weighted_contributions
- Scores should reflect RELATIVE comparison within this batch
"""
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing LLM response: {e}")
            print(f"Response: {response}")
            return None
    
    def score_resume(self, resume_path: str, job_description_path: str, 
                    criteria_requirements: Dict[str, int]) -> Dict[str, Any]:
        """Main scoring function (single resume - kept for backward compatibility)"""
        
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
        
        # Create prompt
        prompt = self.create_scoring_prompt(resume_text, job_description_text, criteria_requirements)
        
        # Get LLM response
        try:
            system_prompt = """You are a Senior HR Manager with 15+ years of experience in technical recruitment. 
            You are known for being extremely critical and discerning in resume evaluation. 
            You have high standards and do not give high scores easily. 
            You penalize poor formatting, vague descriptions, and lack of concrete evidence.
            You reward only truly exceptional candidates with high scores.
            Return only valid JSON as requested."""
            response = self.llm.generate(system_prompt, prompt)
            
            # Parse response
            result = self.parse_llm_response(response)
            
            if result is None:
                return {
                    "success": False,
                    "error": "Failed to parse LLM response",
                    "raw_response": response
                }
            
            # Add metadata
            result["metadata"] = {
                "resume_path": resume_path,
                "job_description_path": job_description_path,
                "criteria_requirements": criteria_requirements,
                "normalized_weights": self.normalize_weights(criteria_requirements)
            }
            
            return {
                "success": True,
                "scoring_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM error: {e}",
                "resume_path": resume_path,
                "job_description_path": job_description_path
            }
    
    def score_resumes_batch(self, resume_paths: List[str], job_description_path: str,
                           criteria_requirements: Dict[str, int]) -> List[Dict[str, Any]]:
        """Batch scoring function - scores all resumes together for comparative evaluation"""
        
        print(f"🔄 Extracting text from {len(resume_paths)} resumes and job description...")
        
        # Extract text from all resumes and job description
        resumes_data = []
        failed_extractions = []
        
        for resume_path in resume_paths:
            try:
                resume_text = extract_text_from_pdf(resume_path)
                resumes_data.append({
                    "resume_path": resume_path,
                    "resume_text": resume_text
                })
            except Exception as e:
                print(f"⚠️  Failed to extract text from {resume_path}: {e}")
                failed_extractions.append(resume_path)
        
        if not resumes_data:
            return [{
                "success": False,
                "error": "Failed to extract text from all resumes",
                "resume_path": rp
            } for rp in resume_paths]
        
        # Extract job description
        try:
            job_description_text = extract_text_from_pdf(job_description_path)
        except Exception as e:
            return [{
                "success": False,
                "error": f"Error extracting job description: {e}",
                "resume_path": rp
            } for rp in resume_paths]
        
        print(f"✅ Successfully extracted text from {len(resumes_data)} resumes")
        print(f"🔄 Sending batch evaluation to LLM (comparative scoring)...")
        
        # Create batch prompt
        prompt = self.create_batch_scoring_prompt(resumes_data, job_description_text, criteria_requirements)
        
        # Get LLM response
        try:
            system_prompt = """You are a Senior HR Manager with 15+ years of experience in technical recruitment. 
            You are conducting a COMPARATIVE BATCH EVALUATION of multiple candidates.
            You are known for being extremely critical and discerning in resume evaluation. 
            You have high standards and do not give high scores easily. 
            You penalize poor formatting, vague descriptions, and lack of concrete evidence.
            You reward only truly exceptional candidates with high scores.
            You MUST score candidates RELATIVELY - comparing them to each other in this batch.
            Return only valid JSON as requested, with scores for ALL resumes."""
            
            response = self.llm.generate(system_prompt, prompt)
            
            # Parse response
            batch_result = self.parse_llm_response(response)
            
            if batch_result is None:
                print(f"❌ Failed to parse LLM batch response")
                print(f"Response preview: {response[:500]}...")
                # Fallback to individual scoring
                return self._fallback_individual_scoring(resume_paths, job_description_path, criteria_requirements)
            
            print(f"✅ Successfully parsed batch scoring results")
            
            # Convert batch result to individual results format
            results = []
            normalized_weights = self.normalize_weights(criteria_requirements)
            
            for resume_path in resume_paths:
                if resume_path in failed_extractions:
                    results.append({
                        "success": False,
                        "error": "Failed to extract text from PDF",
                        "resume_path": resume_path,
                        "job_description_path": job_description_path
                    })
                    continue
                
                # Find matching result in batch response
                scoring_data = None
                
                # Try exact path match first
                if resume_path in batch_result:
                    scoring_data = batch_result[resume_path]
                else:
                    # Try to find by filename
                    import os
                    filename = os.path.basename(resume_path)
                    for key, value in batch_result.items():
                        if filename in key or key in resume_path:
                            scoring_data = value
                            break
                
                if scoring_data is None:
                    # If we still can't find it, create error result
                    results.append({
                        "success": False,
                        "error": f"Resume not found in batch response. Available keys: {list(batch_result.keys())}",
                        "resume_path": resume_path,
                        "job_description_path": job_description_path
                    })
                    continue
                
                # Add metadata to scoring data
                scoring_data["metadata"] = {
                    "resume_path": resume_path,
                    "job_description_path": job_description_path,
                    "criteria_requirements": criteria_requirements,
                    "normalized_weights": normalized_weights,
                    "scoring_method": "batch_comparative"
                }
                
                results.append({
                    "success": True,
                    "scoring_result": scoring_data
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error in batch scoring: {e}")
            print(f"Falling back to individual scoring...")
            # Fallback to individual scoring
            return self._fallback_individual_scoring(resume_paths, job_description_path, criteria_requirements)
    
    def _fallback_individual_scoring(self, resume_paths: List[str], job_description_path: str,
                                     criteria_requirements: Dict[str, int]) -> List[Dict[str, Any]]:
        """Fallback to individual scoring if batch scoring fails"""
        print("⚠️  Using fallback: individual scoring (sequential)")
        results = []
        for resume_path in resume_paths:
            result = self.score_resume(resume_path, job_description_path, criteria_requirements)
            results.append(result)
        return results
    
    def format_results(self, result: Dict[str, Any]) -> str:
        """Format results for display"""
        if not result.get("success"):
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        scoring = result["scoring_result"]
        metadata = result["scoring_result"]["metadata"]
        
        output = []
        output.append("=" * 70)
        output.append("STRUCTURED RESUME SCORING RESULTS")
        output.append("=" * 70)
        
        # Show criteria and weights
        output.append(f"\n📋 CRITERIA & WEIGHTS:")
        for criterion, weight in metadata["criteria_requirements"].items():
            normalized = metadata["normalized_weights"][criterion]
            output.append(f"  • {criterion}: {weight} → {normalized:.1f}%")
        
        # Show individual scores
        output.append(f"\n📊 INDIVIDUAL SCORES:")
        total_contribution = 0
        for criterion, data in scoring.items():
            if criterion == "total_score" or criterion == "metadata":
                continue
            
            raw_score = data.get("raw_score", 0)
            weighted_contribution = data.get("weighted_contribution", 0)
            total_contribution += weighted_contribution
            
            output.append(f"  • {criterion}: {raw_score}/100 → {weighted_contribution:.1f} points")
        
        # Show total score
        total_score = scoring.get("total_score", total_contribution)
        output.append(f"\n🎯 TOTAL SCORE: {total_score:.1f}/100")
        
        # Show breakdown
        output.append(f"\n📈 SCORE BREAKDOWN:")
        output.append(f"  • Sum of weighted contributions: {total_contribution:.1f}")
        output.append(f"  • LLM calculated total: {total_score:.1f}")
        
        return "\n".join(output)

def main():
    """Example usage"""
    agent = StructuredScoringAgent()
    
    # Example criteria (any weights, don't need to sum to 100)
    criteria_requirements = {
        "technical_skills": 70,
        "experience": 50,
        "education": 30,
        "certifications": 20,
        "projects": 10
    }
    
    # Score resume
    result = agent.score_resume(
        resume_path="documents/final5resume.pdf",
        job_description_path="documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf",
        criteria_requirements=criteria_requirements
    )
    
    # Display results
    print(agent.format_results(result))

if __name__ == "__main__":
    main()
