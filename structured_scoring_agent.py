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
    def __init__(self, model: str = "gemini-1.5-flash"):
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
        """Main scoring function"""
        
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
