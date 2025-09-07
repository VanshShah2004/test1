"""
Job Description Parser Module

This module processes PDF job descriptions and extracts structured job criteria
using CrewAI agents and Gemini API for intelligent parsing.

Author: AI Assistant
Version: 1.0
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, FileReadTool

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobDescriptionParserTool:
    """Tool for parsing and structuring job descriptions from PDF files"""
    
    def __init__(self):
        # Initialize Gemini API
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        
        logger.info("✅ JobDescriptionParserTool initialized with Gemini API")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text.strip():
                    raise ValueError("No text could be extracted from the PDF")
                
                logger.info(f"✅ Successfully extracted {len(text)} characters from PDF")
                return text.strip()
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {str(e)}")
            raise
    
    def structure_job_description(self, job_text: str) -> Dict:
        """Use Gemini LLM to extract structured information from job description text"""
        
        prompt = f"""
        You are an expert HR analyst. Analyze the following job description text and extract structured information.
        
        IMPORTANT: Return ONLY a valid JSON object with the following exact structure:
        
        {{
            "position_title": "exact job title from the document",
            "company_name": "company name if mentioned, otherwise null",
            "location": "job location if mentioned, otherwise null",
            "employment_type": "full-time/part-time/contract/internship, or null if not specified",
            "experience_level": "entry/junior/mid/senior/executive, or null if not clear",
            "department": "department or team if mentioned, otherwise null",
            "required_skills": ["list of must-have technical and soft skills"],
            "preferred_skills": ["list of nice-to-have or preferred skills"],
            "required_qualifications": ["education, certifications, experience requirements"],
            "preferred_qualifications": ["preferred education, certifications, experience"],
            "responsibilities": ["list of key job responsibilities and duties"],
            "min_experience_years": 0,
            "max_experience_years": null,
            "education_requirements": ["specific degree or education requirements"],
            "certifications": ["required or preferred certifications"],
            "industry": "industry sector like technology, finance, healthcare, etc.",
            "company_size": "startup/small/medium/large/enterprise, or null if unknown",
            "remote_work": true/false/null,
            "salary_range": "salary information if mentioned, otherwise null",
            "benefits": ["list of benefits mentioned"],
            "key_requirements": ["top 5-8 most critical requirements for this role"],
            "job_summary": "brief 2-3 sentence summary of the role and its purpose"
        }}
        
        EXTRACTION GUIDELINES:
        1. For required_skills: Include technical skills, programming languages, tools, frameworks
        2. For preferred_skills: Include nice-to-have skills that would be beneficial
        3. For min_experience_years: Look for phrases like "3+ years", "minimum 5 years", etc.
        4. For education_requirements: Look for degree requirements like "Bachelor's", "Master's", etc.
        5. For industry: Determine from company description or role context
        6. For remote_work: true if remote/hybrid mentioned, false if on-site required, null if not specified
        7. Extract information as accurately as possible from the text provided
        8. If information is not available or unclear, use null for strings/objects or empty arrays for lists
        
        Job Description Text:
        {job_text}
        
        Return only the JSON object, no additional text or explanations.
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert HR analyst specializing in job description analysis. Extract structured information accurately and return only valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            logger.info("🔍 Analyzing job description with Gemini 1.5 Flash...")
            response = self.llm.invoke(messages)
            
            # Clean up the response to extract JSON
            content = response.content.strip()
            
            # Remove any markdown formatting
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            # Parse JSON
            structured_data = json.loads(content)
            logger.info("✅ Successfully structured job description data")
            
            return structured_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON response: {str(e)}")
            logger.error(f"Raw response: {response.content[:500]}...")
            raise ValueError(f"Invalid JSON response from AI: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Error structuring job description: {str(e)}")
            raise


class JobDescriptionAgent:
    """CrewAI Agent for processing and analyzing job descriptions"""
    
    def __init__(self):
        self.parser_tool = JobDescriptionParserTool()
        self.agent = self._create_agent()
        logger.info("✅ JobDescriptionAgent initialized")
    
    def _create_agent(self):
        """Create the job description processing agent"""
        return Agent(
            role="Senior Job Description Analysis Specialist",
            goal="Extract, validate, and optimize job descriptions to create comprehensive hiring criteria that enable effective candidate screening",
            backstory="""You are a world-class HR analyst with 15+ years of experience in talent acquisition 
            and job description optimization. You have deep expertise in understanding both explicit and 
            implicit job requirements, industry standards, and what makes candidates successful in different roles. 
            You excel at translating verbose, marketing-heavy job postings into clear, actionable hiring criteria 
            that help recruiters make better decisions. You understand the nuances of different industries, 
            company sizes, and role levels, and can provide insights that go beyond surface-level requirements.""",
            tools=[],
            verbose=True,
            allow_delegation=False,
            llm=self.parser_tool.llm,
            model_name="gemini-1.5-flash"
        )
    
    def analyze_job_description(self, structured_jd: Dict) -> str:
        """Use LLM to analyze and enhance job description"""
        
        prompt = f"""
        Analyze the following structured job description data and provide comprehensive insights:
        
        {json.dumps(structured_jd, indent=2)}
        
        Your analysis should include:
        
        1. **Completeness Assessment**: Rate the job description completeness (1-10) and identify missing elements
        2. **Requirement Clarity**: Assess how clear and specific the requirements are
        3. **Market Competitiveness**: Evaluate if requirements are realistic for the market
        4. **Red Flags**: Identify any concerning patterns or unrealistic expectations
        5. **Enhancement Suggestions**: Recommend improvements to attract better candidates
        6. **Screening Priority**: Rank the most important criteria for candidate evaluation
        7. **Interview Focus Areas**: Suggest key areas to probe during interviews
        
        Provide actionable insights that will help in effective candidate screening and hiring decisions.
        Focus on practical recommendations rather than generic advice.
        """
        
        try:
            logger.info("🧠 Running job description analysis...")
            messages = [
                SystemMessage(content="You are an expert HR analyst specializing in job description analysis. Provide comprehensive insights and actionable recommendations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.parser_tool.llm.invoke(messages)
            logger.info("✅ Job description analysis completed")
            return response.content
            
        except Exception as e:
            logger.error(f"❌ Error in job description analysis: {str(e)}")
            return f"Analysis failed: {str(e)}"


class JobDescriptionParser:
    """Main class for parsing job descriptions and converting to resume screening format"""
    
    def __init__(self):
        try:
            self.agent = JobDescriptionAgent()
            logger.info("✅ JobDescriptionParser initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize JobDescriptionParser: {e}")
            raise
    
    def parse_job_description_pdf(self, pdf_path: str) -> Dict:
        """
        Main function to parse job description PDF and return structured data
        
        Args:
            pdf_path (str): Path to the job description PDF file
            
        Returns:
            Dict: Complete parsing results with structured data and analysis
        """
        
        try:
            logger.info(f"🚀 Starting job description parsing for: {pdf_path}")
            
            # Step 1: Extract text from PDF
            logger.info("📄 Step 1: Extracting text from PDF...")
            raw_text = self.agent.parser_tool.extract_text_from_pdf(pdf_path)
            
            # Step 2: Structure the job description
            logger.info("🔍 Step 2: Structuring job description with AI...")
            structured_jd = self.agent.parser_tool.structure_job_description(raw_text)
            
            # Step 3: Analyze with CrewAI agent
            logger.info("🧠 Step 3: Analyzing with CrewAI agent...")
            agent_analysis = self.agent.analyze_job_description(structured_jd)
            
            # Step 4: Convert to resume screening format
            logger.info("🎯 Step 4: Converting to resume screening format...")
            screening_criteria = self._convert_to_screening_format(structured_jd)
            
            result = {
                "success": True,
                "file_path": pdf_path,
                "raw_text": raw_text,
                "structured_data": structured_jd,
                "agent_analysis": agent_analysis,
                "screening_criteria": screening_criteria,
                "processing_timestamp": datetime.now().isoformat(),
                "parser_version": "1.0"
            }
            
            logger.info("✅ Job description parsing completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing job description: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": pdf_path,
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def _convert_to_screening_format(self, structured_jd: Dict) -> Dict:
        """Convert structured job description to the exact format expected by resume screener"""
        
        return {
            'position': structured_jd.get('position_title', 'Unknown Position'),
            'required_skills': structured_jd.get('required_skills', []),
            'preferred_skills': structured_jd.get('preferred_skills', []),
            'min_experience_years': structured_jd.get('min_experience_years', 0),
            'education_level': self._map_education_level(structured_jd.get('education_requirements', [])),
            'industry': structured_jd.get('industry', 'general'),
            'company_size': self._map_company_size(structured_jd.get('company_size')),
            'remote_work': structured_jd.get('remote_work', False) if structured_jd.get('remote_work') is not None else False
        }
    
    def _map_education_level(self, education_requirements: List) -> str:
        """Map education requirements to expected format"""
        if not education_requirements:
            return 'bachelors'
        
        education_text = ' '.join(str(req) for req in education_requirements).lower()
        
        if any(term in education_text for term in ['phd', 'doctorate', 'doctoral', 'ph.d']):
            return 'doctorate'
        elif any(term in education_text for term in ['master', 'mba', 'ms', 'ma', "master's"]):
            return 'masters'
        elif any(term in education_text for term in ['bachelor', 'bs', 'ba', 'degree', "bachelor's"]):
            return 'bachelors'
        elif any(term in education_text for term in ['high school', 'diploma', 'ged']):
            return 'high_school'
        else:
            return 'bachelors'  # Default assumption
    
    def _map_company_size(self, company_size: str) -> str:
        """Map company size to expected format"""
        if not company_size:
            return 'medium'
        
        size_lower = company_size.lower()
        
        if 'startup' in size_lower:
            return 'startup'
        elif any(term in size_lower for term in ['small', 'sme']):
            return 'small'
        elif any(term in size_lower for term in ['large', 'enterprise', 'fortune']):
            return 'large'
        else:
            return 'medium'  # Default


def main(job_description_pdf_path: str) -> Dict:
    """
    Main function to be called from other modules
    
    Args:
        job_description_pdf_path (str): Path to the job description PDF
        
    Returns:
        Dict: Job criteria in the format expected by resume screener
    """
    
    try:
        # Initialize parser
        parser = JobDescriptionParser()
        
        # Parse the job description
        result = parser.parse_job_description_pdf(job_description_pdf_path)
        
        if result.get("success"):
            # Return the screening criteria in the expected format
            return result["screening_criteria"]
        else:
            logger.error(f"Failed to parse job description: {result.get('error')}")
            # Return a default job criteria structure
            return {
                'position': 'General Position',
                'required_skills': [],
                'preferred_skills': [],
                'min_experience_years': 0,
                'education_level': 'bachelors',
                'industry': 'general',
                'company_size': 'medium',
                'remote_work': False
            }
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        # Return default structure on error
        return {
            'position': 'General Position',
            'required_skills': [],
            'preferred_skills': [],
            'min_experience_years': 0,
            'education_level': 'bachelors',
            'industry': 'general',
            'company_size': 'medium',
            'remote_work': False
        }


if __name__ == "__main__":
    """
    Test the job description parser independently
    """
    
    print("=" * 60)
    print("JOB DESCRIPTION PARSER - STANDALONE TEST")
    print("=" * 60)
    
    # Test with a sample PDF
    test_pdf_path = "Bottomline_ Intern + FTE - JD 2026 Batch.pdf"  # Update this path
    
    if os.path.exists(test_pdf_path):
        print(f"📄 Testing with: {test_pdf_path}")
        
        result = main(test_pdf_path)
        
        print("\n📊 EXTRACTED JOB CRITERIA:")
        print(json.dumps(result, indent=2))
        
    else:
        print(f"❌ Test PDF not found: {test_pdf_path}")
        print("Please provide a valid job description PDF to test the parser")