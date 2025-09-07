import io
import sys
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# Suppress deprecation warnings from third-party libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*swigvarlink.*")
warnings.filterwarnings("ignore", message=".*swig.*")

# Also suppress warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'


import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import PrivateAttr
import logging

# Document processing libraries
import PyPDF2
import docx2txt
import pdfplumber
import fitz  # PyMuPDF for better PDF parsing

# NLP and ML libraries
import spacy
from spacy.matcher import Matcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
# OpenAI removed per request
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # Optional dependency; fallback imported conditionally
    ChatGoogleGenerativeAI = None

# Import job description parser
try:
    from jobDescriptionParser import main as parse_job_description
except ImportError:
    print("Warning: jobDescriptionParser not found. Job description parsing will be disabled.")
    parse_job_description = None

# Minimal direct Gemini adapter (no LangChain dependency required)
class _SimpleGeminiClient:
    """Lightweight adapter exposing .invoke(messages) -> object with .content for Gemini."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._genai = genai
        self._model = genai.GenerativeModel(model)
    
    class _Resp:
        def __init__(self, text: str):
            self.content = text
    
    def invoke(self, messages: List):
        # Combine messages into a single prompt string
        parts = []
        for m in messages:
            role = getattr(m, 'type', None) or m.__class__.__name__
            content = getattr(m, 'content', str(m))
            parts.append(f"{role.upper()}:\n{content}\n")
        prompt = "\n".join(parts)
        try:
            res = self._model.generate_content(prompt, generation_config={"temperature": self._temperature})
            text = getattr(res, 'text', None)
            if not text and hasattr(res, 'candidates') and res.candidates:
                text = res.candidates[0].content.parts[0].text
            return _SimpleGeminiClient._Resp(text or "")
        except Exception as e:
            raise e
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResumeParserTool(BaseTool):
    name: str = "Resume Parser"
    description: str = "Extracts and parses content from PDF and Word documents"
    
    def _run(self, file_path: str) -> Dict:
        """Parse resume from file and extract structured information"""
        try:
            # Determine file type and extract text
            text = self._extract_text_from_file(file_path)
            
            if not text.strip():
                return {"error": "Could not extract text from file", "content": ""}
            
            # Parse structured information
            parsed_data = {
                "raw_text": text,
                "contact_info": self._extract_contact_info(text),
                "education": self._extract_education(text),
                "experience": self._extract_experience(text),
                "skills": self._extract_skills(text),
                "certifications": self._extract_certifications(text),
                "projects": self._extract_projects(text),
                "languages": self._extract_languages(text),
                "achievements": self._extract_achievements(text)
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            return {"error": str(e), "content": ""}
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or Word document"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_from_word(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods for better accuracy"""
        text = ""
        
        # Method 1: PyMuPDF (fitz) - better for complex layouts
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: pdfplumber - good for tables and structured data
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 3: PyPDF2 - fallback
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        return text
    
    def _extract_from_word(self, file_path: Path) -> str:
        """Extract text from Word document"""
        try:
            return docx2txt.process(str(file_path))
        except Exception as e:
            logger.error(f"Error extracting from Word document: {e}")
            return ""
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information using regex patterns"""
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Phone pattern (various formats)
        phone_patterns = [
            r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(\+\d{1,3}[-.\s]?)?\d{10}',
            r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info['phone'] = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
                break
        
        # LinkedIn profile
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.search(linkedin_pattern, text, re.IGNORECASE)
        contact_info['linkedin'] = linkedin.group(0) if linkedin else None
        
        # GitHub profile
        github_pattern = r'github\.com/[\w-]+'
        github = re.search(github_pattern, text, re.IGNORECASE)
        contact_info['github'] = github.group(0) if github else None
        
        return contact_info
    
    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education information"""
        education = []
        
        # Degree patterns
        degree_patterns = [
            r'(Bachelor|Master|PhD|Ph\.D|B\.S|B\.A|M\.S|M\.A|MBA|B\.Tech|M\.Tech|B\.E|M\.E|B\.Sc|M\.Sc).*?(?=\n|\.|,|$)',
            r'(Undergraduate|Graduate|Doctorate).*?(?=\n|\.|,|$)'
        ]
        
        # University/College patterns
        institution_keywords = ['University', 'College', 'Institute', 'School', 'Academy']
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for degree mentions
            for pattern in degree_patterns:
                degree_match = re.search(pattern, line, re.IGNORECASE)
                if degree_match:
                    edu_entry = {
                        'degree': degree_match.group(0).strip(),
                        'institution': None,
                        'year': None,
                        'gpa': None
                    }
                    
                    # Look for institution in nearby lines
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        if any(keyword in lines[j] for keyword in institution_keywords):
                            edu_entry['institution'] = lines[j].strip()
                            break
                    
                    # Look for graduation year
                    year_pattern = r'(19|20)\d{2}'
                    year_matches = re.findall(year_pattern, line + ' ' + lines[min(i+1, len(lines)-1)])
                    if year_matches:
                        edu_entry['year'] = year_matches[-1]  # Take the latest year
                    
                    # Look for GPA
                    gpa_pattern = r'GPA[:\s]*(\d+\.?\d*)'
                    gpa_match = re.search(gpa_pattern, line + ' ' + lines[min(i+1, len(lines)-1)], re.IGNORECASE)
                    if gpa_match:
                        edu_entry['gpa'] = gpa_match.group(1)
                    
                    education.append(edu_entry)
                    break
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience information"""
        experience = []
        
        # Common job title keywords
        job_keywords = [
            'Engineer', 'Developer', 'Manager', 'Analyst', 'Consultant', 'Specialist',
            'Director', 'Lead', 'Senior', 'Junior', 'Associate', 'Intern', 'Coordinator',
            'Administrator', 'Designer', 'Architect', 'Scientist', 'Researcher'
        ]
        
        # Date patterns for employment
        date_patterns = [
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(19|20)\d{2}',
            r'(19|20)\d{2}\s*-\s*(19|20)\d{2}',
            r'(19|20)\d{2}\s*to\s*(19|20)\d{2}|present',
            r'\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4}'
        ]
        
        lines = text.split('\n')
        current_exp = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for job titles
            for keyword in job_keywords:
                if keyword.lower() in line.lower() and len(line.split()) <= 6:
                    if current_exp:
                        experience.append(current_exp)
                    
                    current_exp = {
                        'title': line,
                        'company': None,
                        'duration': None,
                        'description': []
                    }
                    
                    # Look for company name in next few lines
                    for j in range(i+1, min(i+4, len(lines))):
                        if lines[j].strip() and not any(pattern in lines[j] for pattern in date_patterns):
                            current_exp['company'] = lines[j].strip()
                            break
                    
                    # Look for duration
                    for j in range(max(0, i-1), min(i+3, len(lines))):
                        for pattern in date_patterns:
                            if re.search(pattern, lines[j], re.IGNORECASE):
                                current_exp['duration'] = lines[j].strip()
                                break
                    break
            
            # Add description lines to current experience
            if current_exp and line.startswith(('•', '-', '*')) and len(line) > 10:
                current_exp['description'].append(line)
        
        if current_exp:
            experience.append(current_exp)
        
        return experience
    
    def _extract_skills(self, text: str) -> Dict:
        """Extract skills categorized by type"""
        skills = {
            'technical': [],
            'programming_languages': [],
            'frameworks': [],
            'databases': [],
            'tools': [],
            'soft_skills': []
        }
        
        # Define skill categories
        skill_categories = {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'Go', 'Rust',
                'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS'
            ],
            'frameworks': [
                'React', 'Angular', 'Vue', 'Django', 'Flask', 'Spring', 'Express', 'Node.js',
                'Laravel', 'Rails', 'Bootstrap', 'jQuery', 'TensorFlow', 'PyTorch', 'Keras'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'Oracle', 'SQLite',
                'MariaDB', 'DynamoDB', 'Neo4j', 'InfluxDB'
            ],
            'tools': [
                'Git', 'Docker', 'Kubernetes', 'Jenkins', 'AWS', 'Azure', 'GCP', 'Terraform',
                'Ansible', 'Linux', 'Windows', 'macOS', 'Jira', 'Confluence', 'Slack'
            ],
            'soft_skills': [
                'Leadership', 'Communication', 'Problem-solving', 'Teamwork', 'Project Management',
                'Critical Thinking', 'Adaptability', 'Time Management', 'Creativity', 'Analytics'
            ]
        }
        
        text_lower = text.lower()
        
        for category, skill_list in skill_categories.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills[category].append(skill)
        
        # Extract additional technical skills from common patterns
        skill_sections = re.findall(r'(?i)(skills?|technologies?|competencies)[\s\S]*?(?=\n\n|\n[A-Z]|$)', text)
        for section in skill_sections:
            # Remove common separators and split
            clean_section = re.sub(r'[•\-\*\n]', ' ', section)
            potential_skills = re.split(r'[,;|]', clean_section)
            
            for skill in potential_skills:
                skill = skill.strip()
                if len(skill) > 2 and len(skill) < 30 and not skill.lower().startswith(('skill', 'tech')):
                    skills['technical'].append(skill)
        
        # Remove duplicates and empty entries
        for category in skills:
            skills[category] = list(set([s for s in skills[category] if s]))
        
        return skills
    
    def _extract_certifications(self, text: str) -> List[Dict]:
        """Extract certification information"""
        certifications = []
        
        cert_keywords = [
            'certified', 'certification', 'certificate', 'credential', 'license', 'accredited'
        ]
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in cert_keywords):
                # Extract year if present
                year_match = re.search(r'(19|20)\d{2}', line)
                year = year_match.group(0) if year_match else None
                
                certifications.append({
                    'name': line.strip(),
                    'year': year
                })
        
        return certifications
    
    def _extract_projects(self, text: str) -> List[Dict]:
        """Extract project information"""
        projects = []
        
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented']
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in project_keywords) and len(line) > 20:
                project = {
                    'name': line.strip(),
                    'description': []
                }
                
                # Look for description in following lines
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip() and lines[j].startswith(('•', '-', '*')):
                        project['description'].append(lines[j].strip())
                
                projects.append(project)
        
        return projects
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract spoken languages"""
        languages = []
        
        language_names = [
            'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian',
            'Chinese', 'Mandarin', 'Japanese', 'Korean', 'Arabic', 'Hindi', 'Bengali',
            'Urdu', 'Turkish', 'Dutch', 'Swedish', 'Norwegian', 'Danish'
        ]
        
        text_lower = text.lower()
        for lang in language_names:
            if lang.lower() in text_lower:
                languages.append(lang)
        
        return list(set(languages))
    
    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements and awards"""
        achievements = []
        
        achievement_keywords = [
            'award', 'achievement', 'recognition', 'honor', 'medal', 'prize', 'winner',
            'achieved', 'accomplished', 'recognized', 'honored'
        ]
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in achievement_keywords):
                achievements.append(line.strip())
        
        return achievements


class LLMAnalysisTool(BaseTool):
    name: str = "LLM Analysis"
    description: str = "Uses LLM to analyze resume content and provide intelligent insights"
    _llm: object = PrivateAttr(default=None)
    
    @property
    def llm(self):
        return self._llm
    
    def __init__(self, llm=None):
        # Initialize pydantic BaseModel field via super().__init__ to avoid "no field" errors
        resolved_llm = llm or self._get_default_llm()
        super().__init__()
        self._llm = resolved_llm
        
    def _get_default_llm(self):
        """Get default LLM"""
        # Try Gemini first (primary)
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                if ChatGoogleGenerativeAI is not None:
                    return ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        temperature=0.2,
                        google_api_key=gemini_api_key
                    )
                else:
                    # Fallback to direct SDK
                    return _SimpleGeminiClient(api_key=gemini_api_key, model="gemini-1.5-flash", temperature=0.2)
            except Exception as e:
                logger.warning(f"Gemini init failed, will try OpenAI fallback: {e}")

        # If we get here, no provider could be initialized
        provider_hint = (
            "Set GEMINI_API_KEY in your environment/.env."
        )
        raise ValueError(f"No LLM provider available. {provider_hint}")
    
    def _get_gemini_llm(self):
        """Create a Gemini LLM instance if possible."""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return None
        # Try LangChain's Gemini first
        if ChatGoogleGenerativeAI is not None:
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.2,
                    google_api_key=gemini_api_key
                )
            except Exception as e:
                logger.warning(f"LangChain Gemini init failed, will try direct SDK: {e}")
        # Fallback to direct Google SDK adapter
        try:
            return _SimpleGeminiClient(api_key=gemini_api_key, model="gemini-1.5-flash", temperature=0.2)
        except Exception as e:
            logger.warning(f"Direct Gemini SDK init failed: {e}")
        return None

    def _get_gemini_flash_llm(self):
        """Fallback to lower-cost/lower-quota Gemini model to avoid free-tier quota caps."""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return None
        if ChatGoogleGenerativeAI is not None:
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.2,
                    google_api_key=gemini_api_key
                )
            except Exception:
                pass
        try:
            return _SimpleGeminiClient(api_key=gemini_api_key, model="gemini-1.5-flash", temperature=0.2)
        except Exception:
            return None
    
    def _safe_invoke(self, messages):
        """Invoke the LLM; on Gemini quota/429 errors, failover to Gemini Flash."""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            err_text = str(e)
            is_quota_or_rate = (
                '429' in err_text or 'insufficient_quota' in err_text or 'rate limit' in err_text.lower()
            )
            # Failover: switch to Gemini Flash (if not already using it)
            if is_quota_or_rate:
                gemini_flash = self._get_gemini_flash_llm()
                if gemini_flash is not None and self.llm is not gemini_flash:
                    logger.warning("Gemini call failed with quota/429. Switching to gemini-1.5-flash and retrying once.")
                    self._llm = gemini_flash
                    return self.llm.invoke(messages)
            # If not quota/rate errors or no alternative, re-raise
            raise

    def _run(self, parsed_resume: Dict, job_criteria: Dict = None) -> Dict:
        """Analyze resume using LLM"""
        try:
            # Generate comprehensive analysis using LLM
            analysis = {
                'overall_assessment': self._analyze_overall_resume(parsed_resume, job_criteria),
                'technical_skills_analysis': self._analyze_technical_skills(parsed_resume),
                'experience_analysis': self._analyze_experience(parsed_resume),
                'education_analysis': self._analyze_education(parsed_resume),
                'strengths_weaknesses': self._analyze_strengths_weaknesses(parsed_resume),
                'scoring': self._generate_scores(parsed_resume, job_criteria),
                'recommendations': self._generate_recommendations(parsed_resume, job_criteria),
                'hiring_decision': self._generate_hiring_decision(parsed_resume, job_criteria)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_overall_resume(self, resume: Dict, job_criteria: Dict = None) -> str:
        """Generate overall resume assessment using LLM"""
        
        prompt_template = """
        You are an expert HR professional analyzing a resume. Provide an overall assessment as a structured, bullet-point report by section, not paragraphs.

        RESUME DATA:
        Contact Info: {contact_info}
        Education: {education}
        Experience: {experience}
        Skills: {skills}
        Certifications: {certifications}
        Projects: {projects}
        Achievements: {achievements}

        JOB CRITERIA (if provided): {job_criteria}

        Format the response EXACTLY with these top-level headings and concise bullet points under each:
        - SUMMARY
          - First impression
          - Resume quality & clarity
          - Market competitiveness
        - SKILLS
          - Strengths
          - Gaps vs job criteria
          - Notable tools/technologies
        - EXPERIENCE
          - Relevance to role
          - Impact highlights
          - Career trajectory
        - PROJECTS
          - Most relevant projects
          - Outcomes/metrics (if any)
        - EDUCATION
          - Degree(s) relevance
          - Notable achievements
        - CERTIFICATIONS
          - Relevant credentials
        - ACHIEVEMENTS
          - Awards/recognitions

        Constraints:
        - Keep each bullet to a single sentence.
        - No prose paragraphs.
        - Be specific and actionable where possible.
        """
        
        messages = [
            SystemMessage(content="You are an expert HR professional with 15+ years of experience in talent acquisition and resume analysis."),
            HumanMessage(content=prompt_template.format(
                contact_info=resume.get('contact_info', {}),
                education=resume.get('education', []),
                experience=resume.get('experience', []),
                skills=resume.get('skills', {}),
                certifications=resume.get('certifications', []),
                projects=resume.get('projects', []),
                achievements=resume.get('achievements', []),
                job_criteria=job_criteria or "No specific job criteria provided"
            ))
        ]
        
        response = self._safe_invoke(messages)
        return response.content
    
    def _analyze_technical_skills(self, resume: Dict) -> str:
        """Analyze technical skills using LLM"""
        
        skills = resume.get('skills', {})
        
        prompt_template = """
        Analyze the technical skills of this candidate:

        Programming Languages: {programming_languages}
        Frameworks: {frameworks}
        Databases: {databases}
        Tools: {tools}
        Technical Skills: {technical}

        Provide analysis on:
        1. Skill diversity and depth
        2. Modern vs legacy technologies
        3. Skill combinations and stack coherence
        4. Missing skills for typical roles
        5. Overall technical competency level

        Rate the technical skills on a scale of 1-100 and justify your rating.
        """
        
        messages = [
            SystemMessage(content="You are a technical hiring manager with expertise in evaluating technical skills across different domains."),
            HumanMessage(content=prompt_template.format(
                programming_languages=skills.get('programming_languages', []),
                frameworks=skills.get('frameworks', []),
                databases=skills.get('databases', []),
                tools=skills.get('tools', []),
                technical=skills.get('technical', [])
            ))
        ]
        
        response = self._safe_invoke(messages)
        return response.content
    
    def _analyze_experience(self, resume: Dict) -> str:
        """Analyze work experience using LLM"""
        
        experience = resume.get('experience', [])
        
        prompt_template = """
        Analyze the work experience of this candidate:

        WORK EXPERIENCE:
        {experience_details}

        Provide analysis on:
        1. Career progression and growth trajectory
        2. Quality of job descriptions and achievements
        3. Industry relevance and diversity
        4. Leadership and responsibility indicators
        5. Experience depth vs breadth

        Rate the experience quality on a scale of 1-100 and justify your rating.
        """
        
        experience_details = ""
        for i, exp in enumerate(experience, 1):
            experience_details += f"""
        Position {i}:
        - Title: {exp.get('title', 'Not specified')}
        - Company: {exp.get('company', 'Not specified')}
        - Duration: {exp.get('duration', 'Not specified')}
        - Responsibilities: {', '.join(exp.get('description', []))}
        """
        
        messages = [
            SystemMessage(content="You are an experienced HR director specializing in evaluating professional work experience and career development."),
            HumanMessage(content=prompt_template.format(experience_details=experience_details))
        ]
        
        response = self._safe_invoke(messages)
        return response.content
    
    def _analyze_education(self, resume: Dict) -> str:
        """Analyze educational background using LLM"""
        
        education = resume.get('education', [])
        
        prompt_template = """
        Analyze the educational background of this candidate:

        EDUCATION:
        {education_details}

        Provide analysis on:
        1. Educational level and relevance
        2. Institution quality and reputation (if known)
        3. Academic performance indicators
        4. Field of study alignment with career
        5. Additional learning and continuous education

        Rate the educational background on a scale of 1-100 and justify your rating.
        """
        
        education_details = ""
        for i, edu in enumerate(education, 1):
            education_details += f"""
        Education {i}:
        - Degree: {edu.get('degree', 'Not specified')}
        - Institution: {edu.get('institution', 'Not specified')}
        - Year: {edu.get('year', 'Not specified')}
        - GPA: {edu.get('gpa', 'Not specified')}
        """
        
        messages = [
            SystemMessage(content="You are an academic advisor and HR professional with expertise in evaluating educational qualifications."),
            HumanMessage(content=prompt_template.format(education_details=education_details))
        ]
        
        response = self._safe_invoke(messages)
        return response.content
    
    def _analyze_strengths_weaknesses(self, resume: Dict) -> Dict[str, List[str]]:
        """Identify strengths and weaknesses using LLM"""
        
        prompt_template = """
        Based on this resume data, identify the candidate's key strengths and weaknesses:

        COMPLETE RESUME DATA:
        {resume_summary}

        Provide:
        1. Top 5 strengths of this candidate
        2. Top 5 areas for improvement/weaknesses
        3. Unique selling points that make this candidate stand out
        4. Red flags or concerns (if any)

        Format your response as:
        STRENGTHS:
        - [strength 1]
        - [strength 2]
        ...

        WEAKNESSES:
        - [weakness 1]
        - [weakness 2]
        ...

        UNIQUE SELLING POINTS:
        - [point 1]
        - [point 2]
        ...

        RED FLAGS:
        - [flag 1] (if any)
        ...
        """
        
        # Create resume summary
        resume_summary = f"""
        Contact: {resume.get('contact_info', {})}
        Education: {len(resume.get('education', []))} entries
        Experience: {len(resume.get('experience', []))} positions
        Skills: {sum(len(v) for v in resume.get('skills', {}).values())} total skills
        Certifications: {len(resume.get('certifications', []))} certifications
        Projects: {len(resume.get('projects', []))} projects
        Achievements: {len(resume.get('achievements', []))} achievements
        
        Key Skills: {', '.join(resume.get('skills', {}).get('programming_languages', [])[:5])}
        """
        
        messages = [
            SystemMessage(content="You are a senior talent acquisition specialist with expertise in candidate assessment and evaluation."),
            HumanMessage(content=prompt_template.format(resume_summary=resume_summary))
        ]
        
        response = self._safe_invoke(messages)
        
        # Parse the response to extract structured data
        content = response.content
        strengths = []
        weaknesses = []
        
        # Extract strengths
        if "STRENGTHS:" in content:
            strengths_section = content.split("STRENGTHS:")[1].split("WEAKNESSES:")[0]
            strengths = [line.strip().lstrip('- ') for line in strengths_section.split('\n') if line.strip().startswith('-')]
        
        # Extract weaknesses
        if "WEAKNESSES:" in content:
            weaknesses_section = content.split("WEAKNESSES:")[1]
            if "UNIQUE SELLING POINTS:" in weaknesses_section:
                weaknesses_section = weaknesses_section.split("UNIQUE SELLING POINTS:")[0]
            weaknesses = [line.strip().lstrip('- ') for line in weaknesses_section.split('\n') if line.strip().startswith('-')]
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "full_analysis": content
        }
    
    def _generate_scores(self, resume: Dict, job_criteria: Dict = None) -> Dict:
        """Generate detailed scores using LLM"""
        
        prompt_template = """
        As an expert HR analyst, provide numerical scores (1-100) for different aspects of this resume:

        RESUME DATA:
        {resume_data}

        JOB CRITERIA: {job_criteria}

        Provide scores for:
        1. Technical Skills (1-100)
        2. Experience Quality (1-100)
        3. Education Level (1-100)
        4. Resume Presentation (1-100)
        5. Career Progression (1-100)
        6. Overall Marketability (1-100)

        For each score, provide a 1-sentence justification.

        Format as:
        Technical Skills: [score]/100 - [justification]
        Experience Quality: [score]/100 - [justification]
        Education Level: [score]/100 - [justification]
        Resume Presentation: [score]/100 - [justification]
        Career Progression: [score]/100 - [justification]
        Overall Marketability: [score]/100 - [justification]

        Also provide an OVERALL SCORE as a weighted average.
        """
        
        resume_data = {
            'skills_count': sum(len(v) for v in resume.get('skills', {}).values()),
            'experience_count': len(resume.get('experience', [])),
            'education_count': len(resume.get('education', [])),
            'certifications_count': len(resume.get('certifications', [])),
            'projects_count': len(resume.get('projects', [])),
            'has_contact_info': bool(resume.get('contact_info', {}).get('email'))
        }
        
        messages = [
            SystemMessage(content="You are a quantitative HR analyst specializing in resume scoring and candidate evaluation metrics."),
            HumanMessage(content=prompt_template.format(
                resume_data=resume_data,
                job_criteria=job_criteria or "General technical role"
            ))
        ]
        
        response = self._safe_invoke(messages)
        
        # Parse scores from response
        content = response.content
        scores = {}
        
        score_patterns = [
            (r'Technical Skills:\s*(\d+)/100', 'technical_skills'),
            (r'Experience Quality:\s*(\d+)/100', 'experience'),
            (r'Education Level:\s*(\d+)/100', 'education'),
            (r'Resume Presentation:\s*(\d+)/100', 'presentation'),
            (r'Career Progression:\s*(\d+)/100', 'career_progression'),
            (r'Overall Marketability:\s*(\d+)/100', 'marketability'),
            (r'OVERALL SCORE.*?(\d+)', 'overall_score')
        ]
        
        for pattern, key in score_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                scores[key] = int(match.group(1))
        
        # Calculate overall score if not provided
        if 'overall_score' not in scores and len(scores) > 0:
            weights = {
                'technical_skills': 0.3,
                'experience': 0.25,
                'education': 0.15,
                'presentation': 0.1,
                'career_progression': 0.1,
                'marketability': 0.1
            }
            
            weighted_score = sum(scores.get(k, 0) * v for k, v in weights.items())
            scores['overall_score'] = int(weighted_score)
        
        scores['detailed_analysis'] = content
        return scores
    
    def _generate_recommendations(self, resume: Dict, job_criteria: Dict = None) -> str:
        """Generate improvement recommendations using LLM"""
        
        prompt_template = """
        Based on this resume analysis, provide specific, actionable recommendations for improvement:

        RESUME SUMMARY:
        - Skills: {skills_summary}
        - Experience: {experience_summary}
        - Education: {education_summary}
        - Certifications: {cert_count} certifications
        - Projects: {project_count} projects

        JOB MARKET CONTEXT: {job_criteria}

        Provide:
        1. Immediate improvements (can be done in 1-2 weeks)
        2. Medium-term improvements (1-3 months)
        3. Long-term career development suggestions (6+ months)
        4. Specific skills to acquire based on market trends
        5. Resume formatting and presentation improvements

        Be specific and actionable in your recommendations.
        """
        
        skills_summary = f"Total: {sum(len(v) for v in resume.get('skills', {}).values())}"
        experience_summary = f"{len(resume.get('experience', []))} positions"
        education_summary = f"{len(resume.get('education', []))} degrees/programs"
        
        messages = [
            SystemMessage(content="You are a career coach and resume improvement specialist with expertise in current job market trends."),
            HumanMessage(content=prompt_template.format(
                skills_summary=skills_summary,
                experience_summary=experience_summary,
                education_summary=education_summary,
                cert_count=len(resume.get('certifications', [])),
                project_count=len(resume.get('projects', [])),
                job_criteria=job_criteria or "General technology sector"
            ))
        ]
        
        response = self._safe_invoke(messages)
        return response.content
    
    def _generate_hiring_decision(self, resume: Dict, job_criteria: Dict = None) -> Dict:
        """Generate hiring decision and rationale using LLM"""
        
        prompt_template = """
        As a hiring manager, make a hiring decision for this candidate:

        CANDIDATE PROFILE:
        {candidate_summary}

        JOB REQUIREMENTS: {job_criteria}

        Provide:
        1. DECISION: HIRE / INTERVIEW / MAYBE / REJECT
        2. CONFIDENCE LEVEL: HIGH / MEDIUM / LOW
        3. DETAILED RATIONALE (2-3 paragraphs explaining your decision)
        4. INTERVIEW FOCUS AREAS (if recommending interview)
        5. SALARY EXPECTATION RANGE (if applicable)
        6. ONBOARDING CONSIDERATIONS (if hiring)

        Base your decision on:
        - Skills match with requirements
        - Experience relevance and quality
        - Growth potential
        - Cultural fit indicators
        - Market competitiveness
        """
        
        # Create candidate summary
        skills = resume.get('skills', {})
        candidate_summary = f"""
        Technical Skills: {', '.join(skills.get('programming_languages', [])[:5])}
        Frameworks: {', '.join(skills.get('frameworks', [])[:3])}
        Experience: {len(resume.get('experience', []))} positions
        Education: {len(resume.get('education', []))} degrees
        Certifications: {len(resume.get('certifications', []))}
        Projects: {len(resume.get('projects', []))}
        Contact Info Complete: {bool(resume.get('contact_info', {}).get('email'))}
        """
        
        messages = [
            SystemMessage(content="You are a senior hiring manager with 10+ years of experience in technical recruitment and candidate evaluation."),
            HumanMessage(content=prompt_template.format(
                candidate_summary=candidate_summary,
                job_criteria=job_criteria or "Senior Software Developer role in technology company"
            ))
        ]
        
        response = self._safe_invoke(messages)
        content = response.content
        
        # Parse decision
        decision = "MAYBE"  # default
        confidence = "MEDIUM"  # default
        
        decision_match = re.search(r'DECISION:\s*(HIRE|INTERVIEW|MAYBE|REJECT)', content, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
        
        confidence_match = re.search(r'CONFIDENCE LEVEL:\s*(HIGH|MEDIUM|LOW)', content, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).upper()
        
        return {
            'decision': decision,
            'confidence': confidence,
            'detailed_rationale': content,
            'recommendation_summary': f"{decision} with {confidence} confidence"
        }


class ResumeScreenerWorkflow:
    """Main workflow class for LLM-powered resume screening"""
    
    def __init__(self, llm=None):
        """Initialize the workflow with LLM"""
        self.parser_tool = ResumeParserTool()
        
        # Initialize the analysis tool with proper LLM handling
        try:
            self.analysis_tool = LLMAnalysisTool(llm=llm)
            logger.info("LLM Analysis Tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Analysis Tool: {e}")
            raise e
    
    def screen_resume(self, file_path: str, job_criteria: Dict = None) -> Dict:
        """Main method to screen a resume using LLM analysis"""
        try:
            logger.info(f"Starting LLM-powered resume screening for: {file_path}")
            
            # Step 1: Parse the resume
            logger.info("Parsing resume...")
            parsed_resume = self.parser_tool._run(file_path)
            
            if "error" in parsed_resume:
                return {
                    "success": False,
                    "error": parsed_resume["error"],
                    "file_path": file_path
                }
            
            # Step 2: LLM Analysis
            logger.info("Performing LLM analysis...")
            llm_analysis = self.analysis_tool._run(parsed_resume, job_criteria)
            
            if "error" in llm_analysis:
                return {
                    "success": False,
                    "error": llm_analysis["error"],
                    "file_path": file_path
                }
            
            # Step 3: Generate comprehensive report
            logger.info("Generating comprehensive screening report...")
            screening_report = self._generate_comprehensive_report(
                file_path, parsed_resume, llm_analysis, job_criteria
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "parsed_data": parsed_resume,
                "llm_analysis": llm_analysis,
                "screening_report": screening_report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error screening resume: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }
    def _generate_comprehensive_report(self, file_path: str, parsed_resume: Dict, 
                                     llm_analysis: Dict, job_criteria: Dict = None) -> Dict:
        """Generate comprehensive screening report"""
        
        # Extract key information for summary
        contact = parsed_resume.get('contact_info', {})
        skills = parsed_resume.get('skills', {})
        
        report = {
            "executive_summary": {
                "candidate_name": file_path.split('/')[-1].replace('.pdf', '').replace('.docx', ''),
                "email": contact.get('email', 'Not provided'),
                "phone": contact.get('phone', 'Not provided'),
                "linkedin": contact.get('linkedin', 'Not provided'),
                "overall_assessment": llm_analysis.get('overall_assessment', ''),
                "hiring_decision": llm_analysis.get('hiring_decision', {}),
                "key_scores": llm_analysis.get('scoring', {})
            },
            
            "detailed_analysis": {
                "technical_skills": llm_analysis.get('technical_skills_analysis', ''),
                "experience_quality": llm_analysis.get('experience_analysis', ''),
                "educational_background": llm_analysis.get('education_analysis', ''),
                "strengths_weaknesses": llm_analysis.get('strengths_weaknesses', {})
            },
            
            "recommendations": {
                "improvement_suggestions": llm_analysis.get('recommendations', ''),
                "interview_focus": self._extract_interview_focus(llm_analysis),
                "next_steps": self._generate_next_steps(llm_analysis)
            },
            
            "data_completeness": {
                "contact_info_complete": self._assess_contact_completeness(contact),
                "resume_sections_present": self._assess_resume_completeness(parsed_resume),
                "missing_elements": self._identify_missing_elements(parsed_resume)
            }
        }
        
        return report
    
    def _extract_interview_focus(self, llm_analysis: Dict) -> List[str]:
        """Extract interview focus areas from LLM analysis"""
        hiring_decision = llm_analysis.get('hiring_decision', {})
        rationale = hiring_decision.get('detailed_rationale', '')
        
        # Look for interview focus areas in the rationale
        focus_areas = []
        if 'INTERVIEW FOCUS' in rationale:
            focus_section = rationale.split('INTERVIEW FOCUS')[1]
            if 'SALARY' in focus_section:
                focus_section = focus_section.split('SALARY')[0]
            
            # Extract bullet points or numbered items
            lines = focus_section.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                    focus_areas.append(line.lstrip('-•* ').lstrip('1234567890. '))
        
        return focus_areas[:5]  # Top 5 focus areas
    
    def _generate_next_steps(self, llm_analysis: Dict) -> List[str]:
        """Generate next steps based on LLM analysis"""
        hiring_decision = llm_analysis.get('hiring_decision', {})
        decision = hiring_decision.get('decision', 'MAYBE')
        
        if decision == 'HIRE':
            return [
                "Extend job offer",
                "Negotiate salary and benefits",
                "Begin reference checks",
                "Prepare onboarding materials",
                "Schedule start date"
            ]
        elif decision == 'INTERVIEW':
            return [
                "Schedule technical screening interview",
                "Prepare targeted interview questions",
                "Arrange technical assessment if needed",
                "Coordinate with hiring team",
                "Set up follow-up interview rounds"
            ]
        elif decision == 'MAYBE':
            return [
                "Request additional information",
                "Compare with other candidates",
                "Consider for alternative positions",
                "Schedule brief screening call",
                "Review portfolio/projects if available"
            ]
        else:  # REJECT
            return [
                "Send polite rejection email",
                "Provide constructive feedback if requested",
                "Keep resume on file for future opportunities",
                "Update candidate tracking system",
                "Consider for other open positions"
            ]
    
    def _assess_contact_completeness(self, contact: Dict) -> Dict:
        """Assess completeness of contact information"""
        required_fields = ['email', 'phone']
        optional_fields = ['linkedin', 'github']
        
        completeness = {
            'required_present': sum(1 for field in required_fields if contact.get(field)),
            'optional_present': sum(1 for field in optional_fields if contact.get(field)),
            'total_score': 0,
            'missing_required': [field for field in required_fields if not contact.get(field)],
            'missing_optional': [field for field in optional_fields if not contact.get(field)]
        }
        
        completeness['total_score'] = (
            (completeness['required_present'] / len(required_fields)) * 70 +
            (completeness['optional_present'] / len(optional_fields)) * 30
        )
        
        return completeness
    
    def _assess_resume_completeness(self, parsed_resume: Dict) -> Dict:
        """Assess completeness of resume sections"""
        sections = {
            'contact_info': bool(parsed_resume.get('contact_info', {}).get('email')),
            'experience': len(parsed_resume.get('experience', [])) > 0,
            'education': len(parsed_resume.get('education', [])) > 0,
            'skills': sum(len(v) for v in parsed_resume.get('skills', {}).values()) > 0,
            'projects': len(parsed_resume.get('projects', [])) > 0,
            'certifications': len(parsed_resume.get('certifications', [])) > 0
        }
        
        present_sections = sum(sections.values())
        total_sections = len(sections)
        
        return {
            'sections_present': sections,
            'completeness_percentage': (present_sections / total_sections) * 100,
            'missing_sections': [k for k, v in sections.items() if not v]
        }
    
    def _identify_missing_elements(self, parsed_resume: Dict) -> List[str]:
        """Identify missing resume elements"""
        missing = []
        
        contact = parsed_resume.get('contact_info', {})
        if not contact.get('email'):
            missing.append("Email address")
        if not contact.get('phone'):
            missing.append("Phone number")
        if not contact.get('linkedin'):
            missing.append("LinkedIn profile")
        
        if not parsed_resume.get('experience'):
            missing.append("Work experience section")
        
        if not parsed_resume.get('education'):
            missing.append("Education section")
        
        skills = parsed_resume.get('skills', {})
        if sum(len(v) for v in skills.values()) == 0:
            missing.append("Skills section")
        
        if not parsed_resume.get('projects'):
            missing.append("Projects/portfolio section")
        
        return missing
    
    def batch_screen_resumes(self, file_paths: List[str], job_criteria: Dict = None) -> List[Dict]:
        """Screen multiple resumes using LLM analysis"""
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {file_path}")
            result = self.screen_resume(file_path, job_criteria)
            results.append(result)
            
            # Add small delay between API calls to avoid rate limiting
            import time
            time.sleep(1)
        
        return results
    
    def compare_candidates(self, screening_results: List[Dict]) -> Dict:
        """Compare multiple candidates using LLM analysis"""
        if not screening_results:
            return {"error": "No screening results provided"}
        
        # Extract successful results only
        successful_results = [r for r in screening_results if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful screening results to compare"}
        
        try:
            # Create comparison prompt
            candidates_summary = []
            for i, result in enumerate(successful_results, 1):
                llm_analysis = result.get('llm_analysis', {})
                scores = llm_analysis.get('scoring', {})
                
                candidates_summary.append(f"""
                Candidate {i} ({result.get('file_path', 'Unknown')}):
                - Overall Score: {scores.get('overall_score', 'N/A')}
                - Technical Skills: {scores.get('technical_skills', 'N/A')}
                - Experience: {scores.get('experience', 'N/A')}
                - Decision: {llm_analysis.get('hiring_decision', {}).get('decision', 'N/A')}
                """)
            
            prompt = f"""
            Compare these candidates and provide:
            1. Ranking from best to worst with rationale
            2. Each candidate's unique strengths
            3. Best fit scenarios for each candidate
            4. Overall hiring recommendations
            
            CANDIDATES:
            {''.join(candidates_summary)}
            
            Provide a comprehensive comparison and ranking.
            """
            
            messages = [
                SystemMessage(content="You are a senior hiring manager comparing multiple candidates for optimal team fit."),
                HumanMessage(content=prompt)
            ]
            
            response = self.analysis_tool.llm.invoke(messages)
            
            return {
                "comparison_analysis": response.content,
                "candidates_count": len(successful_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing candidates: {str(e)}")
            return {"error": str(e)}
    
    def export_results(self, results: List[Dict], output_file: str = "llm_screening_results.json"):
        """Export screening results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results exported to {output_file}")
            return {"success": True, "file": output_file}
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return {"success": False, "error": str(e)}


def create_llm_resume_screener_agent():
    """Create the LLM-powered Resume Screener Agent using CrewAI"""
    
    # Initialize tools
    parser_tool = ResumeParserTool()
    analysis_tool = LLMAnalysisTool()
    
    # Create the agent with LLM integration
    resume_screener = Agent(
        role="AI-Powered Resume Screening Specialist",
        goal="Provide intelligent, comprehensive resume analysis using advanced language models for accurate candidate evaluation",
        backstory="""You are a next-generation AI resume screener that combines traditional parsing 
        capabilities with advanced language model intelligence. You can understand context, nuance, 
        and provide insights that go beyond keyword matching. Your analysis includes career trajectory 
        assessment, skill gap identification, market competitiveness evaluation, and personalized 
        improvement recommendations. You excel at providing detailed, actionable feedback that helps 
        both recruiters make better hiring decisions and candidates improve their profiles.""",
        tools=[parser_tool, analysis_tool],
        verbose=True,
        allow_delegation=False,
        llm=analysis_tool.llm  # Use the same LLM instance
    )
    
    return resume_screener


def main(job_description_pdf_path: str = None, resume_file_path: str = "final5resume.pdf", show_header: bool = True, show_footer: bool = True):
    """Example usage of the LLM-powered Resume Screener Agent"""
    
    if show_header:
        print("=" * 70)
        print("LLM-POWERED RESUME SCREENER AGENT")
        print("=" * 70)
    
    # Check for Gemini key
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: No LLM API key found in environment variables")
        print("Please add to the .env file:")
        print("GEMINI_API_KEY=your_gemini_key_here")
        return
    
    # Initialize the workflow
    try:
        screener = ResumeScreenerWorkflow()
        print("✅ LLM-powered screener initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing screener: {e}")
        return
    
    # Parse job description if provided, otherwise use default criteria
    if job_description_pdf_path and parse_job_description:
        print(f"\n📄 Parsing job description from: {job_description_pdf_path}")
        try:
            job_criteria = parse_job_description(job_description_pdf_path)
            print("✅ Job description parsed successfully!")
        except Exception as e:
            print(f"❌ Error parsing job description: {e}")
            print("🔄 Falling back to default job criteria...")
            job_criteria = {
                'position': 'Senior Software Developer',
                'required_skills': ['Python', 'JavaScript', 'SQL', 'Git'],
                'preferred_skills': ['React', 'AWS', 'Docker', 'Machine Learning'],
                'min_experience_years': 3,
                'education_level': 'masters',
                'industry': 'technology',
                'company_size': 'startup to mid-size',
                'remote_work': True
            }
    else:
        # Default job criteria (fallback)
        job_criteria = {
            'position': 'Senior Software Developer',
            'required_skills': ['Python', 'JavaScript', 'SQL', 'Git'],
            'preferred_skills': ['React', 'AWS', 'Docker', 'Machine Learning'],
            'min_experience_years': 3,
            'education_level': 'masters',
            'industry': 'technology',
            'company_size': 'startup to mid-size',
            'remote_work': True
        }
        if not job_description_pdf_path:
            print("ℹ️  No job description PDF provided, using default criteria")
        elif not parse_job_description:
            print("ℹ️  Job description parser not available, using default criteria")
    
    # Resume path to analyze (parameterized)
    resume_path = resume_file_path  
    
    print(f"\n🔍 Analyzing resume: {resume_path}")
    print("⏳ This may take 30-60 seconds due to LLM processing...")
    
    # Show job criteria and process steps
    print(f"\n📋 JOB CRITERIA:")
    for key, value in job_criteria.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Suppressed verbose analysis process prints
    
    # Run the actual screening
    print("\n🚀 Running actual analysis...")
    result = screener.screen_resume(resume_path, job_criteria)
    
    if not result.get("success"):
        print(f"❌ Screening failed: {result.get('error', 'Unknown error')}")
        return
    
    report = result.get("screening_report", {})
    exec_summary = report.get("executive_summary", {})
    llm_analysis = result.get("llm_analysis", {})
    scores = llm_analysis.get("scoring", {})
    decision = llm_analysis.get("hiring_decision", {})
    
    print("\n================== RESULTS ==================")
    if scores:
        print(f"✅ Overall Score: {scores.get('overall_score', 'N/A')}/100")
    if decision:
        print(f"🎯 Decision: {decision.get('decision', 'N/A')} (Confidence: {decision.get('confidence', 'N/A')})")
    
    overall_assessment = exec_summary.get("overall_assessment") or llm_analysis.get("overall_assessment")
    if overall_assessment:
        print("\n🧠 Overall Assessment:\n")
        print(overall_assessment)
    
    if show_footer:
        print("\n📌 Done. Full structured output is available in the generated report object.")

    # Return scores for external ranking
    return {
        "resume_path": resume_path,
        "scores": scores if scores else {},
        "success": bool(result.get("success"))
    }


if __name__ == "__main__":
    # You can specify a job description PDF path here
    job_description_pdf = "Bottomline_ Intern + FTE - JD 2026 Batch.pdf"  # Update this path as needed
    main(job_description_pdf_path=job_description_pdf)