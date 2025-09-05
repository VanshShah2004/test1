import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

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
import logging

# Document processing libraries
import PyPDF2
import docx2txt
from pdfplumber import PDF
import fitz  # PyMuPDF for better PDF parsing

# NLP and ML libraries
import spacy
from spacy.matcher import Matcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
#from langchain.llms import OpenAI #old
from langchain_openai import OpenAI #new
from langchain.tools import tool

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab') #added
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab') #added

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
            with PDF.open(file_path) as pdf:
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

class SkillMatchingTool(BaseTool):
    name: str = "Skill Matching"
    description: str = "Evaluates and scores resume content based on various factors"
    
    def _run(self, parsed_resume: Dict, evaluation_criteria: Dict = None) -> Dict:
        """Evaluate parsed resume based on comprehensive criteria"""
        
        if evaluation_criteria is None:
            evaluation_criteria = self._get_default_criteria()
        
        evaluation = {
            'overall_score': 0,
            'category_scores': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        # Evaluate each category
        evaluation['category_scores']['technical_skills'] = self._evaluate_technical_skills(parsed_resume)
        evaluation['category_scores']['experience'] = self._evaluate_experience(parsed_resume)
        evaluation['category_scores']['education'] = self._evaluate_education(parsed_resume)
        evaluation['category_scores']['communication'] = self._evaluate_communication(parsed_resume)
        evaluation['category_scores']['achievements'] = self._evaluate_achievements(parsed_resume)
        evaluation['category_scores']['completeness'] = self._evaluate_completeness(parsed_resume)
        
        # Calculate overall score (weighted average)
        weights = {
            'technical_skills': 0.3,
            'experience': 0.25,
            'education': 0.15,
            'communication': 0.1,
            'achievements': 0.1,
            'completeness': 0.1
        }
        
        weighted_score = sum(
            evaluation['category_scores'][category] * weight 
            for category, weight in weights.items()
        )
        evaluation['overall_score'] = round(weighted_score, 2)
        
        # Generate insights
        evaluation['strengths'] = self._identify_strengths(parsed_resume, evaluation['category_scores'])
        evaluation['weaknesses'] = self._identify_weaknesses(parsed_resume, evaluation['category_scores'])
        evaluation['recommendations'] = self._generate_recommendations(parsed_resume, evaluation['category_scores'])
        
        return evaluation
    
    def _get_default_criteria(self) -> Dict:
        """Default evaluation criteria"""
        return {
            'min_experience_years': 2,
            'required_skills': [],
            'preferred_skills': [],
            'education_level': 'bachelor',
            'industry': 'technology'
        }
    
    def _evaluate_technical_skills(self, resume: Dict) -> float:
        """Evaluate technical skills comprehensiveness and relevance"""
        skills = resume.get('skills', {})
        
        score = 0
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        
        if total_skills == 0:
            return 0
        
        # Score based on variety of skill categories
        categories_with_skills = sum(1 for skill_list in skills.values() if skill_list)
        score += (categories_with_skills / len(skills)) * 30  # Max 30 points for diversity
        
        # Score based on total number of relevant skills
        if total_skills >= 20:
            score += 40  # Max 40 points for comprehensive skills
        elif total_skills >= 10:
            score += 25
        elif total_skills >= 5:
            score += 15
        else:
            score += 5
        
        # Bonus for modern/in-demand technologies
        modern_techs = ['Python', 'React', 'Node.js', 'AWS', 'Docker', 'Kubernetes', 'Machine Learning']
        modern_count = sum(1 for tech in modern_techs 
                          for skill_list in skills.values() 
                          for skill in skill_list 
                          if tech.lower() in skill.lower())
        
        score += min(modern_count * 5, 30)  # Max 30 bonus points
        
        return min(score, 100)
    
    def _evaluate_experience(self, resume: Dict) -> float:
        """Evaluate work experience quality and relevance"""
        experience = resume.get('experience', [])
        
        if not experience:
            return 10  # Minimum score for students/entry-level
        
        score = 0
        
        # Score based on number of positions
        score += min(len(experience) * 15, 45)  # Max 45 for 3+ positions
        
        # Score based on description quality
        total_descriptions = sum(len(exp.get('description', [])) for exp in experience)
        if total_descriptions >= 10:
            score += 25
        elif total_descriptions >= 5:
            score += 15
        else:
            score += 5
        
        # Score based on progression (title improvements)
        titles = [exp.get('title', '').lower() for exp in experience]
        progression_keywords = ['senior', 'lead', 'manager', 'director', 'principal']
        
        if any(keyword in ' '.join(titles) for keyword in progression_keywords):
            score += 20
        
        # Score based on company information completeness
        companies_listed = sum(1 for exp in experience if exp.get('company'))
        score += (companies_listed / len(experience)) * 10
        
        return min(score, 100)
    
    def _evaluate_education(self, resume: Dict) -> float:
        """Evaluate educational background"""
        education = resume.get('education', [])
        
        if not education:
            return 20  # Some score for self-taught candidates
        
        score = 40  # Base score for having education
        
        # Score based on degree level
        degrees = [edu.get('degree', '').lower() for edu in education]
        degree_text = ' '.join(degrees)
        
        if any(keyword in degree_text for keyword in ['phd', 'ph.d', 'doctorate']):
            score += 30
        elif any(keyword in degree_text for keyword in ['master', 'm.s', 'm.a', 'mba', 'm.tech']):
            score += 20
        elif any(keyword in degree_text for keyword in ['bachelor', 'b.s', 'b.a', 'b.tech']):
            score += 10
        
        # Score based on GPA if available
        for edu in education:
            gpa = edu.get('gpa')
            if gpa:
                try:
                    gpa_float = float(gpa)
                    if gpa_float >= 3.5:
                        score += 15
                    elif gpa_float >= 3.0:
                        score += 10
                    break
                except ValueError:
                    continue
        
        # Score based on relevant field
        relevant_fields = ['computer', 'engineering', 'technology', 'science', 'mathematics']
        if any(field in degree_text for field in relevant_fields):
            score += 15
        
        return min(score, 100)
    
    def _evaluate_communication(self, resume: Dict) -> float:
        """Evaluate communication skills based on resume quality"""
        raw_text = resume.get('raw_text', '')
        
        if not raw_text:
            return 0
        
        score = 0
        
        # Check for proper structure
        sections = ['experience', 'education', 'skills']
        section_count = sum(1 for section in sections if section in raw_text.lower())
        score += (section_count / len(sections)) * 30
        
        # Check text quality (length, completeness)
        if len(raw_text) >= 1000:
            score += 25
        elif len(raw_text) >= 500:
            score += 15
        else:
            score += 5
        
        # Check for contact information
        contact = resume.get('contact_info', {})
        contact_score = sum(10 for key in ['email', 'phone'] if contact.get(key))
        score += contact_score
        
        # Check for professional profiles
        if contact.get('linkedin'):
            score += 10
        if contact.get('github'):
            score += 15
        
        # Check language quality (basic grammar check)
        sentences = sent_tokenize(raw_text)
        if len(sentences) >= 10:
            score += 10
        
        return min(score, 100)
    
    def _evaluate_achievements(self, resume: Dict) -> float:
        """Evaluate achievements and certifications"""
        achievements = resume.get('achievements', [])
        certifications = resume.get('certifications', [])
        projects = resume.get('projects', [])
        
        score = 0
        
        # Score achievements
        score += min(len(achievements) * 15, 30)
        
        # Score certifications
        score += min(len(certifications) * 10, 40)
        
        # Score projects
        score += min(len(projects) * 8, 30)
        
        # Bonus for having all three types
        if achievements and certifications and projects:
            score += 20
        
        return min(score, 100)
    
    def _evaluate_completeness(self, resume: Dict) -> float:
        """Evaluate resume completeness"""
        required_sections = [
            'contact_info', 'experience', 'education', 'skills'
        ]
        
        score = 0
        
        for section in required_sections:
            section_data = resume.get(section)
            if section_data:
                if isinstance(section_data, list) and section_data:
                    score += 25
                elif isinstance(section_data, dict) and any(section_data.values()):
                    score += 25
                else:
                    score += 10
        
        return min(score, 100)
    
    def _identify_strengths(self, resume: Dict, scores: Dict) -> List[str]:
        """Identify candidate strengths"""
        strengths = []
        
        if scores.get('technical_skills', 0) >= 80:
            strengths.append("Excellent technical skill diversity and depth")
        
        if scores.get('experience', 0) >= 80:
            strengths.append("Strong professional experience with detailed descriptions")
        
        if scores.get('education', 0) >= 80:
            strengths.append("Solid educational background")
        
        if scores.get('communication', 0) >= 80:
            strengths.append("Well-structured and comprehensive resume")
        
        if scores.get('achievements', 0) >= 70:
            strengths.append("Good track record of achievements and certifications")
        
        # Specific skill strengths
        skills = resume.get('skills', {})
        if len(skills.get('programming_languages', [])) >= 5:
            strengths.append("Proficient in multiple programming languages")
        
        if skills.get('frameworks', []):
            strengths.append("Experience with modern frameworks and technologies")
        
        return strengths
    
    def _identify_weaknesses(self, resume: Dict, scores: Dict) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []
        
        if scores.get('technical_skills', 0) < 50:
            weaknesses.append("Limited technical skills or lack of skill diversity")
        
        if scores.get('experience', 0) < 50:
            weaknesses.append("Insufficient work experience details or limited experience")
        
        if scores.get('education', 0) < 40:
            weaknesses.append("Educational background could be strengthened")
        
        if scores.get('communication', 0) < 60:
            weaknesses.append("Resume structure and presentation needs improvement")
        
        if scores.get('achievements', 0) < 30:
            weaknesses.append("Lack of demonstrated achievements or certifications")
        
        # Missing contact information
        contact = resume.get('contact_info', {})
        if not contact.get('email'):
            weaknesses.append("Missing email contact information")
        
        if not contact.get('linkedin') and not contact.get('github'):
            weaknesses.append("Missing professional online presence (LinkedIn/GitHub)")
        
        return weaknesses
    
    def _generate_recommendations(self, resume: Dict, scores: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.get('technical_skills', 0) < 70:
            recommendations.append("Consider adding more technical skills and modern technologies")
        
        if scores.get('experience', 0) < 70:
            recommendations.append("Provide more detailed descriptions of work experience with specific achievements")
        
        if scores.get('communication', 0) < 70:
            recommendations.append("Improve resume formatting and ensure all sections are well-structured")
        
        if not resume.get('certifications'):
            recommendations.append("Consider obtaining relevant professional certifications")
        
        if not resume.get('projects'):
            recommendations.append("Add personal or professional projects to demonstrate practical skills")
        
        contact = resume.get('contact_info', {})
        if not contact.get('linkedin'):
            recommendations.append("Create a professional LinkedIn profile")
        
        if not contact.get('github') and resume.get('skills', {}).get('programming_languages'):
            recommendations.append("Create a GitHub profile to showcase coding projects")
        
        return recommendations

def create_resume_screener_agent():
    """Create the Resume Screener Agent using CrewAI"""
    
    # Initialize tools
    parser_tool = ResumeParserTool()
    matching_tool = SkillMatchingTool()
    
    # Create the agent
    resume_screener = Agent(
        role="Resume Screening Specialist",
        goal="Thoroughly analyze and evaluate resumes to assess candidate suitability",
        backstory="""You are an expert HR professional with over 10 years of experience in 
        talent acquisition and resume screening. You have a keen eye for identifying 
        candidate potential and matching skills to job requirements. You excel at 
        comprehensive resume analysis, considering technical skills, experience quality, 
        educational background, and overall presentation.""",
        tools=[parser_tool, matching_tool],
        verbose=True,
        allow_delegation=False
    )
    
    return resume_screener

class ResumeScreenerWorkflow:
    """Main workflow class for resume screening process"""
    
    def __init__(self, llm=None):
        """Initialize the workflow with optional LLM"""
        self.agent = create_resume_screener_agent()
        self.parser_tool = ResumeParserTool()
        self.matching_tool = SkillMatchingTool()
        self.llm = llm or self._get_default_llm()
        
    def _get_default_llm(self):
        """Get default LLM (you may need to configure this)"""
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            return OpenAI(temperature=0.1, openai_api_key=openai_api_key)
        except:
            logger.warning("OpenAI not configured, using mock LLM")
            return None
    
    def screen_resume(self, file_path: str, job_criteria: Dict = None) -> Dict:
        """Main method to screen a resume"""
        try:
            logger.info(f"Starting resume screening for: {file_path}")
            
            # Step 1: Parse the resume
            logger.info("Parsing resume...")
            parsed_resume = self.parser_tool._run(file_path)
            
            if "error" in parsed_resume:
                return {
                    "success": False,
                    "error": parsed_resume["error"],
                    "file_path": file_path
                }
            
            # Step 2: Evaluate the resume
            logger.info("Evaluating resume...")
            evaluation = self.matching_tool._run(parsed_resume, job_criteria)
            
            # Step 3: Generate comprehensive report
            logger.info("Generating screening report...")
            screening_report = self._generate_screening_report(
                file_path, parsed_resume, evaluation
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "parsed_data": parsed_resume,
                "evaluation": evaluation,
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
    
    def _generate_screening_report(self, file_path: str, parsed_resume: Dict, 
                                 evaluation: Dict) -> Dict:
        """Generate comprehensive screening report"""
        
        report = {
            "candidate_summary": self._create_candidate_summary(parsed_resume),
            "evaluation_summary": self._create_evaluation_summary(evaluation),
            "detailed_analysis": {
                "technical_skills": self._analyze_technical_skills(parsed_resume),
                "experience_analysis": self._analyze_experience(parsed_resume),
                "education_analysis": self._analyze_education(parsed_resume),
                "contact_completeness": self._analyze_contact_info(parsed_resume)
            },
            "recommendation": self._generate_final_recommendation(evaluation),
            "next_steps": self._suggest_next_steps(evaluation)
        }
        
        return report
    
    def _create_candidate_summary(self, parsed_resume: Dict) -> Dict:
        """Create a summary of the candidate"""
        contact = parsed_resume.get('contact_info', {})
        skills = parsed_resume.get('skills', {})
        experience = parsed_resume.get('experience', [])
        education = parsed_resume.get('education', [])
        
        # Calculate total experience (rough estimate)
        total_experience = len(experience)  # Simplified calculation
        
        # Get highest education
        education_levels = {'phd': 4, 'master': 3, 'bachelor': 2, 'associate': 1}
        highest_education = "Not specified"
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            for level, value in education_levels.items():
                if level in degree:
                    if highest_education == "Not specified" or education_levels.get(highest_education, 0) < value:
                        highest_education = level.capitalize()
        
        # Get primary technical skills
        primary_skills = []
        for category in ['programming_languages', 'frameworks', 'databases']:
            primary_skills.extend(skills.get(category, [])[:3])  # Top 3 from each
        
        return {
            "email": contact.get('email', 'Not provided'),
            "phone": contact.get('phone', 'Not provided'),
            "linkedin": contact.get('linkedin', 'Not provided'),
            "github": contact.get('github', 'Not provided'),
            "estimated_experience": f"{total_experience} positions listed",
            "highest_education": highest_education,
            "primary_technical_skills": primary_skills[:8],  # Top 8 skills
            "total_skills_count": sum(len(skill_list) for skill_list in skills.values()),
            "certifications_count": len(parsed_resume.get('certifications', [])),
            "projects_count": len(parsed_resume.get('projects', []))
        }
    
    def _create_evaluation_summary(self, evaluation: Dict) -> Dict:
        """Create evaluation summary"""
        overall_score = evaluation.get('overall_score', 0)
        
        # Determine recommendation level
        if overall_score >= 80:
            recommendation_level = "Highly Recommended"
        elif overall_score >= 65:
            recommendation_level = "Recommended"
        elif overall_score >= 50:
            recommendation_level = "Consider with Reservations"
        else:
            recommendation_level = "Not Recommended"
        
        return {
            "overall_score": overall_score,
            "recommendation_level": recommendation_level,
            "category_scores": evaluation.get('category_scores', {}),
            "top_strengths": evaluation.get('strengths', [])[:3],
            "main_concerns": evaluation.get('weaknesses', [])[:3]
        }
    
    def _analyze_technical_skills(self, parsed_resume: Dict) -> Dict:
        """Detailed technical skills analysis"""
        skills = parsed_resume.get('skills', {})
        
        analysis = {
            "programming_languages": {
                "count": len(skills.get('programming_languages', [])),
                "skills": skills.get('programming_languages', []),
                "assessment": ""
            },
            "frameworks_and_libraries": {
                "count": len(skills.get('frameworks', [])),
                "skills": skills.get('frameworks', []),
                "assessment": ""
            },
            "databases": {
                "count": len(skills.get('databases', [])),
                "skills": skills.get('databases', []),
                "assessment": ""
            },
            "tools_and_platforms": {
                "count": len(skills.get('tools', [])),
                "skills": skills.get('tools', []),
                "assessment": ""
            }
        }
        
        # Add assessments
        if analysis["programming_languages"]["count"] >= 5:
            analysis["programming_languages"]["assessment"] = "Excellent language diversity"
        elif analysis["programming_languages"]["count"] >= 3:
            analysis["programming_languages"]["assessment"] = "Good language knowledge"
        else:
            analysis["programming_languages"]["assessment"] = "Limited language exposure"
        
        if analysis["frameworks_and_libraries"]["count"] >= 3:
            analysis["frameworks_and_libraries"]["assessment"] = "Good framework experience"
        elif analysis["frameworks_and_libraries"]["count"] >= 1:
            analysis["frameworks_and_libraries"]["assessment"] = "Basic framework knowledge"
        else:
            analysis["frameworks_and_libraries"]["assessment"] = "No frameworks mentioned"
        
        return analysis
    
    def _analyze_experience(self, parsed_resume: Dict) -> Dict:
        """Detailed experience analysis"""
        experience = parsed_resume.get('experience', [])
        
        if not experience:
            return {
                "total_positions": 0,
                "experience_quality": "No experience listed",
                "progression_analysis": "Cannot assess",
                "description_quality": "No descriptions available"
            }
        
        # Analyze description quality
        total_descriptions = sum(len(exp.get('description', [])) for exp in experience)
        avg_descriptions = total_descriptions / len(experience) if experience else 0
        
        if avg_descriptions >= 4:
            description_quality = "Excellent - Detailed descriptions"
        elif avg_descriptions >= 2:
            description_quality = "Good - Adequate descriptions"
        elif avg_descriptions >= 1:
            description_quality = "Basic - Minimal descriptions"
        else:
            description_quality = "Poor - Missing descriptions"
        
        # Check for career progression
        titles = [exp.get('title', '').lower() for exp in experience]
        progression_keywords = ['senior', 'lead', 'manager', 'director', 'principal']
        has_progression = any(keyword in ' '.join(titles) for keyword in progression_keywords)
        
        return {
            "total_positions": len(experience),
            "experience_quality": description_quality,
            "progression_analysis": "Shows career growth" if has_progression else "Linear progression",
            "description_quality": f"Average {avg_descriptions:.1f} bullet points per position",
            "companies_with_names": sum(1 for exp in experience if exp.get('company')),
            "positions_with_duration": sum(1 for exp in experience if exp.get('duration'))
        }
    
    def _analyze_education(self, parsed_resume: Dict) -> Dict:
        """Detailed education analysis"""
        education = parsed_resume.get('education', [])
        
        if not education:
            return {
                "education_level": "Not specified",
                "field_relevance": "Cannot assess",
                "gpa_mentioned": False,
                "institution_quality": "Not provided"
            }
        
        # Determine highest degree
        degrees = [edu.get('degree', '').lower() for edu in education]
        degree_text = ' '.join(degrees)
        
        if any(keyword in degree_text for keyword in ['phd', 'ph.d', 'doctorate']):
            education_level = "Doctorate"
        elif any(keyword in degree_text for keyword in ['master', 'm.s', 'm.a', 'mba']):
            education_level = "Master's"
        elif any(keyword in degree_text for keyword in ['bachelor', 'b.s', 'b.a']):
            education_level = "Bachelor's"
        else:
            education_level = "Other/Not specified"
        
        # Check field relevance
        relevant_fields = ['computer', 'engineering', 'technology', 'science', 'mathematics']
        field_relevant = any(field in degree_text for field in relevant_fields)
        
        # Check for GPA
        gpa_mentioned = any(edu.get('gpa') for edu in education)
        
        return {
            "education_level": education_level,
            "field_relevance": "Relevant technical field" if field_relevant else "Non-technical field",
            "gpa_mentioned": gpa_mentioned,
            "institution_quality": "Institutions listed" if any(edu.get('institution') for edu in education) else "No institutions specified",
            "total_degrees": len(education)
        }
    
    def _analyze_contact_info(self, parsed_resume: Dict) -> Dict:
        """Analyze completeness of contact information"""
        contact = parsed_resume.get('contact_info', {})
        
        completeness_score = 0
        missing_items = []
        
        if contact.get('email'):
            completeness_score += 25
        else:
            missing_items.append('Email')
        
        if contact.get('phone'):
            completeness_score += 25
        else:
            missing_items.append('Phone')
        
        if contact.get('linkedin'):
            completeness_score += 25
        else:
            missing_items.append('LinkedIn profile')
        
        if contact.get('github'):
            completeness_score += 25
        else:
            missing_items.append('GitHub profile')
        
        return {
            "completeness_score": completeness_score,
            "missing_items": missing_items,
            "professional_presence": "Strong" if contact.get('linkedin') and contact.get('github') else "Moderate" if contact.get('linkedin') or contact.get('github') else "Weak"
        }
    
    def _generate_final_recommendation(self, evaluation: Dict) -> Dict:
        """Generate final hiring recommendation"""
        overall_score = evaluation.get('overall_score', 0)
        
        if overall_score >= 80:
            decision = "HIRE"
            rationale = "Candidate demonstrates strong qualifications across all evaluation criteria."
        elif overall_score >= 65:
            decision = "INTERVIEW"
            rationale = "Candidate shows good potential with some areas for verification in interview."
        elif overall_score >= 50:
            decision = "MAYBE"
            rationale = "Candidate has basic qualifications but significant gaps need addressing."
        else:
            decision = "REJECT"
            rationale = "Candidate does not meet minimum requirements for this position."
        
        return {
            "decision": decision,
            "confidence": "High" if overall_score >= 75 or overall_score <= 40 else "Medium",
            "rationale": rationale,
            "overall_score": overall_score
        }
    
    def _suggest_next_steps(self, evaluation: Dict) -> List[str]:
        """Suggest next steps based on evaluation"""
        overall_score = evaluation.get('overall_score', 0)
        next_steps = []
        
        if overall_score >= 65:
            next_steps.append("Schedule initial screening interview")
            next_steps.append("Verify technical skills through coding assessment")
            next_steps.append("Check references from previous employers")
        elif overall_score >= 50:
            next_steps.append("Request additional information about experience")
            next_steps.append("Consider for junior or training positions")
            next_steps.append("Evaluate against other candidates in pool")
        else:
            next_steps.append("Send polite rejection email")
            next_steps.append("Keep resume on file for future opportunities")
        
        # Add specific suggestions based on weaknesses
        weaknesses = evaluation.get('weaknesses', [])
        if any('technical' in weakness.lower() for weakness in weaknesses):
            next_steps.append("If proceeding, focus technical interview on practical skills")
        
        if any('experience' in weakness.lower() for weakness in weaknesses):
            next_steps.append("If proceeding, ask detailed questions about work history")
        
        return next_steps
    
    def batch_screen_resumes(self, file_paths: List[str], job_criteria: Dict = None) -> List[Dict]:
        """Screen multiple resumes"""
        results = []
        
        for file_path in file_paths:
            logger.info(f"Processing {file_path}")
            result = self.screen_resume(file_path, job_criteria)
            results.append(result)
        
        return results
    
    def export_results(self, results: List[Dict], output_file: str = "screening_results.json"):
        """Export screening results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")

def main():
    """Example usage of the Resume Screener Agent"""
    
    # Initialize the workflow
    screener = ResumeScreenerWorkflow()
    
    # Example job criteria (optional)
    job_criteria = {
        'min_experience_years': 3,
        'required_skills': ['Python', 'JavaScript', 'SQL'],
        'preferred_skills': ['React', 'AWS', 'Docker'],
        'education_level': 'bachelor',
        'industry': 'technology'
    }
    
    # Screen a single resume
    resume_path = "Nirmit_Jain_Resume_Final.pdf"  # Replace with actual path
    
    print("=" * 60)
    print("RESUME SCREENER AGENT - PROCESSING")
    print("=" * 60)
    
    # Note: You'll need to replace this with an actual file path
    try:
        result = screener.screen_resume(resume_path, job_criteria)
        
        if result['success']:
            print(f"\n✅ Successfully screened resume: {result['file_path']}")
            print(f"📊 Overall Score: {result['evaluation']['overall_score']}/100")
            print(f"🎯 Recommendation: {result['screening_report']['recommendation']['decision']}")
            
            print("\n📋 CANDIDATE SUMMARY:")
            summary = result['screening_report']['candidate_summary']
            print(f"  • Email: {summary.get('email', 'N/A')}")
            print(f"  • Experience: {summary.get('estimated_experience', 'N/A')}")
            print(f"  • Education: {summary.get('highest_education', 'N/A')}")
            print(f"  • Skills Count: {summary.get('total_skills_count', 0)}")
            
            print("\n⭐ TOP STRENGTHS:")
            for strength in result['evaluation']['strengths'][:3]:
                print(f"  • {strength}")
            
            print("\n⚠️  AREAS FOR IMPROVEMENT:")
            for weakness in result['evaluation']['weaknesses'][:3]:
                print(f"  • {weakness}")
            
            print("\n📌 NEXT STEPS:")
            for step in result['screening_report']['next_steps'][:3]:
                print(f"  • {step}")
                
        else:
            print(f"❌ Error screening resume: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: Please provide a valid resume file path")
        print(f"Example usage:")
        print(f"  screener = ResumeScreenerWorkflow()")
        print(f"  result = screener.screen_resume('path/to/your/resume.pdf')")

if __name__ == "__main__":
    main()