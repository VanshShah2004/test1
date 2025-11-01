"""
Pure NLP-based extraction service (NO LLMs).
Uses rule-based parsing, regex patterns, and keyword matching.
"""

from typing import Dict, List
import os

from .nlp_extractors import SkillsExtractor, ExperienceExtractor, EducationClassifier
from .skills_dictionary import TECH_SKILLS_DICT

# Try to import trained components if available
try:
    from .trained_skills_extractor import TrainedEducationClassifier
    TRAINED_AVAILABLE = True
except ImportError:
    TRAINED_AVAILABLE = False


class NLPService:
    """
    Pure NLP-based extraction service.
    No LLMs, no API calls - 100% local, deterministic processing.
    """
    
    def __init__(self, use_trained: bool = True):
        """
        Initialize NLP service.
        
        Args:
            use_trained: If True, uses trained models if available, otherwise rule-based only
        """
        self.skills_extractor = SkillsExtractor(TECH_SKILLS_DICT)
        self.experience_extractor = ExperienceExtractor()
        
        # Use trained classifier if available and requested
        if use_trained and TRAINED_AVAILABLE:
            try:
                self.education_classifier = TrainedEducationClassifier(use_trained=True)
                if self.education_classifier.trained_model:
                    print("✅ Using trained education classifier")
            except Exception as e:
                print(f"⚠️  Failed to load trained classifier, using rule-based: {e}")
                self.education_classifier = EducationClassifier()
        else:
            self.education_classifier = EducationClassifier()
        
        # Try to load trained skills dictionary if available
        self._load_trained_skills_dict()
    
    def _load_trained_skills_dict(self):
        """Load trained skills dictionary if available."""
        trained_dict_path = "services/skills_dictionary_trained.py"
        if os.path.exists(trained_dict_path):
            try:
                # Dynamically import trained dictionary
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "skills_dictionary_trained",
                    trained_dict_path
                )
                trained_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(trained_module)
                
                # Update skills extractor with trained dictionary
                if hasattr(trained_module, 'TECH_SKILLS_DICT'):
                    self.skills_extractor = SkillsExtractor(trained_module.TECH_SKILLS_DICT)
                    print("✅ Using trained skills dictionary")
            except Exception as e:
                print(f"⚠️  Failed to load trained dictionary, using default: {e}")
    
    def parse_resume(self, text: str) -> Dict:
        """
        Parse resume text and extract structured data.
        Returns dict matching ResumeData schema.
        """
        if not text:
            return {
                'extracted_skills': [],
                'education_level': 'bachelors',
                'total_experience_years': 0.0
            }
        
        return {
            'extracted_skills': self.skills_extractor.extract(text),
            'education_level': self.education_classifier.classify(text),
            'total_experience_years': self.experience_extractor.extract_experience(text)
        }
    
    def parse_job_description(self, text: str) -> Dict:
        """
        Parse job description and extract criteria.
        Returns dict matching JobCriteria schema.
        """
        if not text:
            return {
                'position': 'Unknown Position',
                'required_skills': [],
                'preferred_skills': [],
                'min_experience_years': 0,
                'education_level': 'bachelors',
                'industry': 'general',
                'company_size': 'medium',
                'remote_work': False
            }
        
        text_lower = text.lower()
        
        # Extract position (first non-empty line, usually)
        position = self._extract_position(text)
        
        # Extract all skills
        all_skills = self.skills_extractor.extract(text)
        
        # Separate required vs preferred
        required_skills, preferred_skills = self._extract_skills_sections(text, all_skills)
        
        # Extract minimum experience
        min_experience = self._extract_min_experience(text)
        
        # Extract education level
        education_level = self.education_classifier.classify(text)
        
        # Extract other fields
        industry = self._extract_industry(text)
        company_size = self._extract_company_size(text)
        remote_work = self._extract_remote_work(text)
        
        return {
            'position': position,
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'min_experience_years': min_experience,
            'education_level': education_level,
            'industry': industry,
            'company_size': company_size,
            'remote_work': remote_work
        }
    
    def _extract_position(self, text: str) -> str:
        """Extract job position/title from first few lines."""
        lines = text.split('\n')[:15]
        for line in lines:
            line_clean = line.strip()
            if 10 <= len(line_clean) <= 100:
                # Check if it looks like a job title (starts with capital, no digits in first part)
                if line_clean[0].isupper() and not any(char.isdigit() for char in line_clean[:15]):
                    # Common job title keywords
                    if any(word in line_clean.lower() for word in ['engineer', 'developer', 'analyst', 'manager', 'specialist', 'architect', 'scientist', 'consultant', 'director']):
                        return line_clean
                    # Or if it's a short line that looks like a title
                    if len(line_clean.split()) <= 5:
                        return line_clean
        return 'Unknown Position'
    
    def _extract_skills_sections(self, text: str, all_skills: List[str]) -> tuple:
        """Extract required and preferred skills separately."""
        text_lower = text.lower()
        
        required_keywords = ['required', 'must have', 'essential', 'mandatory', 'must possess']
        preferred_keywords = ['preferred', 'nice to have', 'bonus', 'advantageous', 'optional']
        
        required_skills = []
        preferred_skills = []
        
        # Find required section
        required_section = None
        for keyword in required_keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                # Extract next 800 chars
                section_end = min(len(text), idx + 800)
                required_section = text_lower[idx:section_end]
                break
        
        # Find preferred section
        preferred_section = None
        for keyword in preferred_keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                section_end = min(len(text), idx + 800)
                preferred_section = text_lower[idx:section_end]
                break
        
        # Assign skills to sections
        for skill in all_skills:
            skill_lower = skill.lower()
            if required_section and skill_lower in required_section:
                required_skills.append(skill)
            elif preferred_section and skill_lower in preferred_section:
                preferred_skills.append(skill)
        
        # If no clear sections, assume all are required
        if not required_skills and not preferred_skills:
            required_skills = all_skills
        
        return required_skills, preferred_skills
    
    def _extract_min_experience(self, text: str) -> int:
        """Extract minimum experience years requirement."""
        import re
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)[-+]\s*years?\s*experience',
            r'(\d+)\s*years?\s*experience\s*required'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    years = int(match.group(1))
                    return years
                except:
                    continue
        return 0
    
    def _extract_industry(self, text: str) -> str:
        """Extract industry from job description."""
        industry_keywords = {
            'fintech': ['fintech', 'financial technology', 'banking', 'finance'],
            'healthcare': ['healthcare', 'medical', 'hospital', 'pharma'],
            'saas': ['saas', 'software as a service', 'cloud software'],
            'ecommerce': ['ecommerce', 'e-commerce', 'retail', 'online retail'],
            'education': ['education', 'edtech', 'learning'],
            'gaming': ['gaming', 'game development', 'entertainment'],
            'consulting': ['consulting', 'advisory']
        }
        
        text_lower = text.lower()
        for industry, keywords in industry_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return industry
        return 'general'
    
    def _extract_company_size(self, text: str) -> str:
        """Extract company size mention."""
        size_keywords = {
            'startup': ['startup', 'start-up', 'early stage'],
            'small': ['small company', 'small team', '< 50', 'less than 50'],
            'medium': ['medium', 'mid-size', '50-200'],
            'large': ['large', 'enterprise', 'fortune 500', '> 500', 'multinational']
        }
        
        text_lower = text.lower()
        for size, keywords in size_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return size
        return 'medium'
    
    def _extract_remote_work(self, text: str) -> bool:
        """Extract remote work preference."""
        remote_keywords = ['remote', 'work from home', 'wfh', 'hybrid', 'distributed', 'virtual']
        text_lower = text.lower()
        return any(kw in text_lower for kw in remote_keywords)

