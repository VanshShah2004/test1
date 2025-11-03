"""
NLP-based extractors for resume and job description parsing (NO LLMs).
Uses rule-based parsing, regex patterns, and keyword matching.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dateutil import parser as date_parser

# Import skills dictionary
try:
    from .skills_dictionary import TECH_SKILLS_DICT
except ImportError:
    # Fallback minimal dictionary if not available
    TECH_SKILLS_DICT = {
        "programming_languages": ["python", "java", "javascript", "c++", "c", "c#", "go", "rust", "swift", "kotlin"],
        "frameworks": ["react", "angular", "vue", "django", "flask", "spring", "express", "next.js"],
        "databases": ["mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
        "tools": ["git", "jenkins", "jira", "confluence", "docker", "kubernetes"]
    }


class SkillsExtractor:
    """Extract technical skills from text using dictionary-based pattern matching."""
    
    def __init__(self, skills_dict: Dict = None):
        self.skills_dict = skills_dict or TECH_SKILLS_DICT
        
        # Flatten all skills into one set for fast lookup
        self.all_skills = set()
        for category, skills in self.skills_dict.items():
            self.all_skills.update([s.lower() for s in skills])
        
        # Create regex pattern for all skills (word boundaries)
        pattern_parts = [f'\\b{re.escape(skill)}\\b' for skill in self.all_skills]
        self.skills_pattern = re.compile(
            '|'.join(pattern_parts), 
            re.IGNORECASE
        )
        
        # Technical context indicators
        self.tech_indicators = [
            'experience with', 'proficient in', 'skilled in',
            'worked with', 'developed using', 'built with',
            'technologies:', 'skills:', 'expertise in',
            'proficient', 'familiar with', 'knowledge of'
        ]
    
    def extract(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching.
        Returns normalized, deduplicated list.
        """
        if not text:
            return []
        
        # Find all skill mentions
        matches = self.skills_pattern.findall(text.lower())
        
        # Normalize and deduplicate
        extracted = list(set([m.lower() for m in matches]))
        
        # Context validation: Remove false positives
        validated = self._validate_context(text, extracted)
        
        return validated
    
    def _validate_context(self, text: str, skills: List[str]) -> List[str]:
        """
        Validate skills by checking context.
        Remove skills that appear in non-technical contexts.
        """
        validated = []
        text_lower = text.lower()
        
        for skill in skills:
            # Find all occurrences
            pattern = re.compile(f'\\b{re.escape(skill)}\\b', re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            for match in matches:
                # Check surrounding context (20 chars before/after)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text_lower[start:end]
                
                # If context suggests technical mention, keep it
                if any(indicator in context for indicator in self.tech_indicators):
                    if skill not in validated:
                        validated.append(skill)
                    break
        
        # If no validation passed, still include skills found (less strict)
        if not validated and skills:
            return skills[:20]  # Limit to top 20 to avoid noise
        
        return validated


class ExperienceExtractor:
    """Extract experience years from resume text using date parsing."""
    
    def __init__(self):
        # Common date patterns in resumes
        self.date_patterns = [
            # MM/YYYY, M/YYYY
            r'\b(\d{1,2})[/-](\d{4})\b',
            # Month YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',
            # YYYY-MM, YYYY/MM
            r'\b(\d{4})[/-](\d{1,2})\b',
            # Full date: Month DD, YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(\d{4})\b',
        ]
        
        # Employment keywords
        self.employment_keywords = [
            'experience', 'worked', 'employed', 'position', 'role',
            'intern', 'internship', 'full-time', 'part-time',
            'contract', 'consultant', 'engineer', 'developer',
            'manager', 'lead', 'senior', 'junior'
        ]
        
        # Internship indicators
        self.internship_keywords = [
            'intern', 'internship', 'co-op', 'coop', 'trainee'
        ]
        
        # Relative dates
        self.current_keywords = ['present', 'current', 'now', 'ongoing']
    
    def extract_experience(self, text: str) -> float:
        """
        Extract total years of professional experience.
        Returns float (e.g., 2.5 for 2 years 6 months).
        """
        if not text:
            return 0.0
        
        # Find all date mentions
        date_periods = self._extract_date_periods(text)
        
        # Group into employment periods
        employment_periods = self._identify_employment_periods(text, date_periods)
        
        # Calculate total experience
        total_years = self._calculate_total_experience(employment_periods)
        
        return total_years
    
    def _extract_date_periods(self, text: str) -> List[Tuple[datetime, datetime, int]]:
        """Extract all date ranges from text. Returns (start, end, position) tuples."""
        periods = []
        text_lower = text.lower()
        
        # Look for "Start - End" or "Start to End" patterns
        for pattern in self.date_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            dates = []
            
            for match in matches:
                try:
                    date_str = match.group(0)
                    date_obj = date_parser.parse(date_str, fuzzy=True, default=datetime(2000, 1, 1))
                    dates.append((date_obj, match.start()))
                except:
                    continue
            
            # Pair consecutive dates as periods
            if len(dates) >= 2:
                dates.sort(key=lambda x: x[1])  # Sort by position
                for i in range(0, len(dates) - 1, 2):
                    start_date = dates[i][0]
                    end_date = dates[i+1][0] if i+1 < len(dates) else datetime.now()
                    
                    # Validate date range
                    if end_date > start_date:
                        periods.append((start_date, end_date, dates[i][1]))
        
        # Also check for "Present" or "Current"
        present_matches = re.finditer(
            r'\b(present|current|now|ongoing)\b',
            text_lower,
            re.IGNORECASE
        )
        
        for match in present_matches:
            # Look for date before "Present"
            before_text = text[max(0, match.start()-30):match.start()]
            date_before = re.search(r'\b(\d{4}|\w+\s+\d{4})\b', before_text)
            if date_before:
                try:
                    start_date = date_parser.parse(date_before.group(0), fuzzy=True, default=datetime(2000, 1, 1))
                    periods.append((start_date, datetime.now(), match.start()))
                except:
                    pass
        
        return periods
    
    def _identify_employment_periods(
        self, 
        text: str, 
        date_periods: List[Tuple[datetime, datetime, int]]
    ) -> List[Dict]:
        """
        Identify which date periods are employment periods.
        Returns list of dicts with start, end, is_internship flag.
        """
        employment_periods = []
        text_lower = text.lower()
        
        for start_date, end_date, position in date_periods:
            # Find text around this date range (100 chars before/after)
            context_start = max(0, position - 100)
            context_end = min(len(text), position + 200)
            context = text_lower[context_start:context_end]
            
            # Check if employment-related
            is_employment = any(kw in context for kw in self.employment_keywords)
            is_internship = any(kw in context for kw in self.internship_keywords)
            
            if is_employment:
                employment_periods.append({
                    'start': start_date,
                    'end': end_date,
                    'is_internship': is_internship
                })
        
        return employment_periods
    
    def _calculate_total_experience(self, periods: List[Dict]) -> float:
        """
        Calculate total years of experience.
        Full-time: 1x, Internships: 0.5x
        Handle overlapping periods (use union, not sum).
        """
        if not periods:
            return 0.0
        
        # Sort by start date
        periods.sort(key=lambda x: x['start'])
        
        # Merge overlapping periods
        merged_periods = []
        for period in periods:
            if not merged_periods:
                merged_periods.append(period)
            else:
                last_period = merged_periods[-1]
                # Check if overlapping
                if period['start'] <= last_period['end']:
                    # Merge: extend end date if needed
                    last_period['end'] = max(last_period['end'], period['end'])
                else:
                    merged_periods.append(period)
        
        # Calculate total days
        total_days = 0
        for period in merged_periods:
            start = period['start']
            end = period['end']
            days = (end - start).days
            multiplier = 0.5 if period['is_internship'] else 1.0
            total_days += days * multiplier
        
        # Convert to years (365.25 days/year)
        total_years = total_days / 365.25
        
        return round(max(0.0, total_years), 1)


class EducationClassifier:
    """Classify education level from text using keyword matching."""
    
    def __init__(self):
        # Education level keywords with hierarchy (higher = more advanced)
        self.education_patterns = {
            'doctorate': {
                'keywords': ['phd', 'ph.d', 'ph.d.', 'doctorate', 'd.phil', 'doctor of', 'doctoral'],
                'priority': 4
            },
            'masters': {
                'keywords': ['master', 'ms', 'm.s', 'm.sc', 'mba', 'm.a', 'm.eng', 'm.tech', 'm.sc.'],
                'priority': 3
            },
            'bachelors': {
                'keywords': ['bachelor', 'bs', 'b.s', 'b.sc', 'b.a', 'b.eng', 'btech', 'be', 'b.e', 'bachelor\'s'],
                'priority': 2
            },
            'high_school': {
                'keywords': ['high school', 'diploma', 'certificate', 'ged', 'secondary'],
                'priority': 1
            }
        }
    
    def classify(self, text: str) -> str:
        """
        Classify education level from text.
        Returns: 'high_school', 'bachelors', 'masters', or 'doctorate'
        """
        if not text:
            return 'bachelors'  # Default
        
        text_lower = text.lower()
        found_levels = []
        
        # Find all degree mentions
        for level, patterns in self.education_patterns.items():
            for keyword in patterns['keywords']:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                if pattern.search(text_lower):
                    found_levels.append((level, patterns['priority']))
                    break
        
        if not found_levels:
            return 'bachelors'  # Default
        
        # Return highest level found
        found_levels.sort(key=lambda x: x[1], reverse=True)
        return found_levels[0][0]


class JobDescriptionParser:
    """Parse job description and extract structured criteria using NLP."""
    
    def __init__(self, skills_extractor: SkillsExtractor, education_classifier: EducationClassifier):
        self.skills_extractor = skills_extractor
        self.education_classifier = education_classifier
        
        # Required skills indicators
        self.required_keywords = [
            'required', 'must have', 'essential', 'mandatory',
            'required skills', 'must possess', 'should have',
            'requirements:', 'must:', 'essential:'
        ]
        
        # Preferred skills indicators
        self.preferred_keywords = [
            'preferred', 'nice to have', 'bonus', 'advantageous',
            'preferred skills', 'nice to have', 'would be nice',
            'preferred:', 'nice to have:', 'bonus:'
        ]
        
        # Experience requirement patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)[-+]?\s*years?\s*experience',
            r'(\d+)\+?\s*yrs?\s*experience'
        ]
    
    def parse(self, text: str) -> Dict:
        """
        Parse job description and extract structured data.
        Returns dict matching JobCriteria schema.
        """
        if not text:
            return self._default_job_criteria()
        
        # Extract position/title
        position = self._extract_position(text)
        
        # Extract required and preferred skills
        required_skills, preferred_skills = self._extract_skills_sections(text)
        
        # Extract minimum experience
        min_experience = self._extract_min_experience(text)
        
        # Extract education level
        education_level = self.education_classifier.classify(text)
        
        # Extract industry, company size, remote work
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
    
    def _default_job_criteria(self) -> Dict:
        """Return default job criteria."""
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
    
    def _extract_position(self, text: str) -> str:
        """Extract job position/title."""
        lines = text.split('\n')[:15]  # Check first 15 lines
        
        # Look for position indicators
        for line in lines:
            line_clean = line.strip()
            if 5 < len(line_clean) < 100:
                # Check if it looks like a job title
                if not any(char.isdigit() for char in line_clean[:15]):
                    if line_clean[0].isupper() or line_clean.startswith('##'):
                        # Remove markdown
                        line_clean = line_clean.replace('#', '').strip()
                        return line_clean
        
        return 'Unknown Position'
    
    def _extract_skills_sections(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract required and preferred skills separately."""
        text_lower = text.lower()
        all_skills = self.skills_extractor.extract(text)
        
        required_skills = []
        preferred_skills = []
        
        # Find required skills section
        required_section = self._find_section(text, self.required_keywords)
        if required_section:
            required_skills = [
                s for s in all_skills 
                if s.lower() in required_section.lower()
            ]
        
        # Find preferred skills section
        preferred_section = self._find_section(text, self.preferred_keywords)
        if preferred_section:
            preferred_skills = [
                s for s in all_skills 
                if s.lower() in preferred_section.lower()
            ]
        
        # If no clear sections, assume all are required
        if not required_skills and not preferred_skills:
            required_skills = all_skills
        
        return required_skills, preferred_skills
    
    def _find_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Find section containing keywords."""
        text_lower = text.lower()
        for keyword in keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                # Extract section (next 500 chars or until next section)
                section_end = min(len(text), idx + 500)
                section = text[idx:section_end]
                return section
        return None
    
    def _extract_min_experience(self, text: str) -> int:
        """Extract minimum experience years requirement."""
        for pattern in self.experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
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
            'healthcare': ['healthcare', 'medical', 'hospital', 'health'],
            'saas': ['saas', 'software as a service', 'cloud software'],
            'ecommerce': ['ecommerce', 'e-commerce', 'retail', 'online retail'],
            'education': ['education', 'edtech', 'learning', 'educational'],
            'manufacturing': ['manufacturing', 'industrial', 'production'],
            'consulting': ['consulting', 'consultancy', 'advisory']
        }
        
        text_lower = text.lower()
        for industry, keywords in industry_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return industry
        
        return 'general'
    
    def _extract_company_size(self, text: str) -> str:
        """Extract company size mention."""
        size_keywords = {
            'startup': ['startup', 'start-up', 'early stage', 'small team'],
            'small': ['small company', 'small team', '< 50', 'under 50'],
            'medium': ['medium', 'mid-size', '50-200', 'mid-sized'],
            'large': ['large', 'enterprise', 'fortune 500', '> 500', 'multinational']
        }
        
        text_lower = text.lower()
        for size, keywords in size_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return size
        
        return 'medium'
    
    def _extract_remote_work(self, text: str) -> bool:
        """Extract remote work preference."""
        remote_keywords = [
            'remote', 'work from home', 'wfh', 'hybrid', 'distributed',
            'remote work', 'work remotely', 'location: remote'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in remote_keywords)

