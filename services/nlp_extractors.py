"""
Pure NLP-based extractors using rule-based parsing, regex, and pattern matching.
NO LLMs involved - deterministic, fast, local processing.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
try:
    from dateutil import parser as date_parser
except ImportError:
    # Fallback if dateutil not available
    date_parser = None

from .skills_dictionary import ALL_TECH_SKILLS, normalize_skill


class SkillsExtractor:
    """Extract technical skills from text using dictionary-based pattern matching."""
    
    def __init__(self, skills_dict: Dict[str, List[str]] = None):
        if skills_dict:
            self.all_skills = set()
            for category, skills in skills_dict.items():
                self.all_skills.update([s.lower() for s in skills])
        else:
            from .skills_dictionary import ALL_TECH_SKILLS
            self.all_skills = ALL_TECH_SKILLS
        
        # Create regex pattern for all skills (word boundaries)
        pattern_parts = [f'\\b{re.escape(skill)}\\b' for skill in self.all_skills]
        # Sort by length (longest first) to match "react native" before "react"
        pattern_parts.sort(key=len, reverse=True)
        self.skills_pattern = re.compile(
            '|'.join(pattern_parts),
            re.IGNORECASE
        )
        
        # Technical context indicators
        self.tech_indicators = [
            'experience with', 'proficient in', 'skilled in', 'expertise in',
            'worked with', 'developed using', 'built with', 'technologies:',
            'skills:', 'tech stack', 'tools:', 'languages:', 'frameworks:',
            'proficient', 'experienced', 'knowledge of', 'familiar with'
        ]
    
    def extract(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching."""
        if not text:
            return []
        
        text_lower = text.lower()
        matches = []
        
        # Find all skill mentions
        for match in self.skills_pattern.finditer(text):
            skill = match.group(0).lower()
            matches.append((skill, match.start(), match.end()))
        
        # Validate context and normalize
        validated = []
        seen = set()
        
        for skill, start, end in matches:
            # Normalize skill name
            normalized = normalize_skill(skill)
            if normalized in seen:
                continue
            
            # Check surrounding context (30 chars before/after)
            context_start = max(0, start - 30)
            context_end = min(len(text), end + 30)
            context = text_lower[context_start:context_end]
            
            # Validate it's in technical context
            is_technical = any(indicator in context for indicator in self.tech_indicators)
            
            # Also check if it's in a skills section (common section headers)
            is_in_skills_section = any(
                section in text_lower[max(0, start-100):start]
                for section in ['skills', 'technical', 'technologies', 'tools', 'competencies']
            )
            
            if is_technical or is_in_skills_section or normalized in self.all_skills:
                validated.append(normalized)
                seen.add(normalized)
        
        return sorted(list(set(validated)))


class ExperienceExtractor:
    """Extract work experience years from resume text using date parsing."""
    
    def __init__(self):
        # Date patterns (various formats)
        self.date_patterns = [
            # MM/YYYY, M/YYYY
            r'\b(\d{1,2})[/-](\d{4})\b',
            # Month YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})\b',
            # Month DD, YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(\d{4})\b',
            # YYYY-MM, YYYY/MM
            r'\b(\d{4})[/-](\d{1,2})\b',
            # Full month name
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        ]
        
        self.month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        # Employment keywords
        self.employment_keywords = [
            'experience', 'worked', 'employed', 'position', 'role', 'job',
            'intern', 'internship', 'full-time', 'part-time', 'contract',
            'consultant', 'developer', 'engineer', 'analyst', 'manager',
            'consultant', 'freelance', 'volunteer'
        ]
        
        # Internship indicators
        self.internship_keywords = [
            'intern', 'internship', 'co-op', 'coop', 'trainee', 'apprentice'
        ]
        
        # Current date indicators
        self.current_indicators = ['present', 'current', 'now', 'ongoing', 'till date']
    
    def extract_experience(self, text: str) -> float:
        """Extract total years of professional experience."""
        if not text:
            return 0.0
        
        # Find all date mentions
        date_periods = self._extract_date_periods(text)
        
        # Identify employment periods
        employment_periods = self._identify_employment_periods(text, date_periods)
        
        # Calculate total experience
        total_years = self._calculate_total_experience(employment_periods)
        
        return round(total_years, 1)
    
    def _extract_date_periods(self, text: str) -> List[Tuple[datetime, datetime, int, int]]:
        """Extract all date ranges from text. Returns (start, end, start_pos, end_pos)."""
        periods = []
        text_lower = text.lower()
        
        # Find all dates
        all_dates = []
        for pattern in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    date_str = match.group(0)
                    if date_parser:
                        date_obj = date_parser.parse(date_str, fuzzy=True, default=datetime(2000, 1, 1))
                    else:
                        # Basic fallback parsing
                        date_obj = datetime(2000, 1, 1)
                    all_dates.append((date_obj, match.start(), match.end()))
                except:
                    continue
        
        # Sort by position in text
        all_dates.sort(key=lambda x: x[1])
        
        # Find date pairs (start - end patterns)
        # Look for patterns like "MM/YYYY - MM/YYYY" or "Month YYYY to Month YYYY"
        for i in range(len(all_dates) - 1):
            start_date, start_pos, _ = all_dates[i]
            
            # Look for end date within reasonable distance (200 chars)
            for j in range(i + 1, min(i + 5, len(all_dates))):
                end_date, end_pos, _ = all_dates[j]
                
                # Check if there's a separator between them
                separator = text[start_pos:end_pos]
                if any(sep in separator.lower() for sep in ['-', 'to', 'until', 'till', 'through']):
                    # Check if "Present" or "Current" is mentioned
                    if any(curr in separator.lower() for curr in self.current_indicators):
                        end_date = datetime.now()
                    periods.append((start_date, end_date, start_pos, end_pos))
                    break
        
        return periods
    
    def _identify_employment_periods(
        self,
        text: str,
        date_periods: List[Tuple[datetime, datetime, int, int]]
    ) -> List[Dict]:
        """Identify which date periods are employment periods."""
        employment_periods = []
        text_lower = text.lower()
        
        for start_date, end_date, start_pos, end_pos in date_periods:
            # Extract context around this date range (100 chars before, 200 after)
            context_start = max(0, start_pos - 100)
            context_end = min(len(text), end_pos + 200)
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
        """Calculate total years of experience. Full-time: 1x, Internships: 0.5x."""
        if not periods:
            return 0.0
        
        # Handle overlapping periods (merge them)
        merged_periods = self._merge_overlapping_periods(periods)
        
        total_days = 0
        for period in merged_periods:
            start = period['start']
            end = period['end']
            days = (end - start).days
            
            # Apply multiplier
            multiplier = 0.5 if period['is_internship'] else 1.0
            total_days += days * multiplier
        
        # Convert to years
        total_years = total_days / 365.25
        return max(0.0, total_years)
    
    def _merge_overlapping_periods(self, periods: List[Dict]) -> List[Dict]:
        """Merge overlapping employment periods."""
        if not periods:
            return []
        
        # Sort by start date
        sorted_periods = sorted(periods, key=lambda x: x['start'])
        
        merged = []
        current = sorted_periods[0].copy()
        
        for next_period in sorted_periods[1:]:
            # If periods overlap or are adjacent
            if next_period['start'] <= current['end']:
                # Merge: extend end date, use internship flag if either is internship
                current['end'] = max(current['end'], next_period['end'])
                current['is_internship'] = current['is_internship'] and next_period['is_internship']
            else:
                # No overlap, save current and start new
                merged.append(current)
                current = next_period.copy()
        
        merged.append(current)
        return merged


class EducationClassifier:
    """Classify education level from resume text using keyword matching."""
    
    def __init__(self):
        # Education level keywords with hierarchy (higher = more advanced)
        self.education_patterns = {
            'doctorate': {
                'keywords': ['phd', 'ph.d', 'doctorate', 'd.phil', 'doctor of', 'doctoral'],
                'priority': 4
            },
            'masters': {
                'keywords': ['master', 'ms', 'm.s', 'm.sc', 'mba', 'm.a', 'm.eng', 'm.ed', 'mfa'],
                'priority': 3
            },
            'bachelors': {
                'keywords': ['bachelor', 'bs', 'b.s', 'b.sc', 'b.a', 'b.eng', 'btech', 'be', 'b.com', 'b.sc'],
                'priority': 2
            },
            'high_school': {
                'keywords': ['high school', 'diploma', 'certificate', 'ged', 'hsc', 'ssc'],
                'priority': 1
            }
        }
    
    def classify(self, text: str) -> str:
        """Classify education level. Returns highest degree found."""
        if not text:
            return 'bachelors'  # Default
        
        text_lower = text.lower()
        found_levels = []
        
        # Search for degree mentions
        for level, patterns in self.education_patterns.items():
            for keyword in patterns['keywords']:
                # Use word boundary to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    found_levels.append((level, patterns['priority']))
                    break
        
        if not found_levels:
            return 'bachelors'  # Default
        
        # Return highest level found
        found_levels.sort(key=lambda x: x[1], reverse=True)
        return found_levels[0][0]

