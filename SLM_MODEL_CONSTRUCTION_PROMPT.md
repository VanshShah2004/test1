# NLP-Based Extraction System: Construction & Configuration Guide

This document provides comprehensive guidance for building a **pure NLP-based extraction system** using traditional Natural Language Processing techniques (NO LLMs involved).

**Approach**: Rule-based parsing, Named Entity Recognition (NER), pattern matching, text classification, and keyword extraction.

---

## 🎯 Part 1: NLP System Architecture Design Prompt

Use this prompt to design your NLP-based extraction system:

```
You are an expert NLP engineer specializing in rule-based information extraction systems.

**TASK**: Design a pure NLP-based resume and job description parsing system that uses NO language models (LLMs). The system must use only traditional NLP techniques.

## SYSTEM REQUIREMENTS

**Extraction Tasks**:
1. **Resume Parsing**:
   - Extract technical skills (programming languages, frameworks, tools)
   - Extract education level (high_school, bachelors, masters, doctorate)
   - Calculate total years of professional experience from dates

2. **Job Description Parsing**:
   - Extract job position/title
   - Extract required skills and preferred skills (separately)
   - Extract minimum experience years required
   - Extract education level requirement
   - Extract industry, company size, remote work preference

**Key Constraints**:
- NO LLMs or language models (no GPT, Gemini, Claude, etc.)
- Use only traditional NLP: NER, regex, pattern matching, text classification
- Must be deterministic (same input → same output)
- Fast processing (<1 second per document)
- No API costs (all local processing)
- Zero hallucinations (only extract explicit information)

**Technical Approach Required**:
- Named Entity Recognition (NER) for entities extraction
- Regular expressions for pattern matching (dates, years, skills)
- Rule-based parsing for structured data
- Keyword matching with domain dictionaries
- Text classification for education levels
- Date parsing and calculation for experience

**Libraries to Consider**:
- spaCy (NER, POS tagging, entity recognition)
- NLTK (tokenization, text processing)
- regex/re (pattern matching)
- dateparser (date extraction)
- scikit-learn (classification if needed)
- pandas (data processing)

**Output Format**:
- Must return structured JSON matching exact schemas
- Deterministic output format
- Error handling for edge cases

## YOUR DESIGN SHOULD INCLUDE:

1. **System Architecture**:
   - Pipeline stages (text preprocessing → extraction → validation → output)
   - Module breakdown (skills extractor, date parser, education classifier, etc.)
   - Data flow diagram

2. **NLP Techniques Selection**:
   - Which NER approach (spaCy models, custom patterns, or hybrid)
   - Pattern matching strategy (regex patterns for skills, dates)
   - Classification approach (rule-based vs ML classifier)
   - Keyword extraction method (dictionary-based, frequency-based)

3. **Skills Extraction Strategy**:
   - Pre-built skills dictionary (comprehensive tech stack list)
   - Pattern matching for skill mentions
   - Context-aware extraction (avoid false positives)
   - Normalization (lowercase, synonyms handling)

4. **Date/Experience Extraction**:
   - Regex patterns for date formats (MM/YYYY, Month YYYY, etc.)
   - Date parsing and calculation logic
   - Experience calculation algorithm (handling gaps, overlaps)
   - Internship detection and 0.5x weighting

5. **Education Level Classification**:
   - Keyword-based rules (bachelor, master, phd, etc.)
   - Regex patterns for degree mentions
   - Hierarchy logic (highest degree extraction)
   - Default handling for unclear cases

6. **Implementation Plan**:
   - Library installation and setup
   - Model training/configuration (if using spaCy NER)
   - Pattern development workflow
   - Testing and validation approach
   - Performance optimization

7. **Code Structure**:
   - Class architecture
   - Function breakdown
   - Error handling strategy
   - Configuration management (patterns, dictionaries)

**Provide detailed architecture, code examples, and implementation roadmap.**
```

---

## 📚 Part 2: NLP Library & Model Selection

### 2.1 spaCy Model Selection Prompt

```
Select the optimal spaCy model for NER-based resume parsing.

REQUIREMENTS:
- Extract: Skills (PROGRAMMING_LANGUAGE, FRAMEWORK, TOOL entities)
- Extract: Organizations (for companies/universities)
- Extract: Dates (for experience calculation)
- Extract: Person names (for validation)

AVAILABLE MODELS:
- en_core_web_sm (small, fast, 12MB)
- en_core_web_md (medium, more accurate, 40MB)
- en_core_web_lg (large, best accuracy, 560MB)
- en_core_web_trf (transformer-based, 438MB, best but slower)

CONSTRAINTS:
- Must run locally
- Prefer smaller models for speed
- Can combine spaCy NER with custom patterns

RECOMMENDATION:
- Start with: en_core_web_sm + custom entity patterns
- If accuracy insufficient: upgrade to en_core_web_md
- Custom entity recognition for tech skills (not in standard NER)

PROVIDE:
1. Model recommendation with rationale
2. Custom entity pattern examples for tech skills
3. Hybrid approach (spaCy + custom patterns)
4. Installation and setup instructions
```

### 2.2 Skills Dictionary Construction Prompt

```
Create a comprehensive technical skills dictionary for keyword-based extraction.

CATEGORIES NEEDED:
1. Programming Languages (Python, Java, C++, JavaScript, etc.)
2. Web Frameworks (React, Angular, Vue, Django, Flask, etc.)
3. Databases (MySQL, PostgreSQL, MongoDB, Redis, etc.)
4. Cloud Platforms (AWS, Azure, GCP services)
5. DevOps Tools (Docker, Kubernetes, Jenkins, Git, etc.)
6. Data Science (Pandas, NumPy, TensorFlow, PyTorch, etc.)
7. Mobile (React Native, Flutter, Swift, Kotlin, etc.)
8. Testing (JUnit, pytest, Selenium, etc.)

REQUIREMENTS:
- Include common variations and aliases (JS/JavaScript, ML/Machine Learning)
- Include version numbers if relevant (Python 3.9, React 18, etc.)
- Normalized format (lowercase, standardized naming)
- Comprehensive coverage (1000+ skills)

OUTPUT FORMAT:
```python
TECH_SKILLS_DICT = {
    "programming_languages": ["python", "java", "javascript", ...],
    "frameworks": ["react", "angular", "django", ...],
    "databases": ["mysql", "postgresql", "mongodb", ...],
    # ... etc
}
```

PROVIDE:
1. Complete skills dictionary (Python dict structure)
2. Normalization rules (handling variations)
3. Matching strategy (exact match vs fuzzy match)
4. Context-aware extraction (avoid false positives like "React" in "Reacted to")
```

---

## 🔧 Part 3: Rule-Based Extraction Patterns

### 3.1 Resume Skills Extraction Pattern

**Python Implementation Template**:

```python
"""
Rule-based skills extraction using:
1. Pre-built skills dictionary
2. Pattern matching (case-insensitive)
3. Context validation (avoid false positives)
4. Normalization and deduplication
"""

import re
from typing import List, Set
from collections import defaultdict

class SkillsExtractor:
    def __init__(self, skills_dict: dict):
        self.skills_dict = skills_dict
        # Flatten all skills into one set for fast lookup
        self.all_skills = set()
        for category, skills in skills_dict.items():
            self.all_skills.update([s.lower() for s in skills])
        
        # Create regex pattern for all skills (word boundaries)
        pattern_parts = [f'\\b{re.escape(skill)}\\b' for skill in self.all_skills]
        self.skills_pattern = re.compile(
            '|'.join(pattern_parts), 
            re.IGNORECASE
        )
    
    def extract(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching.
        Returns normalized, deduplicated list.
        """
        # Find all skill mentions
        matches = self.skills_pattern.findall(text.lower())
        
        # Normalize and deduplicate
        extracted = list(set([m.lower() for m in matches]))
        
        # Context validation: Remove false positives
        # (e.g., "React" in "Reacted to feedback" - check surrounding words)
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
                # Check surrounding context (10 chars before/after)
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                context = text_lower[start:end]
                
                # Technical context indicators
                tech_indicators = [
                    'experience with', 'proficient in', 'skilled in',
                    'worked with', 'developed using', 'built with',
                    'technologies:', 'skills:', 'expertise in'
                ]
                
                # If context suggests technical mention, keep it
                if any(indicator in context for indicator in tech_indicators):
                    if skill not in validated:
                        validated.append(skill)
                    break
        
        return validated
```

**Configuration Prompt for Skills Extraction**:

```
DESIGN SKILLS EXTRACTION RULES:

1. **Dictionary Structure**:
   - Organize by categories (programming_languages, frameworks, etc.)
   - Include common variations (JS/JavaScript, ML/Machine Learning)
   - Include version numbers (React 18, Python 3.9)

2. **Pattern Matching Rules**:
   - Use word boundaries (\b) to avoid partial matches
   - Case-insensitive matching
   - Handle abbreviations (JS → JavaScript mapping)

3. **Context Validation Rules**:
   - Check for technical context keywords
   - Validate against POS tags (nouns/adjectives in technical context)
   - Remove skills from non-technical sentences

4. **Normalization Rules**:
   - Convert to lowercase
   - Standardize naming (JavaScript not JS in output)
   - Handle synonyms (Git/GitHub → git)

5. **False Positive Prevention**:
   - Exclude common words (e.g., "python" as snake, not language)
   - Check surrounding context before extraction
   - Validate against known technical patterns

PROVIDE:
- Complete regex patterns
- Context validation rules
- Normalization function
- Example edge cases and handling
```

### 3.2 Experience/Date Extraction Pattern

**Python Implementation Template**:

```python
"""
Rule-based date and experience extraction using:
1. Regex patterns for various date formats
2. Date parsing and calculation
3. Employment period detection
4. Experience calculation with internship weighting
"""

import re
from datetime import datetime
from dateutil import parser as date_parser
from typing import List, Tuple, Optional

class ExperienceExtractor:
    def __init__(self):
        # Common date patterns in resumes
        self.date_patterns = [
            # MM/YYYY, M/YYYY
            r'\b(\d{1,2})[/-](\d{4})\b',
            # Month YYYY, Month YYYY
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
            'contract', 'consultant'
        ]
        
        # Internship indicators
        self.internship_keywords = [
            'intern', 'internship', 'co-op', 'coop', 'trainee'
        ]
    
    def extract_experience(self, text: str) -> float:
        """
        Extract total years of professional experience.
        Returns float (e.g., 2.5 for 2 years 6 months).
        """
        # Find all date mentions
        date_periods = self._extract_date_periods(text)
        
        # Group into employment periods
        employment_periods = self._identify_employment_periods(text, date_periods)
        
        # Calculate total experience
        total_years = self._calculate_total_experience(employment_periods)
        
        return total_years
    
    def _extract_date_periods(self, text: str) -> List[Tuple[datetime, datetime]]:
        """Extract all date ranges from text."""
        periods = []
        text_lower = text.lower()
        
        # Look for "Start - End" or "Start to End" patterns
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            dates = []
            for match in matches:
                try:
                    date_str = match.group(0)
                    date_obj = date_parser.parse(date_str, fuzzy=True)
                    dates.append((date_obj, match.start()))
                except:
                    continue
            
            # Pair consecutive dates as periods
            if len(dates) >= 2:
                dates.sort(key=lambda x: x[1])  # Sort by position
                for i in range(0, len(dates) - 1, 2):
                    start_date = dates[i][0]
                    end_date = dates[i+1][0] if i+1 < len(dates) else datetime.now()
                    if end_date > start_date:
                        periods.append((start_date, end_date))
        
        return periods
    
    def _identify_employment_periods(
        self, 
        text: str, 
        date_periods: List[Tuple[datetime, datetime]]
    ) -> List[dict]:
        """
        Identify which date periods are employment periods.
        Returns list of dicts with start, end, is_internship flag.
        """
        employment_periods = []
        text_lower = text.lower()
        
        for start_date, end_date in date_periods:
            # Find text around this date range
            date_str = start_date.strftime('%Y-%m')
            idx = text_lower.find(date_str)
            
            if idx != -1:
                # Check context (50 chars before/after)
                context_start = max(0, idx - 50)
                context_end = min(len(text), idx + 100)
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
    
    def _calculate_total_experience(self, periods: List[dict]) -> float:
        """
        Calculate total years of experience.
        Full-time: 1x, Internships: 0.5x
        """
        total_days = 0
        
        for period in periods:
            start = period['start']
            end = period['end']
            days = (end - start).days
            multiplier = 0.5 if period['is_internship'] else 1.0
            total_days += days * multiplier
        
        # Convert to years (365.25 days/year)
        total_years = total_days / 365.25
        
        return round(total_years, 1)
```

**Configuration Prompt for Date Extraction**:

```
DESIGN DATE AND EXPERIENCE EXTRACTION RULES:

1. **Date Format Patterns**:
   - MM/YYYY (01/2020)
   - Month YYYY (January 2020, Jan 2020)
   - YYYY-MM (2020-01)
   - Full dates (January 15, 2020)
   - Relative dates (Present, Current, Now)

2. **Employment Detection Rules**:
   - Keywords: "Worked", "Experience", "Employed", "Position", "Role"
   - Section headers: "Experience", "Work History", "Employment"
   - Job title patterns (Software Engineer, Developer, etc.)

3. **Internship Detection**:
   - Keywords: "Intern", "Internship", "Co-op", "Trainee"
   - Apply 0.5x multiplier to duration
   - Distinguish from full-time work

4. **Experience Calculation Logic**:
   - Sum all employment periods
   - Handle overlapping periods (use union, not sum)
   - Handle gaps (ignore gaps, only count employed periods)
   - Convert to years (decimal format: 2.5 for 2 years 6 months)

5. **Edge Cases**:
   - "Present" or "Current" → use current date
   - Missing end dates → use current date
   - Invalid dates → skip or use default
   - Multiple date formats in same resume

PROVIDE:
- Complete regex patterns for all date formats
- Employment detection algorithm
- Experience calculation function with examples
- Edge case handling logic
```

### 3.3 Education Level Classification Pattern

**Python Implementation Template**:

```python
"""
Rule-based education level classification using:
1. Keyword matching for degree types
2. Hierarchy logic (highest degree)
3. Pattern matching for degree mentions
"""

import re
from typing import Optional

class EducationClassifier:
    def __init__(self):
        # Education level keywords with hierarchy (higher = more advanced)
        self.education_patterns = {
            'doctorate': {
                'keywords': ['phd', 'ph.d', 'doctorate', 'd.phil', 'doctor of'],
                'degree_types': ['phd', 'doctorate', 'doctoral'],
                'priority': 4
            },
            'masters': {
                'keywords': ['master', 'ms', 'm.s', 'm.sc', 'mba', 'm.a', 'm.eng'],
                'degree_types': ['master', 'masters', 'ms', 'mba'],
                'priority': 3
            },
            'bachelors': {
                'keywords': ['bachelor', 'bs', 'b.s', 'b.sc', 'b.a', 'b.eng', 'btech', 'be'],
                'degree_types': ['bachelor', 'bachelors', 'bs', 'ba', 'btech'],
                'priority': 2
            },
            'high_school': {
                'keywords': ['high school', 'diploma', 'certificate', 'ged'],
                'degree_types': ['high school', 'diploma'],
                'priority': 1
            }
        }
        
        # Regex patterns for degree mentions
        self.degree_pattern = re.compile(
            r'\b(?:' + '|'.join([
                kw for patterns in self.education_patterns.values()
                for kw in patterns['keywords']
            ]) + r')\b',
            re.IGNORECASE
        )
    
    def classify(self, text: str) -> str:
        """
        Classify education level from text.
        Returns: 'high_school', 'bachelors', 'masters', or 'doctorate'
        """
        text_lower = text.lower()
        
        # Find all degree mentions
        found_levels = []
        for level, patterns in self.education_patterns.items():
            for keyword in patterns['keywords']:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    found_levels.append((level, patterns['priority']))
                    break
        
        if not found_levels:
            return 'bachelors'  # Default
        
        # Return highest level found
        found_levels.sort(key=lambda x: x[1], reverse=True)
        return found_levels[0][0]
```

**Configuration Prompt for Education Classification**:

```
DESIGN EDUCATION LEVEL CLASSIFICATION RULES:

1. **Keyword Dictionary**:
   - Degree types: Bachelor, Master, PhD, High School
   - Common abbreviations (BS, MS, PhD, MBA)
   - International equivalents (B.Tech, M.Sc, etc.)

2. **Hierarchy Logic**:
   - doctorate > masters > bachelors > high_school
   - Always return highest degree mentioned
   - Priority-based matching

3. **Pattern Matching**:
   - Match degree keywords in context
   - Validate with surrounding words (degree, in, of)
   - Handle variations and abbreviations

4. **Context Validation**:
   - Check for "Degree in X" pattern
   - Verify it's a degree mention, not just word occurrence
   - Education section detection

5. **Default Handling**:
   - If no clear match: default to "bachelors"
   - If multiple degrees: return highest
   - Handle unclear cases conservatively

PROVIDE:
- Complete keyword dictionary
- Classification algorithm
- Priority hierarchy
- Edge case examples
```

---

## 🏗️ Part 4: Complete NLP Service Architecture

### 4.1 NLP Service Implementation Prompt

```
DESIGN A COMPLETE NLP-BASED EXTRACTION SERVICE:

**Service Structure**:
```
services/
├── nlp_extractor.py          # Main NLP extraction service (NO LLM)
│   ├── SkillsExtractor
│   ├── ExperienceExtractor
│   ├── EducationClassifier
│   └── JobDescriptionParser
├── pdf_utils.py              # Existing PDF extraction
└── skills_dictionary.py      # Skills dictionary data
```

**Requirements**:
1. **ResumeScreenerAgent** replacement:
   - Use NLP extractors instead of LLM
   - Same output format (ResumeData model)
   - Deterministic, fast, accurate

2. **JobDescriptionParserAgent** replacement:
   - NLP-based job criteria extraction
   - Extract: position, skills (required/preferred), experience, education, etc.
   - Same output format (JobCriteria model)

3. **No LLM Dependency**:
   - Zero API calls
   - Pure local processing
   - No language model libraries

**Implementation Details Needed**:
1. Class architecture for each extractor
2. Integration pattern (how extractors work together)
3. Error handling and fallbacks
4. Configuration management (patterns, dictionaries)
5. Performance optimization
6. Testing strategy

PROVIDE:
- Complete Python implementation
- Class structure and methods
- Integration code
- Configuration examples
- Usage examples
```

---

## 📊 Part 5: Job Description NLP Parsing

### 5.1 Job Description Extraction Patterns

**Python Implementation Template**:

```python
"""
NLP-based job description parsing using:
1. Position/title extraction (section headers, bold text)
2. Skills extraction (required vs preferred keywords)
3. Experience requirement extraction (numeric patterns)
4. Education requirement classification
"""

import re
from typing import Dict, List

class JobDescriptionParser:
    def __init__(self, skills_extractor, education_classifier):
        self.skills_extractor = skills_extractor
        self.education_classifier = education_classifier
        
        # Required skills indicators
        self.required_keywords = [
            'required', 'must have', 'essential', 'mandatory',
            'required skills', 'must possess', 'should have'
        ]
        
        # Preferred skills indicators
        self.preferred_keywords = [
            'preferred', 'nice to have', 'bonus', 'advantageous',
            'preferred skills', 'nice to have', 'would be nice'
        ]
        
        # Experience requirement patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
            r'(\d+)[-+]?\s*years?\s*experience'
        ]
    
    def parse(self, text: str) -> Dict:
        """
        Parse job description and extract structured data.
        Returns dict matching JobCriteria schema.
        """
        text_lower = text.lower()
        
        # Extract position/title (usually in first few lines or header)
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
    
    def _extract_position(self, text: str) -> str:
        """Extract job position/title."""
        # Common patterns: First line, bold text, "Position:" label
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        # Look for position indicators
        for line in lines:
            line_clean = line.strip()
            if len(line_clean) > 5 and len(line_clean) < 100:
                # Check if it looks like a job title
                if not any(char.isdigit() for char in line_clean[:10]):
                    if line_clean[0].isupper():
                        return line_clean
        
        return 'Unknown Position'
    
    def _extract_skills_sections(self, text: str) -> tuple[List[str], List[str]]:
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
                years = int(match.group(1))
                return years
        return 0
    
    def _extract_industry(self, text: str) -> str:
        """Extract industry from job description."""
        industry_keywords = {
            'fintech': ['fintech', 'financial technology', 'banking'],
            'healthcare': ['healthcare', 'medical', 'hospital'],
            'saas': ['saas', 'software as a service', 'cloud software'],
            'ecommerce': ['ecommerce', 'e-commerce', 'retail'],
            # Add more industries
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
            'small': ['small company', 'small team', '< 50'],
            'medium': ['medium', 'mid-size', '50-200'],
            'large': ['large', 'enterprise', 'fortune 500', '> 500']
        }
        
        text_lower = text.lower()
        for size, keywords in size_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return size
        
        return 'medium'
    
    def _extract_remote_work(self, text: str) -> bool:
        """Extract remote work preference."""
        remote_keywords = ['remote', 'work from home', 'wfh', 'hybrid', 'distributed']
        text_lower = text.lower()
        return any(kw in text_lower for kw in remote_keywords)
```

---

## 🧪 Part 6: Testing & Validation Strategy

### 6.1 NLP System Testing Prompt

```
CREATE A TESTING STRATEGY FOR NLP-BASED EXTRACTION SYSTEM:

**Test Categories**:

1. **Skills Extraction Tests**:
   - Test with resumes containing 10-50 skills
   - Test edge cases (skill names in non-technical context)
   - Test normalization (JS → JavaScript)
   - Test false positive prevention

2. **Experience Extraction Tests**:
   - Test various date formats (MM/YYYY, Month YYYY, etc.)
   - Test with internships (0.5x weighting)
   - Test with gaps in employment
   - Test with "Present" dates
   - Test overlapping periods

3. **Education Classification Tests**:
   - Test all education levels
   - Test abbreviations (BS, MS, PhD)
   - Test multiple degrees (return highest)
   - Test unclear cases (default handling)

4. **Job Description Parsing Tests**:
   - Test required vs preferred skills separation
   - Test experience requirement extraction
   - Test position title extraction
   - Test industry/remote work detection

5. **Performance Tests**:
   - Speed: <1 second per document
   - Accuracy: >90% precision, >85% recall
   - Deterministic: Same input → Same output

6. **Integration Tests**:
   - End-to-end resume parsing
   - End-to-end job description parsing
   - Output format validation (JSON schema)
   - Error handling (malformed PDFs, empty text)

PROVIDE:
- Test dataset (sample resumes, job descriptions)
- Unit tests for each extractor
- Integration test suite
- Performance benchmarks
- Accuracy evaluation methodology
```

---

## 📦 Part 7: Implementation Checklist

```
NLP-BASED SYSTEM IMPLEMENTATION CHECKLIST:

□ 1. Environment Setup
   □ Install spaCy and download model (en_core_web_sm)
   □ Install required packages (regex, dateparser, etc.)
   □ Create skills dictionary file
   □ Set up project structure

□ 2. Core Extractors Implementation
   □ SkillsExtractor class (dictionary-based)
   □ ExperienceExtractor class (date parsing)
   □ EducationClassifier class (rule-based)
   □ JobDescriptionParser class (composite)

□ 3. Pattern Development
   □ Date regex patterns (all formats)
   □ Skills matching patterns
   □ Employment detection patterns
   □ Education keyword patterns

□ 4. Dictionary Creation
   □ Comprehensive skills dictionary (1000+ skills)
   □ Industry keywords dictionary
   □ Company size keywords
   □ Education level keywords

□ 5. Integration
   □ Create NLPExtractorService (replaces LLMService)
   □ Update ResumeScreenerAgent to use NLP
   □ Update JobDescriptionParserAgent to use NLP
   □ Maintain same output interfaces (ResumeData, JobCriteria)

□ 6. Testing
   □ Unit tests for each extractor
   □ Integration tests
   □ Edge case handling
   □ Performance benchmarks

□ 7. Validation
   □ Compare NLP output vs LLM output (baseline)
   □ Measure accuracy (precision/recall)
   □ Validate JSON output format
   □ Test deterministic behavior

□ 8. Documentation
   □ Document all patterns and rules
   □ Document skills dictionary structure
   □ Create usage examples
   □ Document edge case handling
```

---

## 🚀 Quick Start: NLP Service Implementation

### Example: Complete NLP Service Class

```python
# services/nlp_extractor.py
from typing import Dict, List
from .skills_extractor import SkillsExtractor
from .experience_extractor import ExperienceExtractor
from .education_classifier import EducationClassifier
from .job_description_parser import JobDescriptionParser

class NLPExtractorService:
    """
    Pure NLP-based extraction service (NO LLMs).
    Uses rule-based parsing, NER, pattern matching.
    """
    
    def __init__(self):
        # Initialize extractors
        from .skills_dictionary import TECH_SKILLS_DICT
        self.skills_extractor = SkillsExtractor(TECH_SKILLS_DICT)
        self.experience_extractor = ExperienceExtractor()
        self.education_classifier = EducationClassifier()
        self.job_parser = JobDescriptionParser(
            self.skills_extractor,
            self.education_classifier
        )
    
    def parse_resume(self, text: str) -> Dict:
        """Parse resume text and extract structured data."""
        return {
            'extracted_skills': self.skills_extractor.extract(text),
            'education_level': self.education_classifier.classify(text),
            'total_experience_years': self.experience_extractor.extract_experience(text)
        }
    
    def parse_job_description(self, text: str) -> Dict:
        """Parse job description and extract criteria."""
        return self.job_parser.parse(text)
```

---

## 📝 Notes

- **No LLMs**: This entire approach uses zero language models
- **Deterministic**: Same input always produces same output
- **Fast**: Local processing, no API calls, <1 second per document
- **Free**: No API costs, only library dependencies
- **Maintainable**: Rule-based system is transparent and debuggable
- **Extensible**: Easy to add new patterns, keywords, rules

---

**Last Updated**: Pure NLP-based approach (no LLMs)
**Recommended Libraries**: spaCy (NER), regex (patterns), dateparser (dates), NLTK (text processing)
