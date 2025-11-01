# Code Flow Documentation

This document details the architecture, data flow, and component interactions in the LLM-Powered Resume Screening System.

## 📐 Architecture Overview

The system follows a multi-agent architecture with two main evaluation paths:

1. **Orchestrator Flow**: Quick rule-based matching for initial screening
2. **Structured Scoring Flow**: Comprehensive LLM-based evaluation with weighted criteria

```
┌─────────────────────────────────────────────────────────────┐
│                     Entry Point: pipeline.py                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────┐          ┌──────────────────────┐
│ Orchestrator  │          │ Structured Scoring    │
│   Flow        │          │   Agent (Batch)       │
└───────┬───────┘          └──────────┬───────────┘
        │                              │
        │                              │
        └──────────────┬───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Gap Analysis    │
              │  & Ranking      │
              └─────────────────┘
```

## 🔄 Detailed Flow

### Phase 1: Orchestrator Flow (`main_orchestrator.py`)

**Purpose**: Quick initial matching using rule-based logic

```
pipeline.py::run_pipeline()
    │
    ├──> main_orchestrator.run(job_pdf, resume_pdfs)
    │       │
    │       ├──> JobDescriptionParserAgent.parse(job_pdf)
    │       │       │
    │       │       ├──> services/pdf_utils.extract_text_from_pdf()
    │       │       │       └──> PyPDF2.PdfReader() extracts text
    │       │       │
    │       │       ├──> services/llm.LLMService.generate()
    │       │       │       └──> LangChain + Google Gemini API
    │       │       │
    │       │       └──> Returns: JobCriteria (Pydantic model)
    │       │               - position
    │       │               - required_skills[]
    │       │               - preferred_skills[]
    │       │               - min_experience_years
    │       │               - education_level
    │       │               - industry, company_size, remote_work
    │       │
    │       ├──> ResumeScreenerAgent.parse(resume_pdf) [for each resume]
    │       │       │
    │       │       ├──> services/pdf_utils.extract_text_from_pdf()
    │       │       │
    │       │       ├──> services/llm.LLMService.generate()
    │       │       │       └──> LLM extracts:
    │       │       │           - extracted_skills[]
    │       │       │           - education_level
    │       │       │           - total_experience_years (parsed from dates)
    │       │       │
    │       │       └──> Returns: ResumeData (Pydantic model)
    │       │               - file_path
    │       │               - raw_text
    │       │               - extracted_skills[]
    │       │               - education_level
    │       │               - total_experience_years
    │       │
    │       └──> MatchmakerAgent.score(resume, job) [for each resume]
    │               │
    │               ├──> Skills Matching
    │               │       └──> Set intersection: (job_skills ∩ resume_skills) / job_skills
    │               │
    │               ├──> Experience Matching
    │               │       └──> Compares resume.total_experience_years vs job.min_experience_years
    │               │           - If resume_exp >= job_required: 100 points
    │               │           - If resume_exp > 0 but < required: 40-90 points (proportional)
    │               │           - If resume_exp = 0 but job requires: 30 points
    │               │
    │               ├──> Education Matching
    │               │       └──> Compares education levels (high_school < bachelors < masters < doctorate)
    │               │
    │               └──> Returns: MatchResult
    │                       - job: JobCriteria
    │                       - resume: ResumeData
    │                       - scores: MatchScores (skills_match, experience_match, education_match, overall_score)
    │
    └──> Returns: List[MatchResult] (one per resume)
```

### Phase 2: Structured Scoring Flow (`structured_scoring_agent.py`)

**Purpose**: Comprehensive LLM-based evaluation with weighted criteria

```
pipeline.py::run_pipeline()
    │
    ├──> StructuredScoringAgent.score_resumes_batch(resume_paths, job_description_path, criteria_requirements)
    │       │
    │       ├──> Extract text from all PDFs (parallel)
    │       │       ├──> Extract job description text
    │       │       └──> Extract all resume texts
    │       │
    │       ├──> Create batch scoring prompt
    │       │       └──> structured_scoring_agent.create_batch_scoring_prompt()
    │       │               - Includes all resumes for comparative evaluation
    │       │               - Includes job description
    │       │               - Includes criteria weights
    │       │               - Instructs LLM to score relatively across candidates
    │       │
    │       ├──> LLM Batch Evaluation
    │       │       └──> Single LLM call evaluates all resumes together
    │       │               - Returns JSON with scores for each resume
    │       │               - Each resume scored on all criteria:
    │       │                 * raw_score (0-100)
    │       │                 * weight_given
    │       │                 * normalized_percentage
    │       │                 * weighted_contribution
    │       │               - total_score (sum of weighted contributions)
    │       │
    │       ├──> Parse LLM Response
    │       │       └──> structured_scoring_agent.parse_llm_response()
    │       │               - Extracts JSON from LLM response
    │       │               - Handles markdown code blocks
    │       │               - Error handling with fallback
    │       │
    │       └──> Returns: List[Dict[str, Any]]
    │               Each dict contains:
    │               - success: bool
    │               - scoring_result: {
    │                   "technical_skills": {...},
    │                   "experience": {...},
    │                   "education": {...},
    │                   ... (other criteria)
    │                   "total_score": float,
    │                   "metadata": {...}
    │                 }
    │
    └──> Fallback (if batch fails)
            └──> _fallback_individual_scoring()
                    └──> Scores each resume individually (sequential)
```

### Phase 3: Gap Analysis & Enrichment (`pipeline.py`)

**Purpose**: Combine orchestrator and structured scoring results for gap analysis

```
pipeline.py::run_pipeline()
    │
    ├──> For each resume result:
    │       │
    │       ├──> Extract MatchResult from orchestrator results
    │       │
    │       ├──> Calculate Missing Skills
    │       │       └──> Set difference: job.required_skills - resume.extracted_skills
    │       │
    │       ├──> Calculate Experience Gap
    │       │       └──> max(0, job.min_experience_years - resume.total_experience_years)
    │       │
    │       ├──> Calculate Education Gap
    │       │       └──> Check if resume.education_level >= job.education_level
    │       │
    │       └──> Add gap_analysis to scoring_result
    │               {
    │                 "missing_skills": [...],
    │                 "experience_gap_years": float | None,
    │                 "education_gap": str | None
    │               }
    │
    └──> Display results and save to outputs/
```

### Phase 4: Output Generation (`pipeline.py`)

```
pipeline.py::run_pipeline()
    │
    ├──> Rank candidates by total_score (descending)
    │
    ├──> Save to outputs/scoring_results.json
    │       └──> Full structured results
    │
    ├──> Save to outputs/scoring_results_YYYYMMDD_HHMMSS.json
    │       └──> Timestamped snapshot
    │
    └──> Save to outputs/scoring_results_flat.csv
            └──> Flattened for Excel/analysis
                - resume_path
                - job_description_path
                - total_score
                - {criterion}__raw_score
                - {criterion}__weight_given
                - {criterion}__normalized_percentage
                - {criterion}__weighted_contribution
                - missing_skills
                - experience_gap_years
                - education_gap
```

## 🧩 Component Details

### Services Layer

#### `services/llm.py` - LLMService
- **Purpose**: Wrapper around LangChain + Google Gemini
- **Key Methods**:
  - `__init__(model, temperature)`: Initialize with API key from environment
  - `generate(system_prompt, human_prompt)`: Generate LLM response
- **Dependencies**: `langchain`, `langchain-google-genai`, `python-dotenv`

#### `services/pdf_utils.py` - PDF Text Extraction
- **Purpose**: Extract text from PDF files
- **Key Functions**:
  - `extract_text_from_pdf(pdf_path)`: Extracts all text from PDF pages
- **Dependencies**: `PyPDF2`
- **Error Handling**: Raises `FileNotFoundError` if PDF missing, `ValueError` if no text found

### Agents Layer

#### `agents/job_description_parser_agent.py` - JobDescriptionParserAgent
- **Input**: PDF path
- **Output**: `JobCriteria` (Pydantic model)
- **Process**:
  1. Extract PDF text
  2. Send to LLM with structured prompt
  3. Parse JSON response
  4. Validate and return `JobCriteria`

#### `agents/resume_screener_agent.py` - ResumeScreenerAgent
- **Input**: PDF path
- **Output**: `ResumeData` (Pydantic model)
- **Process**:
  1. Extract PDF text
  2. Send to LLM with extraction prompt
  3. Parse JSON response (skills, education, experience)
  4. **Experience Calculation**:
     - LLM calculates from employment dates
     - Full-time work = 1x duration
     - Internships = 0.5x duration
     - Returns as float (e.g., 2.5 years)
  5. Validate and return `ResumeData`

#### `agents/matchmaker_agent.py` - MatchmakerAgent
- **Input**: `ResumeData`, `JobCriteria`
- **Output**: `MatchResult` with `MatchScores`
- **Scoring Logic**:
  - **Skills Match**: `(job_skills ∩ resume_skills) / job_skills * 100`
  - **Experience Match**: 
    - 100 if resume_exp >= job_required OR job doesn't require experience
    - 40-90 (proportional) if resume_exp > 0 but < required
    - 30 if resume_exp = 0 but job requires
  - **Education Match**: 
    - 100 if resume_edu >= job_edu
    - 70 if resume_edu < job_edu
  - **Overall**: `0.6 * skills + 0.25 * experience + 0.15 * education`

### Scoring Layer

#### `structured_scoring_agent.py` - StructuredScoringAgent
- **Purpose**: LLM-based comprehensive scoring with weighted criteria
- **Key Methods**:
  - `normalize_weights(criteria_requirements)`: Normalizes weights to percentages
  - `create_scoring_prompt()`: Single resume scoring prompt
  - `create_batch_scoring_prompt()`: Batch comparative scoring prompt
  - `score_resume()`: Score single resume (backward compatibility)
  - `score_resumes_batch()`: **Primary method** - scores all resumes together
  - `parse_llm_response()`: Extract JSON from LLM response
  - `format_results()`: Pretty-print results

**Batch Scoring Benefits**:
- Reduces hallucination through relative comparison
- More consistent scoring across candidates
- Single LLM call (faster, cheaper)

**Criteria Evaluated**:
- Technical Skills
- Experience
- Education
- Presentation
- Certifications (optional)
- Projects (optional)
- Soft Skills (optional)
- Industry Knowledge (optional)

### Models Layer

#### `common/models.py` - Pydantic Models

**JobCriteria**:
```python
- position: str
- required_skills: List[str]
- preferred_skills: List[str]
- min_experience_years: int
- education_level: str
- industry: str
- company_size: str
- remote_work: bool
```

**ResumeData**:
```python
- file_path: str
- raw_text: str
- extracted_skills: List[str]
- education_level: str
- total_experience_years: float  # Parsed from resume dates
```

**MatchScores**:
```python
- skills_match: int (0-100)
- experience_match: int (0-100)
- education_match: int (0-100)
- overall_score: int (0-100)
```

**MatchResult**:
```python
- job: JobCriteria
- resume: ResumeData
- scores: MatchScores
- analysis_notes: Dict[str, Any]
```

## 🔀 Data Flow Diagram

```
┌──────────────┐
│  Job PDF     │
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐      ┌──────────────┐
│ JobDescriptionParser    │─────▶│  JobCriteria │
│   Agent                 │      └──────┬───────┘
└─────────────────────────┘             │
                                        │
┌──────────────┐                       │
│ Resume PDFs  │                       │
└──────┬───────┘                       │
       │                                │
       ▼                                │
┌─────────────────────────┐      ┌─────┴───────┐
│ ResumeScreenerAgent     │      │             │
│   (per resume)          │      │             │
└──────┬──────────────────┘      │             │
       │                          │             │
       ▼                          ▼             ▼
┌──────────────┐      ┌──────────────────────────────┐
│ ResumeData   │─────▶│ MatchmakerAgent              │
└──────────────┘      │   (rule-based matching)      │
                      └──────────┬───────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │   MatchResult[]      │
                      └──────────┬───────────┘
                                 │
                                 │ (for gap analysis)
                                 │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│ StructuredScoring   │  │  Gap Analysis        │  │   Output Files   │
│ Agent (Batch)       │  │  Calculation         │  │   JSON & CSV      │
│  (LLM evaluation)   │  │                      │  │                  │
└──────────┬──────────┘  └──────────┬───────────┘  └─────────┬────────┘
           │                         │                       │
           └─────────────────────────┴───────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │ Final Results │
                            │   (Ranked)    │
                            └───────────────┘
```

## 🎛️ Configuration Flow

```
criteria_requirements.json
        │
        ▼
┌───────────────────────┐
│ load_criteria_from_   │
│   file()              │
└───────────┬───────────┘
            │
            ▼
    ┌───────────────┐
    │ Merge scoring_ │
    │ criteria +      │
    │ additional_     │
    │ criteria        │
    └───────┬─────────┘
            │
            ▼
┌─────────────────────────┐
│ normalize_weights()     │
│   (auto-normalize)      │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ Pass to LLM   │
    │   in prompt    │
    └───────────────┘
```

## 🚨 Error Handling

### Resume Parsing Errors
- **PDF extraction fails**: Skips resume, logs error
- **LLM parsing fails**: Uses default values (empty skills, "bachelors", 0.0 experience)
- **JSON parse error**: Falls back to defaults

### Job Parsing Errors
- **PDF extraction fails**: Raises exception (pipeline stops)
- **LLM parsing fails**: Raises exception (pipeline stops)
- **JSON parse error**: Raises exception (pipeline stops)

### Batch Scoring Errors
- **Batch LLM call fails**: Falls back to individual scoring (`_fallback_individual_scoring`)
- **Response parsing fails**: Falls back to individual scoring
- **Individual scoring fails**: Marks result as `{"success": False, "error": "..."}`

### Output Errors
- **JSON save fails**: Logs warning, continues
- **CSV save fails**: Logs warning, continues
- Results are still displayed in console

## 🔧 Extensibility Points

1. **Add New Criteria**: Edit `criteria_requirements.json` and update prompts
2. **Custom Scoring Logic**: Modify `MatchmakerAgent.score()`
3. **Different LLM**: Change model in `services/llm.py` or pass to agents
4. **Additional Agents**: Create new agent classes following existing patterns
5. **Custom Output Format**: Modify `_flatten_scoring_result()` in `pipeline.py`

## 📊 Performance Considerations

- **Batch Scoring**: Reduces LLM calls from N (one per resume) to 1
- **Parallel PDF Extraction**: Could be parallelized (currently sequential)
- **Caching**: No caching implemented (each run calls LLM)
- **Token Usage**: Batch scoring uses more tokens per call but fewer total calls

## 🔐 Security Notes

- API keys stored in `.env` (not committed)
- PDF paths are user-provided (validate in production)
- LLM responses are parsed and validated (Pydantic models)
- No input sanitization for file paths (assumes trusted environment)

