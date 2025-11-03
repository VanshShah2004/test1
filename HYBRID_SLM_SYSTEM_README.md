# Hybrid LLM + SLM Resume Scoring System

## Overview

This system implements a **hybrid resume scoring approach** that combines:
- **LLM (Gemini)**: For nuanced, context-aware evaluation (65% weight)
- **SLM (NLP-based)**: For deterministic, rule-based scoring (35% weight)

The system automatically falls back to 100% SLM if the LLM fails (network errors, rate limits, etc.).

---

## Architecture

### Components

1. **NLP Extractors** (`services/nlp_extractors.py`)
   - `SkillsExtractor`: Dictionary-based skill extraction
   - `ExperienceExtractor`: Date parsing and experience calculation
   - `EducationClassifier`: Rule-based education level classification
   - `JobDescriptionParser`: Job criteria extraction

2. **SLM Agents** (NLP-based, NO LLMs)
   - `SLMResumeScreenerAgent`: Resume parsing using NLP
   - `SLMJobDescriptionParserAgent`: Job description parsing using NLP
   - `SLMScoringAgent`: Rule-based resume scoring

3. **Hybrid Scoring Agent** (`agents/hybrid_scoring_agent.py`)
   - Combines LLM and SLM results
   - Weighted consensus: 0.65 LLM + 0.35 SLM
   - Automatic fallback to 100% SLM if LLM fails

4. **Validation System** (`slm_training_validation.py`)
   - Validates SLM model against kaggle_dataset
   - Calculates accuracy, precision, recall, MAE, RMSE, R²

---

## Usage

### Basic Example

```python
from agents.hybrid_scoring_agent import HybridScoringAgent

# Initialize agent
hybrid_agent = HybridScoringAgent()

# Define criteria
criteria_requirements = {
    "technical_skills": 60,
    "experience": 40,
    "education": 15,
    "presentation": 10
}

# Score multiple resumes
resume_paths = [
    "documents/resume1.pdf",
    "documents/resume2.pdf",
    "documents/resume3.pdf"
]

results = hybrid_agent.score_resumes_batch(
    resume_paths=resume_paths,
    job_description_path="documents/job_description.pdf",
    criteria_requirements=criteria_requirements
)

# Display results
for result in results:
    if result.get("success"):
        print(hybrid_agent.format_results(result))
```

### Batch Processing Flow

The `score_resumes_batch()` method follows this process:

1. **Step 1: LLM Scoring (Batch)**
   - Scores ALL resumes using Gemini (batch API call)
   - More efficient than individual calls
   - May fail due to network/API issues

2. **Step 2: SLM Scoring (Sequential)**
   - Scores ALL resumes using NLP-based rules
   - Deterministic, always available
   - Fast local processing

3. **Step 3: Weighted Mean Calculation**
   - For each resume: `hybrid_score = 0.65 * LLM_score + 0.35 * SLM_score`
   - If LLM failed: `hybrid_score = 1.0 * SLM_score` (fallback)

---

## Scoring Weights

- **LLM Weight**: 0.65 (65%)
  - Better at nuanced evaluation
  - Context-aware reasoning
  - Can handle complex scenarios

- **SLM Weight**: 0.35 (35%)
  - Deterministic and reliable
  - Fast and cost-free
  - Consistent output

- **Fallback**: 1.0 (100% SLM)
  - Used when LLM unavailable
  - Ensures system always works
  - Prints warning messages

---

## Output Format

Each result contains:

```python
{
    "success": True,
    "scoring_result": {
        "criterion_name": {
            "raw_score": 85.0,          # 0-100 scale
            "weight_given": 60,          # Original weight
            "normalized_percentage": 30.0,  # Normalized %
            "weighted_contribution": 25.5,  # Contribution to total
            "llm_score": 88.0,          # LLM-only score
            "slm_score": 80.0,          # SLM-only score
            "discrepancy": 8.0           # Difference
        },
        "total_score": 82.5,             # Final hybrid score
        "metadata": {
            "scoring_method": "hybrid_llm_slm",  # or "slm_fallback"
            "llm_weight": 0.65,
            "slm_weight": 0.35,
            "llm_total_score": 85.0,
            "slm_total_score": 78.0,
            "hybrid_total_score": 82.5
        }
    }
}
```

---

## Training & Validation

### Running Validation

```python
from slm_training_validation import SLMTrainingValidator

validator = SLMTrainingValidator(
    dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl"
)

# Generate full validation report
report = validator.generate_full_report(sample_size=100)

# Report includes:
# - Parsing accuracy (skills, education)
# - Scoring metrics (MAE, RMSE, R², correlation)
# - Per-criterion metrics
```

### Validation Metrics

- **Parsing Accuracy**: Skills extraction recall, education classification accuracy
- **Scoring Metrics**: 
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score (coefficient of determination)
  - Correlation coefficient

---

## File Structure

```
├── agents/
│   ├── hybrid_scoring_agent.py          # Main hybrid scoring agent
│   ├── slm_resume_screener_agent.py     # SLM resume parser
│   ├── slm_job_description_parser_agent.py  # SLM job parser
│   └── slm_scoring_agent.py             # SLM scoring agent
│
├── services/
│   ├── nlp_extractors.py                # NLP-based extractors
│   ├── skills_dictionary.py              # Skills dictionary
│   └── ...
│
├── slm_training_validation.py            # Validation system
├── example_hybrid_scoring.py             # Usage example
└── HYBRID_SLM_SYSTEM_README.md          # This file
```

---

## Error Handling

### LLM Failures

The system handles LLM failures gracefully:

- **Network Errors**: Falls back to SLM
- **Rate Limits**: Falls back to SLM
- **API Errors**: Falls back to SLM
- **Invalid Responses**: Falls back to SLM

When fallback occurs:
- Warning message is printed
- Result marked as `"scoring_method": "slm_fallback"`
- Error reason stored in metadata

### SLM Failures

If SLM also fails (rare):
- Result marked as `"success": False`
- Error message provided
- System continues with other resumes

---

## Performance

### Speed
- **LLM Batch**: ~2-5 seconds per batch (depends on API)
- **SLM Sequential**: ~0.1-0.5 seconds per resume
- **Total**: Typically <10 seconds for 10 resumes

### Cost
- **LLM**: API costs per token (Gemini pricing)
- **SLM**: Free (local processing)
- **Hybrid**: 65% of full LLM cost (when both available)

### Accuracy
- **LLM**: Better at nuanced evaluation
- **SLM**: Deterministic, consistent
- **Hybrid**: Combines strengths, reduces hallucinations

---

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `langchain-google-genai`: For LLM (Gemini)
- `python-dateutil`: For date parsing
- `scikit-learn`: For validation metrics
- `numpy`: For calculations

---

## Example Output

```
======================================================================
HYBRID SCORING: LLM + SLM Batch Processing
======================================================================

📊 STEP 1: Scoring all resumes with LLM (Gemini)...
✅ LLM scoring completed for 3/3 resumes

📊 STEP 2: Scoring all resumes with SLM (NLP-based)...
  [1/3] SLM scored: resume1.pdf
  [2/3] SLM scored: resume2.pdf
  [3/3] SLM scored: resume3.pdf
✅ SLM scoring completed for 3/3 resumes

📊 STEP 3: Calculating weighted mean (LLM: 0.65, SLM: 0.35)...
  ✅ Hybrid score calculated for resume1.pdf
  ✅ Hybrid score calculated for resume2.pdf
  ✅ Hybrid score calculated for resume3.pdf

✅ Hybrid scoring completed for 3 resumes

📈 Summary:
  • Hybrid (LLM + SLM): 3 resumes
  • Fallback (SLM only): 0 resumes
  • Failed: 0 resumes
```

---

## Notes

- SLM uses **pure NLP** (no language models)
- LLM uses **Gemini API** (requires API key)
- System is **fault-tolerant** (handles failures gracefully)
- Results are **deterministic** (SLM) and **context-aware** (LLM hybrid)

---

**Last Updated**: Hybrid batch scoring implementation with weighted consensus

