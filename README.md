# LLM-Powered Resume Screening System

An intelligent resume screening system that uses Large Language Models (LLMs) to evaluate candidates against job descriptions with structured, weighted scoring criteria.

## 🚀 Features

- **AI-Powered Resume Parsing**: Extracts skills, education, and experience years from resumes using LLM
- **Job Description Analysis**: Parses job requirements including required skills, experience, and education
- **Structured Scoring**: Evaluates candidates across multiple weighted criteria (technical skills, experience, education, etc.)
- **Batch Comparative Evaluation**: Scores all resumes together for relative comparison and consistency
- **Gap Analysis**: Identifies missing skills, experience gaps, and education mismatches
- **Interactive Dashboard**: Streamlit-based visualization for results analysis
- **Export Options**: Saves results as JSON and CSV for further analysis

## 📋 Requirements

- Python 3.8+
- Google Gemini API Key (set as `GEMINI_API_KEY` environment variable)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd test1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 📁 Project Structure

```
test1/
├── agents/                    # Agent modules for parsing and matching
│   ├── job_description_parser_agent.py
│   ├── resume_screener_agent.py
│   └── matchmaker_agent.py
├── common/                    # Shared data models
│   └── models.py
├── services/                   # Core services (LLM, PDF parsing)
│   ├── llm.py
│   └── pdf_utils.py
├── documents/                  # Input PDFs (job descriptions and resumes)
├── outputs/                    # Generated results (JSON, CSV)
├── pipeline.py                # Main pipeline entry point
├── main_orchestrator.py       # Orchestrates agent workflow
├── structured_scoring_agent.py # LLM-based structured scoring
├── dashboard.py                # Streamlit dashboard
├── criteria_requirements.json  # Scoring criteria configuration
└── requirements.txt
```

## 🎯 Usage

### Basic Usage

Run the pipeline with a job description and multiple resumes:

```bash
python pipeline.py
```

The default configuration processes:
- Job Description: `documents/Bottomline_ Intern + FTE - JD 2026 Batch.pdf`
- Resumes: `documents/final5resume.pdf`, `documents/Nirmit_Jain_Resume_Final.pdf`, `documents/final4resume.pdf`

### Custom Usage

Modify `pipeline.py` or import it in your own script:

```python
from pipeline import run_pipeline

results = run_pipeline(
    job_pdf="path/to/job_description.pdf",
    resume_pdfs=[
        "path/to/resume1.pdf",
        "path/to/resume2.pdf",
        "path/to/resume3.pdf"
    ],
    criteria_requirements={
        "technical_skills": 40,
        "experience": 30,
        "education": 20,
        "presentation": 10
    }
)
```

### Interactive Dashboard

Launch the Streamlit dashboard to visualize results:

```bash
streamlit run dashboard.py
```

The dashboard reads results from `outputs/scoring_results.json` and provides:
- Per-criterion score comparisons
- Total score distribution
- Gap analysis visualizations
- Interactive filtering and selection

## ⚙️ Configuration

### Scoring Criteria

Edit `criteria_requirements.json` to customize scoring weights:

```json
{
  "scoring_criteria": {
    "technical_skills": {
      "weight": 60,
      "description": "Programming languages, frameworks, tools, and technical competencies"
    },
    "experience": {
      "weight": 60,
      "description": "Relevant work experience, internships, and project experience"
    },
    "education": {
      "weight": 15,
      "description": "Educational background, degrees, and academic achievements"
    },
    "presentation": {
      "weight": 10,
      "description": "Resume formatting, clarity, organization, and professionalism"
    }
  }
}
```

Weights are automatically normalized, so they don't need to sum to 100.

## 📊 Output Format

The pipeline generates several output files:

1. **`outputs/scoring_results.json`**: Complete structured results with metadata
2. **`outputs/scoring_results_YYYYMMDD_HHMMSS.json`**: Timestamped snapshot
3. **`outputs/scoring_results_flat.csv`**: Flattened CSV for Excel/analysis

Each result includes:
- **Per-criterion scores**: Raw scores (0-100) and weighted contributions
- **Total score**: Sum of weighted contributions
- **Gap analysis**: Missing skills, experience gaps, education mismatches
- **Metadata**: File paths, criteria weights, normalization info

## 🔍 How It Works

1. **Job Parsing**: Extracts job requirements (skills, experience, education) from PDF
2. **Resume Parsing**: Extracts candidate information (skills, education, experience years) from PDFs
3. **Quick Matching**: Performs initial skill-based matching using rule-based logic
4. **Structured Scoring**: Uses LLM for comprehensive evaluation across weighted criteria
5. **Gap Analysis**: Compares job requirements vs candidate qualifications
6. **Ranking**: Orders candidates by total weighted score

For detailed flow documentation, see [FLOW.md](FLOW.md).

## 🤖 LLM Integration

The system uses Google's Gemini API (`gemini-2.0-flash-exp` by default) for:
- Resume text parsing and extraction
- Job description analysis
- Structured scoring evaluation

You can change the model in `services/llm.py` or when initializing `LLMService`.

## 📝 Notes

- **Experience Calculation**: Full-time experience counts as 1x, internships count as 0.5x
- **Batch Scoring**: All resumes are scored together in one LLM call for relative comparison
- **Error Handling**: The system gracefully handles parsing errors and provides fallbacks
- **Windows Support**: Console encoding is automatically fixed for Windows systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypdf2.readthedocs.io/)

