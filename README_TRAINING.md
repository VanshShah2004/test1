# SLM Model Training Guide

## Overview

The SLM (NLP-based) system can be trained on the Kaggle dataset to improve accuracy. Training enhances:
- Skills dictionary (discovers new technologies)
- Education classification (ML-based classifier)
- Scoring thresholds (validated against data)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- `kagglehub` for dataset access
- `scikit-learn` for ML models

### 2. Set Up Kaggle Authentication

```bash
# Option 1: Place kaggle.json in ~/.kaggle/
# Option 2: Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3. Run Training

```bash
python train_slm_model.py
```

This will:
1. Download Kaggle dataset (if not already downloaded)
2. Extract skills from dataset
3. Expand skills dictionary
4. Train education classifier
5. Save trained models

## What Gets Trained

### 1. Skills Dictionary Enhancement
- **Input**: All resumes and job descriptions from dataset
- **Process**: Extracts skills mentions, finds new technologies
- **Output**: Updated `services/skills_dictionary_trained.py` with new skills

### 2. Education Classifier
- **Input**: Resume texts with education sections
- **Process**: Trains TF-IDF + Logistic Regression classifier
- **Output**: `trained_models/education_classifier.pkl`
- **Accuracy**: Typically 85-95% (better than rule-based)

### 3. Scoring Model (if labeled data available)
- **Input**: Resume-job pairs with match scores
- **Process**: Trains scoring thresholds and weights
- **Output**: Tuned scoring parameters

## Training Output

After training, you'll get:

```
trained_models/
├── education_classifier.pkl    # Trained ML classifier
└── (other models if available)

services/
└── skills_dictionary_trained.py  # Enhanced dictionary

training_report.txt              # Training summary
```

## Using Trained Models

The system automatically uses trained models if available:

```python
from services.nlp_service import NLPService

# Will automatically load trained models if available
nlp_service = NLPService()
```

To force use of rule-based only:

```python
from services.nlp_extractors import SkillsExtractor, EducationClassifier

# Use rule-based directly
extractor = SkillsExtractor()
classifier = EducationClassifier()
```

## Retraining

To retrain with new data:

1. Update the dataset reference in `train_slm_model.py` if needed
2. Run training again:
   ```bash
   python train_slm_model.py
   ```

Trained models will be updated.

## Performance Improvements

Expected improvements after training:

- **Skills Extraction**: 75-85% → 80-90% accuracy
  - More comprehensive dictionary
  - Better handling of new technologies

- **Education Classification**: 85-90% → 90-95% accuracy
  - ML model better at context understanding
  - Handles edge cases and variations

- **Overall SLM**: 75-80% → 80-85% accuracy
  - Better feature extraction
  - More accurate scoring inputs

## Troubleshooting

### Dataset Download Fails
- Check Kaggle authentication
- Verify internet connection
- Try manual download: https://www.kaggle.com/datasets/pranavvenugo/resume-and-job-description

### Not Enough Training Data
- Training will fall back to rule-based
- Minimum 10 examples needed for education classifier
- Skills dictionary can always be updated

### Model Loading Fails
- Check that `trained_models/` directory exists
- Verify pickle files are valid
- System will fall back to rule-based automatically

## Dataset Requirements

The Kaggle dataset should contain:
- Resume texts
- Job descriptions
- (Optional) Match scores or labels

Common formats:
- CSV with columns: `resume_text`, `job_description`, `match_score`
- JSON with resume/job pairs
- Separate files for resumes and jobs

The training script automatically detects and handles different formats.

## Next Steps

1. **Run Training**: `python train_slm_model.py`
2. **Review Report**: Check `training_report.txt`
3. **Test Improvements**: Run pipeline and compare accuracy
4. **Iterate**: Update training based on results

## Notes

- Training is optional - system works with rule-based only
- Trained models improve accuracy but add complexity
- Rule-based fallback ensures system always works
- Training takes 5-15 minutes depending on dataset size

