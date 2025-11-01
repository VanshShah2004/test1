"""
Training script for SLM (NLP-based) model using Kaggle dataset.
Improves skills dictionary, patterns, and scoring thresholds.
"""

import os
import json
import pandas as pd
import re
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

try:
    import kagglehub
except ImportError:
    print("⚠️  kagglehub not installed. Install with: pip install kagglehub")
    kagglehub = None

from services.skills_dictionary import TECH_SKILLS_DICT, ALL_TECH_SKILLS, SKILL_ALIASES
from services.nlp_extractors import SkillsExtractor, ExperienceExtractor, EducationClassifier


class SLMTrainer:
    """Train and improve SLM components using Kaggle dataset."""
    
    def __init__(self):
        self.dataset_path = None
        self.training_data = None
        
    def load_kaggle_dataset(self, dataset_ref: str = "pranavvenugo/resume-and-job-description") -> str:
        """Load Kaggle dataset and return path."""
        if kagglehub is None:
            raise ImportError("kagglehub not installed. Run: pip install kagglehub")
        
        print("📥 Downloading Kaggle dataset...")
        path = kagglehub.dataset_download(dataset_ref)
        print(f"✅ Dataset downloaded to: {path}")
        self.dataset_path = path
        return path
    
    def load_training_data(self, dataset_path: str) -> Dict[str, Any]:
        """Load and parse training data from dataset."""
        print("📂 Loading training data...")
        
        training_data = {
            'resumes': [],
            'job_descriptions': [],
            'labels': [],  # If available in dataset
            'dataframes': []  # Store multiple dataframes
        }
        
        # Check what files are in the dataset
        if os.path.isdir(dataset_path):
            files = os.listdir(dataset_path)
            print(f"📄 Found files: {files}")
            
            # Try to load common file formats
            for file in files:
                file_path = os.path.join(dataset_path, file)
                
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"✅ Loaded CSV: {file} ({len(df)} rows)")
                    print(f"   Columns: {list(df.columns)}")
                    
                    # Store dataframe with filename key
                    training_data['dataframes'].append(df)
                    training_data[f'df_{file.replace(".csv", "")}'] = df
                    
                    # Identify resume vs job description files
                    if 'resume' in file.lower() or 'Resume' in file:
                        training_data['resume_df'] = df
                    if 'job' in file.lower() or 'training' in file.lower():
                        training_data['job_df'] = df
                    
                elif file.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ Loaded JSON: {file}")
                    training_data['json_data'] = data
                    
                elif file.endswith('.txt') or file.endswith('.md'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"✅ Loaded text file: {file}")
                    
        elif os.path.isfile(dataset_path) and dataset_path.endswith('.zip'):
            print("📦 Dataset is a zip file. Please extract it first.")
            print(f"   Path: {dataset_path}")
        
        self.training_data = training_data
        return training_data
    
    def extract_skills_from_dataset(self, training_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract skills from training data to expand dictionary."""
        print("🔍 Extracting skills from training data...")
        
        all_text = ""
        skill_mentions = Counter()
        new_skills = defaultdict(list)
        
        # Extract text from various sources
        # Check resume dataframe
        if 'resume_df' in training_data:
            df = training_data['resume_df']
            # Try common column names for resume text
            text_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['text', 'content', 'resume', 'str', 'html']
            )]
            
            for col in text_columns:
                all_text += " " + " ".join(df[col].fillna("").astype(str))
        
        # Check job description dataframe
        if 'job_df' in training_data:
            df = training_data['job_df']
            # Try common column names for job descriptions
            text_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['description', 'text', 'content', 'job']
            )]
            
            for col in text_columns:
                all_text += " " + " ".join(df[col].fillna("").astype(str))
        
        # Also check all dataframes
        for df in training_data.get('dataframes', []):
            text_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['text', 'content', 'resume', 'description', 'skill']
            )]
            
            for col in text_columns:
                all_text += " " + " ".join(df[col].fillna("").astype(str))
        
        if 'json_data' in training_data:
            # Flatten JSON and extract text
            data = training_data['json_data']
            if isinstance(data, list):
                for item in data:
                    all_text += " " + json.dumps(item)
            elif isinstance(data, dict):
                all_text += " " + json.dumps(data)
        
        # Use existing extractor to find skills
        extractor = SkillsExtractor()
        found_skills = extractor.extract(all_text)
        
        # Count skill mentions
        for skill in found_skills:
            skill_mentions[skill] += 1
        
        # Find potentially new skills (not in current dictionary)
        current_skills_lower = {s.lower() for s in ALL_TECH_SKILLS}
        
        # Look for technical terms using patterns
        tech_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized tech terms
            r'\b[a-z]+(?:\.js|\.py|\.net|\.ts|\.jsx|\.tsx)\b',  # File extensions
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower()
                if match_lower not in current_skills_lower:
                    # Check if it looks like a tech skill
                    if len(match) > 2 and len(match) < 30:
                        new_skills['potential'].append(match_lower)
        
        print(f"✅ Found {len(found_skills)} unique skills")
        print(f"   Top 20: {dict(skill_mentions.most_common(20))}")
        print(f"   Potential new skills: {len(set(new_skills['potential']))}")
        
        return {
            'extracted_skills': found_skills,
            'skill_counts': dict(skill_mentions),
            'new_skills': list(set(new_skills['potential']))[:100],  # Top 100
            'all_text_length': len(all_text)
        }
    
    def train_education_classifier(self, training_data: Dict[str, Any]) -> Any:
        """Train an ML-based education classifier."""
        print("🎓 Training education classifier...")
        
        # Create training examples from dataset
        X = []  # Text features
        y = []  # Labels (education levels)
        
        classifier = EducationClassifier()
        
        # Extract education mentions from dataset
        # Use resume dataframe
        if 'resume_df' in training_data:
            df = training_data['resume_df']
            
            # Try to find text columns
            text_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['text', 'content', 'resume', 'str', 'html']
            )]
            
            for col in text_columns:
                for text in df[col].fillna("").astype(str):
                    if len(text) > 50:  # Meaningful text
                        # Classify using current rule-based classifier to get labels
                        education_level = classifier.classify(text)
                        X.append(text[:500])  # Limit length
                        y.append(education_level)
        
        # Also check all dataframes
        for df in training_data.get('dataframes', []):
            text_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['text', 'content', 'resume', 'str']
            )]
            
            for col in text_columns:
                for text in df[col].fillna("").astype(str):
                    if len(text) > 50:
                        education_level = classifier.classify(text)
                        X.append(text[:500])
                        y.append(education_level)
        
        if len(X) < 10:
            print("⚠️  Not enough training data. Using rule-based classifier only.")
            return None
        
        # Train TF-IDF + Logistic Regression
        print(f"   Training on {len(X)} examples...")
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_vectorized = vectorizer.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Education classifier trained")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Classification report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy
        }
    
    def train_scoring_model(self, training_data: Dict[str, Any]) -> Any:
        """Train scoring thresholds based on labeled data (if available)."""
        print("📊 Training scoring model...")
        
        # This would require labeled data (resume + job + match score)
        # For now, we'll use the dataset to validate and tune thresholds
        
        total_rows = 0
        score_columns = []
        
        # Check all dataframes
        for df in training_data.get('dataframes', []):
            total_rows += len(df)
            
            # Try to find score/match columns
            found_score_cols = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['score', 'match', 'rating', 'label', 'category']
            )]
            score_columns.extend(found_score_cols)
        
        if total_rows > 0:
            print(f"   Analyzing {total_rows} examples...")
            
            if score_columns:
                print(f"   Found potential label columns: {set(score_columns)}")
                print("   Note: Could train scoring model if labels represent match scores")
            else:
                print("   No labeled scores found. Using rule-based scoring only.")
        
        return None
    
    def update_skills_dictionary(self, extracted_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Update skills dictionary with new skills from dataset."""
        print("📝 Updating skills dictionary...")
        
        updated_dict = TECH_SKILLS_DICT.copy()
        
        # Get new skills to add
        new_skills_list = extracted_data.get('new_skills', [])
        skill_counts = extracted_data.get('skill_counts', {})
        
        # Filter new skills (must appear multiple times to be reliable)
        reliable_new_skills = [
            skill for skill, count in skill_counts.items()
            if count >= 3 and skill not in ALL_TECH_SKILLS
        ]
        
        # Add to appropriate category (or "others" if unclear)
        added_count = 0
        for skill in reliable_new_skills[:50]:  # Top 50 most frequent
            # Try to categorize
            category = self._categorize_skill(skill)
            if category not in updated_dict:
                updated_dict[category] = []
            
            if skill not in updated_dict[category]:
                updated_dict[category].append(skill)
                added_count += 1
        
        print(f"✅ Added {added_count} new skills to dictionary")
        
        return updated_dict
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill into appropriate category."""
        skill_lower = skill.lower()
        
        # Simple keyword-based categorization
        if any(kw in skill_lower for kw in ['db', 'database', 'sql', 'nosql']):
            return 'databases'
        elif any(kw in skill_lower for kw in ['cloud', 'aws', 'azure', 'gcp']):
            return 'cloud_platforms'
        elif any(kw in skill_lower for kw in ['test', 'qa', 'selenium', 'jest']):
            return 'testing'
        elif any(kw in skill_lower for kw in ['docker', 'kubernetes', 'ci/cd', 'jenkins']):
            return 'devops_tools'
        elif any(kw in skill_lower for kw in ['pandas', 'numpy', 'tensorflow', 'pytorch', 'ml', 'ai']):
            return 'data_science'
        elif skill_lower.endswith(('.js', '.py', '.ts', '.java', '.cpp', '.go')):
            return 'programming_languages'
        elif any(kw in skill_lower for kw in ['react', 'angular', 'vue', 'framework']):
            return 'frameworks'
        else:
            return 'others'
    
    def save_trained_models(self, models: Dict[str, Any], output_dir: str = "trained_models"):
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving trained models to {output_dir}...")
        
        for name, model in models.items():
            if model is not None:
                path = os.path.join(output_dir, f"{name}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   ✅ Saved: {name}")
        
        print(f"✅ Models saved to {output_dir}/")
    
    def generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate training report."""
        report = []
        report.append("=" * 70)
        report.append("SLM MODEL TRAINING REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append("📊 Training Statistics:")
        report.append(f"   - Skills extracted: {results.get('skills_extracted', 0)}")
        report.append(f"   - New skills found: {results.get('new_skills_count', 0)}")
        report.append(f"   - Education classifier accuracy: {results.get('education_accuracy', 'N/A')}")
        report.append("")
        
        report.append("✅ Improvements Made:")
        if results.get('skills_dictionary_updated'):
            report.append("   - Skills dictionary expanded with new technologies")
        if results.get('education_model_trained'):
            report.append("   - Education classifier trained on dataset")
        if results.get('scoring_thresholds_tuned'):
            report.append("   - Scoring thresholds tuned based on data")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Main training function."""
    print("🚀 Starting SLM Model Training")
    print("=" * 70)
    print()
    
    trainer = SLMTrainer()
    
    try:
        # Step 1: Load Kaggle dataset
        dataset_path = trainer.load_kaggle_dataset()
        
        # Step 2: Load training data
        training_data = trainer.load_training_data(dataset_path)
        
        # Check if we have any data loaded
        has_data = (
            len(training_data.get('dataframes', [])) > 0 or 
            training_data.get('json_data') is not None or
            training_data.get('resume_df') is not None or
            training_data.get('job_df') is not None
        )
        
        if not has_data:
            print("⚠️  Could not load structured data from dataset.")
            print("   Attempting to use dataset path directly...")
            return
        
        # Step 3: Extract skills and expand dictionary
        extracted_data = trainer.extract_skills_from_dataset(training_data)
        
        # Step 4: Update skills dictionary
        updated_dict = trainer.update_skills_dictionary(extracted_data)
        
        # Save updated dictionary
        output_file = "services/skills_dictionary_trained.py"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('Trained skills dictionary (enhanced with Kaggle dataset)\n')
            f.write('"""\n\n')
            f.write('TECH_SKILLS_DICT = ')
            f.write(json.dumps(updated_dict, indent=2))
            f.write('\n\n')
            f.write('# Flatten all skills\n')
            f.write('ALL_TECH_SKILLS = set()\n')
            f.write('for category, skills in TECH_SKILLS_DICT.items():\n')
            f.write('    ALL_TECH_SKILLS.update(skills)\n')
        
        print(f"✅ Saved updated dictionary to {output_file}")
        
        # Step 5: Train education classifier
        education_model = trainer.train_education_classifier(training_data)
        
        # Step 6: Train scoring model (if labeled data available)
        scoring_model = trainer.train_scoring_model(training_data)
        
        # Step 7: Save trained models
        models_to_save = {}
        if education_model:
            models_to_save['education_classifier'] = education_model
        if scoring_model:
            models_to_save['scoring_model'] = scoring_model
        
        if models_to_save:
            trainer.save_trained_models(models_to_save)
        
        # Step 8: Generate report
        results = {
            'skills_extracted': len(extracted_data.get('extracted_skills', [])),
            'new_skills_count': len(extracted_data.get('new_skills', [])),
            'education_accuracy': f"{education_model['accuracy']:.2%}" if education_model else 'N/A',
            'skills_dictionary_updated': True,
            'education_model_trained': education_model is not None,
            'scoring_thresholds_tuned': scoring_model is not None
        }
        
        report = trainer.generate_training_report(results)
        print(report)
        
        # Save report
        with open('training_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ Training complete!")
        print(f"📄 Report saved to: training_report.txt")
        print(f"📦 Trained models saved to: trained_models/")
        print(f"📝 Updated dictionary: {output_file}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

