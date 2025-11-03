"""
Quick script to train SLM model and evaluate it.
Run this to train the model on kaggle_dataset and get accuracy/precision metrics.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from simplified version first (avoids TensorFlow issues)
try:
    # Set environment variables before importing
    import os
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    from slm_training.train_slm_simple import main as train_main
    USE_SIMPLE = True
except ImportError:
    from slm_training.train_slm_model import train_slm_model
    from slm_training.evaluate_trained_model import evaluate_trained_model, display_evaluation_results
    USE_SIMPLE = False

def main():
    """Train and evaluate SLM model."""
    print("="*70)
    print("SLM TRAINING PIPELINE")
    print("="*70)
    
    if USE_SIMPLE:
        # Use simplified training (avoids TensorFlow issues)
        print("\n📝 Using simplified training script (PyTorch only)")
        train_main()
        
        # Then evaluate
        print("\n" + "="*70)
        print("STEP 2: EVALUATING TRAINED MODEL")
        print("="*70)
        
        try:
            from slm_training.evaluate_trained_model import evaluate_trained_model, display_evaluation_results
            evaluation_results = evaluate_trained_model(
                model_path="models/trained_slm",
                dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
                test_size=200
            )
            
            if evaluation_results:
                display_evaluation_results(evaluation_results)
                
                print("\n" + "="*70)
                print("FINAL TRAINING SUMMARY")
                print("="*70)
                print(f"\n✅ Model trained and saved to: models/trained_slm/")
                print(f"\n📊 Accuracy & Precision Results:")
                print(f"  • Accuracy (±10 points): {evaluation_results['overall_metrics']['accuracy']:.1%}")
                print(f"  • Precision (High Scores ≥80): {evaluation_results['classification_metrics']['precision_high_scores']:.1%}")
                print(f"  • Recall (High Scores ≥80): {evaluation_results['classification_metrics']['recall_high_scores']:.1%}")
                print(f"  • F1 Score: {evaluation_results['classification_metrics']['f1_high_scores']:.3f}")
                print(f"  • RMSE: {evaluation_results['overall_metrics']['rmse']:.2f} points")
                print(f"  • MAE: {evaluation_results['overall_metrics']['mae']:.2f} points")
                print(f"  • R² Score: {evaluation_results['overall_metrics']['r2_score']:.3f}")
                print(f"\n💡 The trained model will now be used by HybridScoringAgent")
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
            print("   Training completed, but evaluation could not run")
    else:
        # Use original training script
        print("\n" + "="*70)
        print("STEP 1: TRAINING SLM MODEL")
        print("="*70)
        
        training_results = train_slm_model(
            model_name="distilbert-base-uncased",
            dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
            output_dir="models/trained_slm",
            num_epochs=5,
            batch_size=8,
            learning_rate=2e-5
        )
        
        if training_results is None:
            print("\n❌ Training failed. Cannot proceed to evaluation.")
            return
        
        # Step 2: Evaluate model
        print("\n" + "="*70)
        print("STEP 2: EVALUATING TRAINED MODEL")
        print("="*70)
        
        evaluation_results = evaluate_trained_model(
            model_path="models/trained_slm",
            dataset_path="kaggle_dataset/hf_clean_data/resume_score_details.jsonl",
            test_size=200
        )
        
        if evaluation_results:
            display_evaluation_results(evaluation_results)
            
            print("\n" + "="*70)
            print("FINAL TRAINING SUMMARY")
            print("="*70)
            print(f"\n✅ Model trained and saved to: models/trained_slm/")
            print(f"\n📊 Accuracy & Precision Results:")
            print(f"  • Accuracy (±10 points): {evaluation_results['overall_metrics']['accuracy']:.1%}")
            print(f"  • Precision (High Scores ≥80): {evaluation_results['classification_metrics']['precision_high_scores']:.1%}")
            print(f"  • Recall (High Scores ≥80): {evaluation_results['classification_metrics']['recall_high_scores']:.1%}")
            print(f"  • F1 Score: {evaluation_results['classification_metrics']['f1_high_scores']:.3f}")
            print(f"  • RMSE: {evaluation_results['overall_metrics']['rmse']:.2f} points")
            print(f"  • MAE: {evaluation_results['overall_metrics']['mae']:.2f} points")
            print(f"  • R² Score: {evaluation_results['overall_metrics']['r2_score']:.3f}")
            print(f"\n💡 The trained model will now be used by HybridScoringAgent")
        else:
            print("\n❌ Evaluation failed")


if __name__ == "__main__":
    main()

