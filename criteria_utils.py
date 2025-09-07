#!/usr/bin/env python3
"""
Utility functions for criteria requirements processing
"""

import json
from typing import Dict, Any

def calculate_total_weight(criteria: Dict[str, Any]) -> int:
    """Calculate total weight from criteria dictionary"""
    scoring_criteria = criteria.get("scoring_criteria", {})
    additional_criteria = criteria.get("additional_criteria", {})
    all_criteria = {**scoring_criteria, **additional_criteria}
    
    return sum(info.get("weight", 10) for info in all_criteria.values())

def add_calculated_metadata(criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Add calculated metadata to criteria (total_weight, normalized_weights, etc.)"""
    # Create a copy to avoid modifying original
    enhanced_criteria = criteria.copy()
    
    # Calculate total weight
    total_weight = calculate_total_weight(criteria)
    
    # Add calculated metadata
    enhanced_criteria["calculated_metadata"] = {
        "total_weight": total_weight,
        "normalized_weights": {},
        "weight_summary": {}
    }
    
    # Calculate normalized weights
    scoring_criteria = criteria.get("scoring_criteria", {})
    additional_criteria = criteria.get("additional_criteria", {})
    all_criteria = {**scoring_criteria, **additional_criteria}
    
    for key, info in all_criteria.items():
        weight = info.get("weight", 10)
        normalized = weight / total_weight if total_weight > 0 else 0
        enhanced_criteria["calculated_metadata"]["normalized_weights"][key] = normalized
        enhanced_criteria["calculated_metadata"]["weight_summary"][key] = {
            "original_weight": weight,
            "normalized_weight": normalized,
            "percentage_of_total": (weight / total_weight) * 100 if total_weight > 0 else 0
        }
    
    return enhanced_criteria

def load_and_enhance_criteria(criteria_file: str = None) -> Dict[str, Any]:
    """Load criteria from file and add calculated metadata"""
    if criteria_file is None:
        criteria_file = "criteria_requirements.json"
    
    try:
        with open(criteria_file, 'r') as f:
            criteria = json.load(f)
        
        # Add calculated metadata
        enhanced_criteria = add_calculated_metadata(criteria)
        
        return enhanced_criteria
        
    except FileNotFoundError:
        print(f"❌ Criteria file not found: {criteria_file}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {criteria_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading criteria: {e}")
        return None

def print_criteria_summary(criteria: Dict[str, Any]):
    """Print a summary of the criteria with calculated values"""
    if not criteria:
        print("❌ No criteria provided")
        return
    
    metadata = criteria.get("calculated_metadata", {})
    total_weight = metadata.get("total_weight", 0)
    
    print(f"📊 CRITERIA SUMMARY")
    print("=" * 50)
    print(f"Total Weight: {total_weight}")
    print()
    
    scoring_criteria = criteria.get("scoring_criteria", {})
    additional_criteria = criteria.get("additional_criteria", {})
    all_criteria = {**scoring_criteria, **additional_criteria}
    
    print("Individual Weights:")
    for key, info in all_criteria.items():
        weight = info.get("weight", 10)
        description = info.get("description", key.replace("_", " ").title())
        percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
        print(f"  • {description}: {weight} ({percentage:.1f}% of total)")
    
    print()
    if total_weight == 100:
        print("✅ Weights sum to exactly 100%")
    elif total_weight < 100:
        print(f"ℹ️  Weights sum to {total_weight}% (will be normalized up)")
    else:
        print(f"ℹ️  Weights sum to {total_weight}% (will be normalized down)")

if __name__ == "__main__":
    # Test the utility functions
    print("CRITERIA UTILS TEST")
    print("=" * 50)
    
    # Load and enhance criteria
    enhanced_criteria = load_and_enhance_criteria()
    
    if enhanced_criteria:
        print_criteria_summary(enhanced_criteria)
        
        print("\n" + "=" * 50)
        print("ENHANCED CRITERIA STRUCTURE:")
        print("=" * 50)
        print(json.dumps(enhanced_criteria["calculated_metadata"], indent=2))
