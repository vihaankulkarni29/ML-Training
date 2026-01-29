#!/usr/bin/env python
"""
MutationScan CLI: Antimicrobial Resistance Prediction

Command-line interface for querying trained AMR prediction models.

Examples:
    List available antibiotics:
        python main.py --list
    
    Predict resistance:
        python main.py --drug Ciprofloxacin --mut S83L
        python main.py --drug Rifampicin --mut S450L
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference import ResistancePredictor


def format_result_table(result: dict) -> str:
    """Format prediction result as a nice table."""
    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        return f"❌ ERROR: {error_msg}"
    
    lines = [
        "✅ PREDICTION RESULT",
        "=" * 60,
        f"Antibiotic       : {result['antibiotic']}",
        f"Mutation         : {result['wt']}{result['position']}{result['mutant']}",
        f"Resistance Prob  : {result['resistance_prob']:.1%}",
        f"Risk Level       : {result['risk_level']}",
        "=" * 60,
    ]
    
    return "\n".join(lines)


def list_antibiotics(predictor: ResistancePredictor):
    """Print available antibiotics."""
    try:
        antibiotics = predictor.get_available_models()
        
        print("\n" + "=" * 60)
        print("AVAILABLE ANTIBIOTIC MODELS")
        print("=" * 60)
        
        for i, ab in enumerate(antibiotics, 1):
            print(f"{i:2d}. {ab}")
        
        print("=" * 60)
        print(f"Total models: {len(antibiotics)}\n")
        
    except Exception as e:
        print(f"❌ Error listing models: {e}", file=sys.stderr)
        sys.exit(1)


def predict(predictor: ResistancePredictor, drug: str, mutation: str):
    """Run prediction for a single mutation."""
    try:
        result = predictor.predict(mutation=mutation, antibiotic=drug)
        print("\n" + format_result_table(result) + "\n")
        
        if not result.get('success'):
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Prediction error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MutationScan: Antimicrobial Resistance Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List available models:
    python main.py --list
  
  Predict resistance:
    python main.py --drug Ciprofloxacin --mut S83L
    python main.py --drug Rifampicin --mut S450L
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available antibiotic models'
    )
    
    parser.add_argument(
        '--drug',
        type=str,
        help='Antibiotic name (e.g., Ciprofloxacin)'
    )
    
    parser.add_argument(
        '--mut',
        type=str,
        help='Mutation string (e.g., S83L)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models/)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = ResistancePredictor(model_dir=args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # List mode
    if args.list:
        list_antibiotics(predictor)
        return
    
    # Predict mode
    if args.drug and args.mut:
        predict(predictor, drug=args.drug, mutation=args.mut)
        return
    
    # No valid command
    if not args.list and not (args.drug and args.mut):
        parser.print_help()
        print("\n⚠️  Please provide either --list or both --drug and --mut")
        sys.exit(1)


if __name__ == "__main__":
    main()
