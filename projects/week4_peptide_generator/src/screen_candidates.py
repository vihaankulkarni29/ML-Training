"""
Screening Pipeline for AI-Generated Peptide Candidates.
Loads generated sequences, computes features, predicts potency (MIC),
and filters for high-efficacy candidates suitable for experimental validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ============================================================================
# Feature Engineering Functions
# ============================================================================

def get_properties(seq: str) -> Dict[str, float]:
    """
    Compute physicochemical properties for a peptide sequence.

    Args:
        seq: Amino acid sequence string

    Returns:
        Dictionary of physicochemical properties
    """
    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": 0,
            "positive_charge": 0,
        }

    seq = seq.strip()
    try:
        analysis = ProteinAnalysis(seq)
        return {
            "mol_weight": float(analysis.molecular_weight()),
            "aromaticity": float(analysis.aromaticity()),
            "instability_index": float(analysis.instability_index()),
            "isoelectric_point": float(analysis.isoelectric_point()),
            "gravy": float(analysis.gravy()),
            "length": len(seq),
            "positive_charge": seq.count("K") + seq.count("R"),
        }
    except Exception as e:
        print(f"⚠️  Error analyzing sequence '{seq[:20]}...': {e}")
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": len(seq),
            "positive_charge": 0,
        }


def load_vectorizer(vectorizer_path: str) -> Optional:
    """
    Load k-mer vectorizer from disk.

    Args:
        vectorizer_path: Path to vectorizer pickle file

    Returns:
        Loaded vectorizer or None if not found
    """
    if not vectorizer_path:
        return None

    vec_path = Path(vectorizer_path)
    if not vec_path.exists():
        print(f"⚠️  Vectorizer not found at {vectorizer_path}")
        return None

    try:
        vectorizer = joblib.load(vectorizer_path)
        print(f"✓ Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        print(f"✗ Error loading vectorizer: {e}")
        return None


def extract_features(
    sequence: str,
    vectorizer,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    """
    Extract both physicochemical and k-mer features for a sequence.

    Args:
        sequence: Amino acid sequence
        vectorizer: Fitted CountVectorizer for k-mers

    Returns:
        Tuple of (physicochemical properties dict, k-mer features dataframe)
    """
    # Physicochemical properties
    props = get_properties(sequence)

    # K-mer features
    kmer_df = None
    if vectorizer is not None:
        try:
            kmer_matrix = vectorizer.transform([sequence])
            kmer_df = pd.DataFrame(
                kmer_matrix.toarray(),
                columns=[f"kmer_{kmer}" for kmer in vectorizer.get_feature_names_out()],
            )
        except Exception as e:
            print(f"⚠️  Error extracting k-mers for '{sequence[:20]}...': {e}")

    return props, kmer_df


# ============================================================================
# Model & Prediction Functions
# ============================================================================

def load_model(model_path: str) -> Optional:
    """
    Load trained MIC prediction model.

    Args:
        model_path: Path to model pickle file

    Returns:
        Loaded model or None if not found
    """
    path = Path(model_path)

    if not path.exists():
        print(f"✗ Model not found at {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def predict_mic(
    sequence: str,
    model,
    vectorizer,
) -> Optional[float]:
    """
    Predict MIC (µM) for a peptide sequence.

    Args:
        sequence: Amino acid sequence
        model: Trained RandomForestRegressor
        vectorizer: Fitted CountVectorizer for k-mers

    Returns:
        Predicted MIC in µM, or None if prediction fails
    """
    if model is None:
        return None

    try:
        # Extract features
        props, kmer_df = extract_features(sequence, vectorizer)

        # Create feature dataframe
        props_df = pd.DataFrame([props])

        # Combine features
        if kmer_df is not None:
            full_features = pd.concat([props_df, kmer_df], axis=1)
        else:
            full_features = props_df

        # Align with model's expected features
        if hasattr(model, "feature_names_in_"):
            full_features = full_features.reindex(
                columns=model.feature_names_in_, fill_value=0
            )

        # Predict neg_log_mic
        pred_neg_log_mic = float(model.predict(full_features)[0])

        # Convert to MIC (µM)
        mic_uM = 10 ** (-pred_neg_log_mic)

        return mic_uM

    except Exception as e:
        print(f"⚠️  Prediction error for '{sequence[:20]}...': {e}")
        return None


# ============================================================================
# Screening & Filtering Functions
# ============================================================================

def categorize_potency(mic_uM: float) -> str:
    """
    Categorize peptide potency based on MIC value.

    Args:
        mic_uM: MIC in µM

    Returns:
        Potency category string
    """
    if mic_uM < 2:
        return "💎 Excellent (High Potency)"
    elif mic_uM < 5:
        return "✅ Good (Moderate Potency)"
    elif mic_uM < 10:
        return "⚠️  Moderate (Fair Potency)"
    elif mic_uM < 50:
        return "❌ Weak (Low Potency)"
    else:
        return "❌ Inactive (Very Low Potency)"


def screen_candidates(
    candidates_csv: str,
    model_path: str,
    vectorizer_path: str = "",
    mic_threshold: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Screen generated peptide candidates for potency.

    Args:
        candidates_csv: Path to generated_peptides.csv
        model_path: Path to MIC predictor model
        vectorizer_path: Path to k-mer vectorizer
        mic_threshold: MIC cutoff for "potent" (µM)

    Returns:
        Tuple of (all results, potent candidates)
    """
    print("\n" + "=" * 70)
    print("PEPTIDE CANDIDATE SCREENING PIPELINE")
    print("=" * 70)

    # 1. Load candidates
    print("\n1. Loading generated candidates...")
    candidates_path = Path(candidates_csv)
    if not candidates_path.exists():
        print(f"✗ Candidates file not found: {candidates_csv}")
        return pd.DataFrame(), pd.DataFrame()

    df_candidates = pd.read_csv(candidates_csv)
    print(f"✓ Loaded {len(df_candidates)} generated sequences")

    # 2. Load model and vectorizer
    print("\n2. Loading predictive models...")
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    if model is None:
        print("✗ Cannot proceed without MIC predictor model")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Predict MIC for each candidate
    print("\n3. Predicting MIC for candidates...")
    predictions = []

    for idx, row in df_candidates.iterrows():
        sequence = row["sequence"]
        mic_uM = predict_mic(sequence, model, vectorizer)

        if mic_uM is not None:
            predictions.append(
                {
                    "sequence": sequence,
                    "length": len(sequence),
                    "predicted_mic_uM": mic_uM,
                    "predicted_neg_log_mic": -np.log10(mic_uM),
                    "potency_category": categorize_potency(mic_uM),
                }
            )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_candidates):
            print(f"   Processed {idx + 1}/{len(df_candidates)} sequences")

    df_results = pd.DataFrame(predictions)

    if df_results.empty:
        print("✗ No successful predictions")
        return df_results, pd.DataFrame()

    # 4. Filter for potent candidates
    print(f"\n4. Filtering for potent candidates (MIC < {mic_threshold} µM)...")
    df_potent = df_results[df_results["predicted_mic_uM"] < mic_threshold].copy()
    df_potent = df_potent.sort_values("predicted_mic_uM")

    print(f"✓ Found {len(df_potent)} potent candidates out of {len(df_results)} generated")
    print(f"   Potency rate: {len(df_potent) / len(df_results) * 100:.1f}%")

    # 5. Statistics
    print("\n5. Potency Statistics:")
    print("-" * 70)
    print(f"   Total screened:        {len(df_results)}")
    print(f"   Potent (MIC < 5 µM):   {len(df_potent)}")
    print(f"   Good (MIC < 10 µM):    {len(df_results[df_results['predicted_mic_uM'] < 10])}")
    print(f"   Weak (MIC < 50 µM):    {len(df_results[df_results['predicted_mic_uM'] < 50])}")

    if not df_potent.empty:
        print(f"\n   Best Candidate:")
        best = df_potent.iloc[0]
        print(f"   Sequence:      {best['sequence']}")
        print(f"   Length:        {best['length']} AA")
        print(f"   Predicted MIC: {best['predicted_mic_uM']:.4f} µM")
        print(f"   -log10(MIC):   {best['predicted_neg_log_mic']:.3f}")
        print(f"   Category:      {best['potency_category']}")

    print("-" * 70)

    return df_results, df_potent


def save_results(
    df_results: pd.DataFrame,
    df_potent: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """
    Save screening results to CSV files.

    Args:
        df_results: All predictions
        df_potent: Filtered potent candidates
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all results
    all_results_path = output_path / "screening_results_all.csv"
    df_results.to_csv(all_results_path, index=False)
    print(f"\n✓ All predictions saved to {all_results_path}")

    # Save potent candidates
    if not df_potent.empty:
        potent_path = output_path / "final_candidates.csv"
        df_potent.to_csv(potent_path, index=False)
        print(f"✓ Potent candidates saved to {potent_path}")
    else:
        print("⚠️  No potent candidates to save")


# ============================================================================
# Model Path Resolution
# ============================================================================

def find_model_paths() -> Tuple[Optional[str], Optional[str]]:
    """
    Intelligently locate model and vectorizer files.

    Returns:
        Tuple of (model_path, vectorizer_path) or (None, None) if not found
    """
    # Navigate from src/ -> week4_peptide_generator/ -> projects/ -> sibling MIC Regression/
    current_dir = Path(__file__).parent.parent.parent
    
    # Possible paths to try
    model_candidates = [
        current_dir / "MIC Regression" / "models" / "mic_predictor.pkl",
        Path(__file__).resolve().parent.parent.parent / "MIC Regression" / "models" / "mic_predictor.pkl",
        Path("../MIC Regression/models/mic_predictor.pkl").resolve(),
    ]

    vectorizer_candidates = [
        current_dir / "MIC Regression" / "models" / "vectorizer.pkl",
        current_dir / "MIC Regression" / "data" / "processed" / "kmer_vectorizer.pkl",
        Path(__file__).resolve().parent.parent.parent / "MIC Regression" / "models" / "vectorizer.pkl",
    ]

    model_path = None
    vectorizer_path = None

    for candidate in model_candidates:
        if candidate.exists():
            model_path = str(candidate)
            print(f"✓ Found model at {model_path}")
            break

    for candidate in vectorizer_candidates:
        if candidate.exists():
            vectorizer_path = str(candidate)
            print(f"✓ Found vectorizer at {vectorizer_path}")
            break

    return model_path, vectorizer_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main screening pipeline."""
    print("\n" + "=" * 70)
    print("AI-GENERATED PEPTIDE CANDIDATE SCREENING")
    print("=" * 70)

    # Find candidate file
    candidates_path = Path("results/generated_peptides.csv")
    if not candidates_path.exists():
        candidates_path = Path("../results/generated_peptides.csv")

    if not candidates_path.exists():
        print(f"✗ Generated peptides file not found")
        print(f"   Tried: results/generated_peptides.csv")
        return

    print(f"✓ Using candidates from: {candidates_path}")

    # Find model paths
    model_path, vectorizer_path = find_model_paths()

    if model_path is None:
        print("\n✗ MIC predictor model not found")
        print("   Checked standard locations:")
        print("   - ../MIC Regression/models/mic_predictor.pkl")
        print("   - projects/MIC Regression/models/mic_predictor.pkl")
        return

    if vectorizer_path is None:
        print("\n⚠️  K-mer vectorizer not found (continuing with physicochemical features only)")

    # Run screening
    df_all, df_potent = screen_candidates(
        candidates_csv=str(candidates_path),
        model_path=model_path,
        vectorizer_path=vectorizer_path or "",
        mic_threshold=5.0,
    )

    # Save results
    if not df_all.empty:
        save_results(df_all, df_potent, output_dir="results")

    # Summary
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    if not df_potent.empty:
        print(f"\n✅ SUCCESS: Identified {len(df_potent)} potent candidates!")
        print(f"\nTop 3 Candidates for Experimental Validation:")
        print("-" * 70)

        for i, (_, row) in enumerate(df_potent.head(3).iterrows(), 1):
            print(
                f"\n{i}. Sequence: {row['sequence']}"
                f"\n   Length: {row['length']} AA | MIC: {row['predicted_mic_uM']:.4f} µM"
                f"\n   Category: {row['potency_category']}"
            )

        print("\n" + "=" * 70)
    else:
        print(f"\n⚠️  No potent candidates found (MIC < 5 µM)")
        print(f"   Consider lowering the MIC threshold or regenerating with different temperature")

    print()


if __name__ == "__main__":
    main()
