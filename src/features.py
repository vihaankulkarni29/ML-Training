"""Protein sequence feature extraction for MIC regression.

Loads sequences, computes physicochemical properties via Biopython, and
extracts k-mer (dipeptide) features to capture sequence order information.
Combines both feature sets for downstream modeling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.feature_extraction.text import CountVectorizer


def _safe_analyze(seq: Any) -> Dict[str, float]:
    """Compute physicochemical properties for one sequence safely."""
    if not isinstance(seq, str) or len(seq) == 0:
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": 0,
            "positive_charge": 0,
        }

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
    except Exception:
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": 0,
            "positive_charge": 0,
        }


def extract_features(input_path: str, output_path: str) -> pd.DataFrame:
    """Load sequences, compute physicochemical + k-mer features, save augmented DataFrame."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    if "SEQUENCE" not in df.columns:
        raise ValueError("Input CSV must contain a 'SEQUENCE' column.")

    # === STEP 1: Physicochemical Properties ===
    print("Computing physicochemical properties...")
    feature_keys = [
        "mol_weight",
        "aromaticity",
        "instability_index",
        "isoelectric_point",
        "gravy",
        "length",
        "positive_charge",
    ]
    features: Dict[str, list] = {k: [] for k in feature_keys}

    total = len(df)
    for idx, seq in enumerate(df["SEQUENCE"]):
        vals = _safe_analyze(seq)
        for k in feature_keys:
            features[k].append(vals[k])
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1} / {total} sequences...")

    for k, v in features.items():
        df[k] = v

    # === STEP 2: K-mer (Dipeptide) Features ===
    print("Extracting k-mer (2-mer) features...")
    # Treat sequences as character sequences, extract bigrams (dipeptides)
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(2, 2),  # Dipeptides only
        lowercase=False,      # Amino acids are case-sensitive
        min_df=5              # Ignore rare k-mers (appear in <5 sequences)
    )
    
    # Fit and transform sequences to sparse matrix
    kmer_matrix = vectorizer.fit_transform(df["SEQUENCE"].fillna(""))
    
    # Convert to DataFrame with k-mer column names
    kmer_df = pd.DataFrame(
        kmer_matrix.toarray(),
        columns=[f"kmer_{kmer}" for kmer in vectorizer.get_feature_names_out()]
    )
    
    print(f"  Extracted {kmer_df.shape[1]} dipeptide features (min_df=5)")
    
    # === STEP 3: Combine All Features ===
    df_combined = pd.concat([df, kmer_df], axis=1)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_file, index=False)
    
    # Save vectorizer for deployment
    vectorizer_path = output_file.parent / "kmer_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✓ Saved augmented features to {output_file}")
    print(f"✓ Saved k-mer vectorizer to {vectorizer_path}")
    print(f"Total features: {len(feature_keys)} physicochemical + {kmer_df.shape[1]} k-mers = {len(feature_keys) + kmer_df.shape[1]}")
    
    return df_combined


if __name__ == "__main__":
    INPUT_PATH = "projects/MIC Regression/data/raw/ecolitraining_set_80.csv"
    OUTPUT_PATH = "projects/MIC Regression/data/processed/processed_features.csv"
    extract_features(INPUT_PATH, OUTPUT_PATH)
