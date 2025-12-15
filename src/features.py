"""Protein sequence feature extraction for MIC regression.

Loads sequences, computes physicochemical properties via Biopython, and
saves enriched features for downstream modeling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


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
    """Load sequences, compute features, and save augmented DataFrame."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    if "SEQUENCE" not in df.columns:
        raise ValueError("Input CSV must contain a 'SEQUENCE' column.")

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
            print(f"Processed {idx + 1} / {total} sequences...")

    for k, v in features.items():
        df[k] = v

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved features to {output_file}")
    return df


if __name__ == "__main__":
    INPUT_PATH = "projects/MIC Regression/data/raw/ecolitraining_set_80.csv"
    OUTPUT_PATH = "data/processed_features.csv"
    extract_features(INPUT_PATH, OUTPUT_PATH)
