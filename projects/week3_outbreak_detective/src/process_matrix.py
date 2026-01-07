import os
import math
import argparse
import pandas as pd
import numpy as np

# Scientific Validation: Sparse Data Bias Mitigation
# This script transforms a massive genomic dataset from Long format
# into a binary Wide matrix suitable for clustering.
# 
# CRITICAL: Filters out samples from rare locations (<5 isolates) to prevent
# clustering artifacts from single-sample or under-represented locations.
# This ensures statistical robustness of downstream analysis.
#
# Memory management notes:
# - Limit loaded columns with usecols
# - Use categorical dtypes for high-cardinality strings to reduce RAM
# - Build binary matrix via crosstab (dense) then downcast to int8
# - Filter near-constant columns to reduce dimensionality


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    usecols = ["BioSample", "Element symbol", "Location"]
    dtypes = {
        # categories assigned after read to leverage pandas' categorical encoding
        # keep as object initially to avoid mixed types inference
        "BioSample": "object",
        "Element symbol": "object",
        "Location": "object",
    }
    df = pd.read_csv(input_path, usecols=usecols, dtype=dtypes, low_memory=False)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows missing essential keys
    df = df.dropna(subset=["BioSample", "Element symbol"]).copy()

    # Categorical conversion to reduce memory usage dramatically for repeated strings
    for col in ["BioSample", "Element symbol", "Location"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def filter_rare_locations(df: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    """
    Scientific Validation: Remove samples from locations with <min_samples isolates.
    Prevents clustering artifacts from single-sample or under-represented locations.
    
    Args:
        df: Input dataframe with BioSample and Location columns
        min_samples: Minimum number of isolates per location (default: 5)
        
    Returns:
        Filtered dataframe with only samples from well-represented locations
    """
    if "Location" not in df.columns:
        return df
    
    # Count samples per location
    location_counts = df['Location'].value_counts()
    rare_locations = location_counts[location_counts < min_samples].index.tolist()
    
    # Get unique BioSamples in rare locations
    samples_to_drop = df[df['Location'].isin(rare_locations)]['BioSample'].nunique()
    
    # Filter out rare locations
    df_filtered = df[~df['Location'].isin(rare_locations)].copy()
    
    if len(rare_locations) > 0:
        print(f"\n✓ Scientific Validation: Sparse Data Mitigation")
        print(f"   Dropped {samples_to_drop} samples from {len(rare_locations)} rare locations (<{min_samples} isolates)")
        print(f"   Remaining samples: {df_filtered['BioSample'].nunique():,}")
        print(f"   Remaining locations: {df_filtered['Location'].nunique()}")
    
    return df_filtered


def extract_metadata(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    # Group by BioSample and take the first Location value observed
    if "Location" in df.columns:
        meta = (
            df[["BioSample", "Location"]]
            .dropna(subset=["BioSample"])
            .drop_duplicates(subset=["BioSample", "Location"], keep="first")
            .groupby("BioSample", observed=True)["Location"].first()
            .reset_index()
        )
    else:
        meta = pd.DataFrame({"BioSample": df["BioSample"].unique(), "Location": np.nan})

    # Save metadata
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta.to_csv(out_path, index=False)
    return meta


def build_binary_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Create binary presence/absence matrix: index=BioSample, columns=Element symbol
    # Use crosstab for efficiency; it counts occurrences, we binarize afterwards
    ct = pd.crosstab(df["BioSample"], df["Element symbol"], dropna=False)

    # Convert counts to binary 0/1
    ct = (ct > 0).astype(np.int8)
    return ct


def filter_columns(matrix: pd.DataFrame, low_thresh: float = 0.001, high_thresh: float = 0.999) -> pd.DataFrame:
    # Drop genes that are too rare (<0.1%) or too common (>99.9%)
    n = matrix.shape[0]
    if n == 0:
        return matrix

    min_count = max(1, math.ceil(low_thresh * n))
    max_count = math.floor(high_thresh * n)

    col_counts = matrix.sum(axis=0)
    keep = (col_counts >= min_count) & (col_counts <= max_count)
    filtered = matrix.loc[:, keep.values]
    # Ensure int8 to save memory
    return filtered.astype(np.int8, copy=False)


def compute_sparsity(matrix: pd.DataFrame) -> float:
    total = matrix.size
    if total == 0:
        return 0.0
    zeros = (matrix == 0).sum().sum()
    return float(zeros) / float(total)


def main():
    parser = argparse.ArgumentParser(description="Transform long-format genomic data into a binary genotype matrix with metadata.")
    parser.add_argument("--input", default=os.path.join("data", "india.csv"), help="Path to input CSV (long format)")
    parser.add_argument("--out-matrix", default=os.path.join("data", "genotype_matrix.csv"), help="Output CSV path for genotype matrix")
    parser.add_argument("--out-metadata", default=os.path.join("data", "metadata.csv"), help="Output CSV path for metadata")
    parser.add_argument("--low", type=float, default=0.001, help="Drop genes present in < this fraction of samples")
    parser.add_argument("--high", type=float, default=0.999, help="Drop genes present in > this fraction of samples")
    parser.add_argument("--min-location-samples", type=int, default=5, help="Minimum samples per location (statistical robustness)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = load_data(args.input)
    
    # Scientific Validation: Filter rare locations
    df = filter_rare_locations(df, min_samples=args.min_location_samples)

    print("Extracting metadata ...")
    _ = extract_metadata(df, args.out_metadata)

    print("Building genotype matrix ...")
    X = build_binary_matrix(df)

    print("Filtering columns by frequency ...")
    Xf = filter_columns(X, low_thresh=args.low, high_thresh=args.high)

    # Compute and print stats
    sparsity = compute_sparsity(Xf)
    print(f"\n📊 Final Matrix Statistics:")
    print(f"   Shape: {Xf.shape[0]} samples × {Xf.shape[1]} genes")
    print(f"   Sparsity: {sparsity * 100:.2f}%")

    # Save matrix
    os.makedirs(os.path.dirname(args.out_matrix), exist_ok=True)
    Xf.to_csv(args.out_matrix, index=True)
    print(f"   Saved genotype matrix → {args.out_matrix}")


if __name__ == "__main__":
    main()
