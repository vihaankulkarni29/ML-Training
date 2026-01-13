"""Download or synthesize mass spec data for antibiotic resistance experiments."""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Synthetic constants
MOCK_SAMPLES = 1000
SIGNAL_LENGTH = 6000
NUM_CLASSES = 3  # Example antibiotics: Ciprofloxacin, Ceftriaxone, Meropenem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DRIAMS subset or generate synthetic spectra for development."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="Generate mock data or view download instructions",
    )
    parser.add_argument("--year", type=int, default=2018, help="DRIAMS year (real mode)")
    parser.add_argument(
        "--species",
        nargs="+",
        default=["Escherichia coli", "Staphylococcus aureus"],
        help="Target species (real mode)",
    )
    parser.add_argument(
        "--antibiotics",
        nargs="+",
        default=["Ciprofloxacin", "Ceftriaxone"],
        help="Target antibiotics (real mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to store outputs",
    )
    return parser.parse_args()


def try_import_maldi_learn():
    try:
        from maldi_learn.datasets import load_driams  # type: ignore

        return load_driams
    except Exception as exc:  # noqa: BLE001
        print("maldi_learn not available; real download disabled.")
        print("Install with: pip install git+https://github.com/BorgwardtLab/maldi-learn.git")
        print(f"Details: {exc}")
        return None


def download_subset(
    load_fn,
    year: int,
    species: List[str],
    antibiotics: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list, y_list, meta_list = [], [], []
    for sp in species:
        for ab in antibiotics:
            try:
                data = load_fn(
                    site="driamsa",
                    year=str(year),
                    species=sp,
                    antibiotic=ab,
                    return_metadata=True,
                    verbose=False,
                )
            except TypeError:
                data = load_fn(
                    site="driamsa",
                    year=str(year),
                    species=sp,
                    antibiotics=[ab],
                    return_metadata=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {sp} / {ab} due to download error: {exc}")
                continue

            if data is None:
                print(f"No data for {sp} / {ab} in {year}")
                continue

            try:
                X, y, meta = data
            except ValueError:
                X, y, meta = data["X"], data["y"], data["metadata"]

            X_list.append(np.asarray(X))
            y_list.append(np.asarray(y))
            meta_df = pd.DataFrame(meta)
            meta_df["species"] = sp
            meta_df["antibiotic"] = ab
            meta_list.append(meta_df)

    if not X_list:
        raise RuntimeError("No data downloaded; check parameters or connectivity.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    meta_all = pd.concat(meta_list, ignore_index=True)
    return X_all, y_all, meta_all


def generate_synthetic_data(output_dir: Path) -> None:
    print("\n[INFO] Generating SYNTHETIC Data (Mode: Dev)...")
    print("       This allows you to test the pipeline without downloading 86GB.")
    print(f"       Generating {MOCK_SAMPLES} spectra of length {SIGNAL_LENGTH}...")

    X = np.abs(np.random.randn(MOCK_SAMPLES, SIGNAL_LENGTH) * 0.1)
    for i in range(MOCK_SAMPLES):
        num_peaks = np.random.randint(5, 20)
        peak_locs = np.random.randint(0, SIGNAL_LENGTH, num_peaks)
        X[i, peak_locs] += np.random.uniform(1.0, 5.0, num_peaks)

    y = np.random.randint(0, 2, size=(MOCK_SAMPLES, NUM_CLASSES))

    meta_df = pd.DataFrame(
        {
            "sample_id": [f"mock_{i}" for i in range(MOCK_SAMPLES)],
            "species": ["Escherichia coli"] * (MOCK_SAMPLES // 2)
            + ["Staphylococcus aureus"] * (MOCK_SAMPLES - (MOCK_SAMPLES // 2)),
            "year": [2018] * MOCK_SAMPLES,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / "X.npy", X.astype(np.float32))
    np.save(output_dir / "y.npy", y.astype(np.float32))
    meta_df.to_csv(output_dir / "metadata.csv", index=False)

    print(f"[SUCCESS] Saved synthetic data to {output_dir}")
    print(f"          X shape: {X.shape}")
    print(f"          y shape: {y.shape}")


def download_real_instructions() -> None:
    print("\n[INFO] REAL Data Download Instructions (DRIAMS-B)")
    print("----------------------------------------------------")
    print("The DRIAMS dataset is massive. For this project, start with DRIAMS-B (3.7GB).")
    print("\nOPTION 1: Manual Download (Recommended)")
    print("1. Go to: https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q")
    print("2. Download 'DRIAMS_B.tar.gz'")
    print("3. Extract and look for the 'binned_6000' folder.")
    print("\nOPTION 2: Use the 'maldi-learn' package (Advanced)")
    print("   pip install git+https://github.com/BorgwardtLab/maldi-learn.git")
    print("----------------------------------------------------")
    print("For now, re-run this script with '--mode synthetic' to build your model immediately.")


def main() -> None:
    args = parse_args()

    if args.mode == "synthetic":
        generate_synthetic_data(Path(args.output_dir))
        return

    load_fn = try_import_maldi_learn()
    if load_fn is None:
        download_real_instructions()
        sys.exit(0)

    try:
        X, y, meta = download_subset(load_fn, args.year, args.species, args.antibiotics)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_path = args.output_dir / "X.npy"
    y_path = args.output_dir / "y.npy"
    meta_path = args.output_dir / "metadata.csv"

    try:
        np.save(x_path, X)
        np.save(y_path, y)
        meta.to_csv(meta_path, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to save outputs: {exc}")
        sys.exit(1)

    print(f"Downloaded {len(X)} spectra, Shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Metadata rows: {len(meta)}")
    print(f"Saved X → {x_path}")
    print(f"Saved y → {y_path}")
    print(f"Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
