"""Example usage of BiophysicalEncoder and AMRDataPipeline."""

import pandas as pd
from src.features import BiophysicalEncoder
from src.data_pipeline import AMRDataPipeline


# === Example 1: Single Mutation Encoding ===
print("Example 1: Encoding Single Mutations")
print("="*60)

encoder = BiophysicalEncoder()

mutations = [
    "S83L",      # Quinolone resistance in gyrA
    "D94G",      # ParE mutation in fluoroquinolone resistance
    "A1P",       # Proline insertion
    "W182R",     # Aromatic → charged change
    "invalid",   # Will fail
]

for mut in mutations:
    features = encoder.get_features(mut)
    if features:
        print(f"\n{mut}:")
        print(f"  ΔHydrophobicity: {features['delta_hydrophobicity']:+.2f}")
        print(f"  ΔCharge: {features['delta_charge']:+.2f}")
        print(f"  ΔMW: {features['delta_mw']:+.2f} Da")
        print(f"  Aromatic change: {bool(features['is_aromatic_change'])}")
        print(f"  Proline involved: {bool(features['is_proline_change'])}")
    else:
        print(f"\n{mut}: ❌ FAILED TO PARSE")


# === Example 2: Full Pipeline ===
print("\n" + "="*60)
print("Example 2: Full Data Pipeline")
print("="*60)

# Create sample raw data
sample_data = pd.DataFrame({
    'mutation': [
        'gyrA_S83L',
        'parE_D94G',
        'rpoB_S450L',
        'gyrA_S83L',  # duplicate
        'invalid_mutation',
    ],
    'antibiotic': [
        'Ciprofloxacin',
        'Ciprofloxacin',
        'Rifampicin',
        'Ciprofloxacin',
        'Ciprofloxacin',
    ],
    'phenotype': ['R', 'R', 'R', 'S', 'unknown'],
})

sample_data.to_csv('data/raw_amr.csv', index=False)

pipeline = AMRDataPipeline()
df_processed, n_initial, n_final = pipeline.process()
pipeline.save()
