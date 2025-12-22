# Week 3: Outbreak Detective

Identify hidden bacterial "clans" by clustering genomic fingerprints from 100k+ samples.

## Structure

```
week3_outbreak_detective/
├── data/
│   └── india.csv              # raw input (ignored in git)
├── results/                   # figures, reports
├── src/
│   ├── process_matrix.py      # Long -> Wide binary matrix + metadata
│   └── cluster_analysis.py    # PCA + KMeans + Plotly (stub)
└── README.md
```

## Quick Start

1) Place `india.csv` into `data/` (already copied for you).

2) Build binary genotype matrix and metadata:

```bash
python src/process_matrix.py --input data/india.csv \
  --out-matrix data/genotype_matrix.csv \
  --out-metadata data/metadata.csv \
  --low 0.001 --high 0.999
```

- Output: `data/genotype_matrix.csv`, `data/metadata.csv`
- Stats printed: Matrix shape and sparsity.

3) (Later) Run clustering and 3D visualization (to be implemented):

```bash
python src/cluster_analysis.py
```

## Notes
- Uses categorical dtypes and `pd.crosstab` for memory efficiency.
- Filters near-constant genes (<0.1% or >99.9%) to improve clustering quality.
- Recommend Python 3.10+ with `pandas` and `numpy` installed.
