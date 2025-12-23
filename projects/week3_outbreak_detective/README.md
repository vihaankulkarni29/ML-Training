# Week 3: Outbreak Detective

Identify hidden bacterial "clans" by clustering genomic fingerprints from 120,000+ Asian bacterial samples using machine learning.

## 🔬 Project Overview

This project analyzes antimicrobial resistance (AMR) genes across 34,871 bacterial samples from Asia to discover hidden genomic clusters that may represent spreading pathogen "clans" across borders.

**Tech Stack:**
- **Data Prep:** Pandas (Pivot Tables, Categorical Encoding)
- **Dimensionality Reduction:** PCA (54 genes → 3 dimensions)
- **Clustering:** K-Means (6 bacterial clans identified)
- **Visualization:** Matplotlib (3D scatter plots)

## 📊 Results Summary

**Dataset:**
- 119,926 gene detections across 34,871 unique bacterial samples
- 54 AMR genes analyzed (after filtering from 63)
- 726 geographic locations (China, Japan, Thailand, India, etc.)

**Clusters Identified:**
- **Clan 0:** 21,705 samples (62.2%) - Dominant Japan/China cluster
- **Clan 1:** 3,347 samples (9.6%) - China/India heavy
- **Clan 2:** 1,397 samples (4.0%) - China Guangxi region
- **Clan 3:** 3,269 samples (9.4%) - Guangdong province cluster
- **Clan 4:** 579 samples (1.7%) - Rare Jiangsu/Thailand variant
- **Clan 5:** 4,574 samples (13.1%) - China/Thailand spread

**Key Findings:**
- PCA captured 12.19% variance (3 components)
- 95.31% sparse matrix (most samples have few genes)
- Clear geographic clustering patterns observed

## 📁 Project Structure

```
week3_outbreak_detective/
├── data/
│   ├── india.csv              # Raw input (120k rows, 27.5 MB)
│   ├── genotype_matrix.csv    # Binary matrix (34,871 x 54 genes)
│   └── metadata.csv           # Sample → Location mapping
├── results/
│   ├── outbreak_clans_3d.png      # 3D visualization
│   └── cluster_results.csv        # Full clustering results
├── src/
│   ├── process_matrix.py      # Long → Wide matrix transformation
│   └── cluster_analysis.py    # PCA + K-Means + Visualization
└── README.md
```

## 🚀 Quick Start

**1. Data Preprocessing** - Convert long-format CSV to binary matrix:

```bash
python src/process_matrix.py \
  --input data/india.csv \
  --out-matrix data/genotype_matrix.csv \
  --out-metadata data/metadata.csv \
  --low 0.001 --high 0.999
```

**Expected Output:**
```
Matrix Shape: (34871, 54)
Sparsity: 95.31%
Saved genotype_matrix.csv
Saved metadata.csv
```

**2. Clustering Analysis** - Run PCA + K-Means:

```bash
python src/cluster_analysis.py \
  --matrix data/genotype_matrix.csv \
  --metadata data/metadata.csv \
  --n-clusters 6 \
  --out-plot results/outbreak_clans_3d.png \
  --out-results results/cluster_results.csv
```

**Expected Output:**
```
PCA variance: 12.19%
6 clans identified
Saved 3D visualization
Saved cluster results
```

## 📈 Outputs

1. **`genotype_matrix.csv`** - Binary presence/absence matrix (samples × genes)
2. **`metadata.csv`** - Geographic metadata for each sample
3. **`outbreak_clans_3d.png`** - 3D scatter plot showing genomic clusters
4. **`cluster_results.csv`** - Full results with clan assignments and PCA coordinates

## 🧬 Technical Details

**Data Transformation:**
- Input: Long format (BioSample, Gene, Location) → 119,926 rows
- Output: Wide binary matrix (34,871 samples × 54 genes)
- Filtering: Removed genes present in <0.1% or >99.9% of samples

**Machine Learning Pipeline:**
1. **StandardScaler:** Normalize gene counts (mean=0, var=1)
2. **PCA:** Reduce 54 dimensions → 3 principal components
3. **K-Means:** Cluster samples into 6 genomic clans
4. **Matplotlib:** 3D visualization with color-coded clans

**Memory Optimizations:**
- Categorical dtypes for high-cardinality strings
- Int8 for binary matrix (vs default int64)
- Crosstab for efficient pivoting

## 🌍 Geographic Insights

Top locations per clan:
- **Clan 0:** Japan (3,423), China (1,551)
- **Clan 1:** China (440), India (136)
- **Clan 5:** China (489), Thailand (348)

## 📦 Requirements

```bash
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## 🔗 Repository

GitHub: [vihaankulkarni29/ML-Training](https://github.com/vihaankulkarni29/ML-Training/tree/main/projects/week3_outbreak_detective)
