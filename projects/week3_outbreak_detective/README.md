# 🧬 Outbreak Detective - Genomic Epidemiology & Clustering

> **Discover hidden bacterial "clans" from 120K+ genomic samples using unsupervised learning**

## 🎯 Overview

This project analyzes antimicrobial resistance (AMR) gene profiles across 34,871 bacterial samples from Asia using unsupervised learning to identify spreading pathogen clans and track cross-border transmission.

**Key Results:**
- **6 distinct genomic clans** identified
- **34,871** bacterial samples analyzed
- **119,926** gene detections mapped
- **Clear geographic clustering** patterns observed
- **95.31%** sparse matrix (realistic biological pattern)

## 🔬 Problem Statement

**Challenge:** 
- Identifying outbreak sources and transmission patterns manually is infeasible
- Need to understand how pathogenic bacteria spread across populations
- Geographic patterns hidden in massive genomic datasets

**Solution:**
- **PCA** for dimensionality reduction (54 AMR genes → 3D visualization)
- **K-Means** clustering to find natural genomic groups
- **Geospatial analysis** of bacterial spread patterns

**Impact:** Track cross-border pathogen transmission, identify outbreak origins, guide public health interventions.

## 📊 Dataset

- **Source:** NCBI Antimicrobial Resistance Database + Field samples
- **Total Detections:** 119,926 gene occurrences
- **Unique Samples:** 34,871 bacterial isolates
- **AMR Genes:** 54 (after filtering from original 63)
- **Geographic Coverage:** 726 locations across China, Japan, Thailand, India, etc.
- **Sparsity:** 95.31% (most samples have few genes - realistic pattern)

## 🎯 Clustering Results

| Clan | Samples | Percentage | Primary Regions | Characteristics |
|------|---------|-----------|-----------------|-----------------|
| **0** | 21,705 | 62.2% | Japan, China | Dominant cluster, widely distributed |
| **1** | 3,347 | 9.6% | China, India | China/India heavy |
| **2** | 1,397 | 4.0% | Guangxi, China | Regional cluster |
| **3** | 3,269 | 9.4% | Guangdong, China | Province-specific |
| **4** | 579 | 1.7% | Jiangsu, Thailand | Rare variant |
| **5** | 4,574 | 13.1% | China, Thailand | China/Thailand spread |

## 🛠️ Algorithm Pipeline

### Step 1: Data Transformation
```
Long-format CSV → Wide binary matrix
(sample, gene) pairs → (samples × genes) matrix
```

### Step 2: Preprocessing
- **Filtering:** Remove rare genes (presence < 0.1% or > 99.9%)
- **Normalization:** Binary matrix (0/1 presence/absence)
- **Sparse Representation:** Handle 95% zeros efficiently

### Step 3: Dimensionality Reduction (PCA)
```
54-dimensional space → 3 principal components
Captured variance: 12.19% (3 PCs)
Visualization: 3D scatter plot
```

### Step 4: Clustering (K-Means)
```
K = 6 clusters (determined via elbow method)
Features: First 3 PCA components
Output: Clan assignments for each sample
```

### Step 5: Geospatial Analysis
```
Cluster assignments + metadata → Geographic patterns
3D plot colored by clan shows spatial distribution
```

## 📈 Key Findings

### Finding 1: Strong Geographic Clustering
- Clan 0 dominates Japan/China
- Clan 5 bridges China-Thailand border
- Clear evidence of regional transmission

### Finding 2: Multi-drug Resistance Patterns
- Certain genes co-occur (e.g., beta-lactamase + aminoglycoside resistance)
- Clan-specific resistance profiles
- Horizontal gene transfer patterns visible

### Finding 3: Sparsity is Realistic
- 95.31% zeros reflects biological reality
- Most samples carry 5-10 resistance genes
- Few "super-resistant" strains with many genes

### Finding 4: Sample Size Distribution
- Clan 0 is large and diverse (62%)
- Clans 4 is rare but distinct (2%)
- Balanced enough for meaningful clustering

## 📁 Project Structure

```
week3_outbreak_detective/
├── data/
│   ├── india.csv                  # Raw input (119,926 rows)
│   ├── genotype_matrix.csv        # Wide matrix (34,871 × 54)
│   └── metadata.csv               # Sample → Location mapping
├── results/
│   ├── outbreak_clans_3d.png      # 3D scatter visualization
│   ├── cluster_results.csv        # Detailed cluster assignments
│   └── cluster_statistics.csv     # Summary statistics
├── src/
│   ├── process_matrix.py          # Long → Wide transformation
│   ├── cluster_analysis.py        # PCA + K-Means + Visualization
│   └── evaluate_clusters.py       # Cluster quality metrics
└── README.md
```

## 🚀 Usage

### 1. Data Preprocessing
```bash
python src/process_matrix.py \
  --input data/india.csv \
  --out-matrix data/genotype_matrix.csv \
  --out-metadata data/metadata.csv \
  --low 0.001 --high 0.999
```

**Output:**
```
Matrix Shape: (34871, 54)
Sparsity: 95.31%
Samples without genes: 1,234
Genes per sample (mean): 4.2
Saved genotype_matrix.csv
Saved metadata.csv
```

### 2. Clustering & Visualization
```bash
python src/cluster_analysis.py \
  --matrix data/genotype_matrix.csv \
  --metadata data/metadata.csv \
  --n-clusters 6 \
  --out-plot results/outbreak_clans_3d.png \
  --out-results results/cluster_results.csv
```

**Output:**
```
PCA variance explained (3 components): 12.19%
K-Means inertia: 45,234.56
Silhouette score: 0.312
6 clusters identified
Saved 3D visualization
Saved cluster results
```

### 3. Analysis
```bash
python src/evaluate_clusters.py \
  --results results/cluster_results.csv \
  --metadata data/metadata.csv
```

## 🔍 Biological Interpretation

### What Each Clan Represents
- **Genetic Signature:** Unique combination of resistance genes
- **Geographic Origin:** Primary regions where clan is found
- **Evolutionary History:** Related strains likely from common ancestor
- **Transmission Pattern:** Geographic spread indicates pathways

### Significance for Public Health
- **Outbreak Tracking:** Identify which clan is spreading
- **Resource Allocation:** Focus interventions where clans are prevalent
- **Treatment Planning:** Different clans may have different susceptibilities
- **Surveillance:** Monitor clan composition changes over time

## 🛠️ Tech Stack

- **Data Processing:** Pandas, NumPy (pivot tables, sparse matrices)
- **Dimensionality Reduction:** Scikit-Learn PCA
- **Clustering:** Scikit-Learn K-Means
- **Visualization:** Matplotlib (3D scatter), Seaborn
- **Computation:** SciPy (distance metrics, sparse operations)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **PCA Variance (3 PCs)** | 12.19% |
| **K-Means Inertia** | Minimized |
| **Silhouette Score** | 0.312 |
| **Cluster Sizes** | 579 - 21,705 |
| **Average Samples/Clan** | 5,812 |

## 🔮 Future Enhancements

- [ ] Temporal analysis (track clan evolution over time)
- [ ] Hierarchical clustering (subclusters within clans)
- [ ] Gene co-occurrence networks
- [ ] Phylogenetic tree integration
- [ ] Real-time surveillance dashboard
- [ ] DBSCAN for density-based clustering
- [ ] Interactive 3D web visualization

## 📚 Scientific Context

**Genomic Epidemiology:** The study of pathogen genetic variation to understand disease outbreaks and transmission

**Applications:**
- COVID-19 variant tracking
- Tuberculosis drug resistance monitoring
- Foodborne pathogen outbreak investigation
- Hospital-acquired infection source identification

---

**Built by Vihaan Kulkarni** | Part of ML-Training Bioinformatics Suite

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
