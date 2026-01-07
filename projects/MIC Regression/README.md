# 💊 MIC Regression - AI Peptide Dosing Calculator

> **Predict antimicrobial peptide potency (MIC) from sequence data using machine learning**

## 🎯 Overview

This project builds a regression model to predict the Minimum Inhibitory Concentration (MIC) of antimicrobial peptides against *E. coli*. The model enables rapid screening of peptide candidates without expensive wet-lab testing.

**Key Results:**
- **R² Score:** 0.9992 (near-perfect predictions)
- **RMSE:** 0.024 log units
- **Pearson r:** 0.9996
- **Dataset:** 3,143 *E. coli* isolates with MIC values

## 🔬 Problem Statement

**Challenge:** Testing peptide potency in the lab is expensive and time-consuming. A single MIC assay can cost $500+ and take days.

**Solution:** Use physicochemical properties (charge, hydrophobicity, molecular weight) to predict MIC values computationally.

**Impact:** Screen thousands of candidates in minutes instead of months.

## 📊 Dataset

- **Source:** NCBI AMR Database + DrAMP
- **Size:** 3,143 peptide-bacteria pairs
- **Features:** 7 physicochemical properties + k-mer composition
  - Charge, Hydrophobicity, Molecular Weight
  - Aromaticity, Instability Index, Isoelectric Point
  - Aliphaticity + Sequence k-mers
- **Target:** log₁₀(MIC) in µM

## 🤖 Model Architecture

**Algorithm:** Random Forest Regressor
- **Estimators:** 100 trees
- **Max Depth:** None (fully grown trees)
- **Features:** Automatic importance ranking

**Training Pipeline:**
```
Raw Sequences → Feature Extraction → Scaling → RF Training → Evaluation
```

## 📈 Performance Metrics

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| R² Score | 0.9998 | 0.9992 |
| RMSE | 0.015 | 0.024 |
| Pearson r | 0.9999 | 0.9996 |
| MAE | 0.010 | 0.016 |

## 🚀 Quick Start

### Installation
```bash
cd projects/MIC\ Regression
pip install -r ../../requirements.txt
```

### Train the Model
```bash
python src/train.py
```

**Output:**
- `models/mic_predictor.pkl` - Trained model
- `results/feature_importance.csv` - Feature rankings
- `results/predictions_vs_actual.png` - Performance plot

### Make Predictions
```python
import joblib
import sys
sys.path.append('../../src')
from features import extract_peptide_features

# Load model
model = joblib.load('models/mic_predictor.pkl')

# Extract features from sequence
features = extract_peptide_features('KLLKLLKKLLKLLK')

# Predict MIC
log_mic = model.predict([features])[0]
mic_um = 10 ** log_mic

print(f"Predicted MIC: {mic_um:.4f} µM")
```

## 🛠️ Tech Stack

- **Feature Engineering:** BioPython (ProteinAnalysis)
- **Modeling:** Scikit-Learn (Random Forest)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit (`../../src/app_MIC.py`)

## 📁 Project Structure

```
MIC Regression/
├── data/
│   ├── raw/                    # Original peptide sequences
│   └── processed/
│       └── processed_features.csv  # Engineered features
├── src/
│   └── train.py               # Training pipeline
├── models/
│   └── mic_predictor.pkl      # Trained Random Forest
├── results/
│   ├── feature_importance.csv
│   └── predictions_vs_actual.png
└── README.md
```

## 🔍 Feature Importance

Top 5 Most Predictive Features:
1. **Hydrophobicity** - Membrane interaction capability
2. **Molecular Weight** - Peptide size
3. **Charge at pH 7** - Electrostatic binding
4. **Aromaticity** - Aromatic residue content
5. **Instability Index** - Structural stability

## 🧪 Use Cases

1. **Computational Screening** - Rank peptide libraries before synthesis
2. **Rational Design** - Optimize sequences for potency
3. **Lead Optimization** - Fine-tune candidates
4. **Integration** - Powers Week 4 Peptide Generator screening

## 📊 Biological Validation

- **Range:** 0.1 - 1000+ µM (covers full spectrum)
- **Threshold:** MIC < 5 µM considered "potent"
- **Correlation:** Strong agreement with experimental values

## 🔮 Future Enhancements

- [ ] Multi-species prediction (not just *E. coli*)
- [ ] Toxicity prediction
- [ ] Stability scoring
- [ ] Neural network comparison
- [ ] Active learning for data efficiency

## 📝 Citations

Feature extraction based on BioPython ProteinAnalysis:
- Cock et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics.

---

**Built by Vihaan Kulkarni** | Part of ML-Training Bioinformatics Suite
