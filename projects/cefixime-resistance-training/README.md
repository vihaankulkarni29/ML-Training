# 🛡️ Ceftriaxone Resistance Predictor

> **Binary classification model for instant antibiotic resistance detection from genomic data**

## 🎯 Problem Statement

**Challenge:** Traditional antibiotic susceptibility testing takes 24-48 hours, delaying critical treatment decisions. Antibiotic-resistant bacteria cause ~1.3M deaths annually (WHO).

**Solution:** Use machine learning to predict ceftriaxone resistance from genomic markers (AMR genes) in seconds, enabling rapid clinical decision-making.

**Impact:** Faster diagnosis → Better antibiotic selection → Reduced mortality

## 📊 Dataset

- **Source:** NCBI Pathogen Detection Database
- **Size:** 4,383 *E. coli* bacterial isolates
- **Features:** 51 antimicrobial resistance (AMR) genes (binary presence/absence)
- **Target:** Ceftriaxone resistance (Susceptible vs Resistant)
- **Class Distribution:** 
  - Resistant: 2,192 samples (50%)
  - Susceptible: 2,191 samples (50%)

## 🔍 Key Insights

### Insight 1: Gene Distribution
- **Finding:** 51 resistance genes show varying prevalence across isolates
- **Top Genes:** blaCTX-M, aac(3), sul2 most common
- **Pattern:** Multi-drug resistance genes often co-occur

### Insight 2: Resistance Patterns
- **Finding:** Certain gene combinations strongly predict resistance
- **Clinical Relevance:** Beta-lactamase genes (blaCTX-M, blaTEM) are key drivers
- **Geographic Variation:** Resistance patterns vary by region

## 🤖 Modeling Approach

### Final Model: Random Forest Classifier
- **Algorithm:** Random Forest with 100 estimators
- **Class Weighting:** Balanced to prioritize sensitivity
- **Feature Selection:** All 51 AMR genes (no reduction needed)

### Performance Metrics (5-Fold Cross-Validation)

| Metric | Mean ± Std |
|--------|------------|
| **Accuracy** | 94.9% ± 0.4% |
| **Sensitivity (Recall)** | 93.9% ± 0.7% |
| **Specificity** | 95.9% ± 0.6% |
| **ROC-AUC** | 0.987 ± 0.003 |
| **F1-Score** | 94.8% |

**Medical AI Focus:** Model optimized for **high sensitivity** to avoid missing resistant cases (false negatives are more dangerous than false positives in clinical settings).

### Test Set Performance
- **Accuracy:** 95.1%
- **Sensitivity:** 94.2%
- **Specificity:** 96.0%
- **Positive Predictive Value:** 95.8%
- **Negative Predictive Value:** 94.5%

### Model Explainability

**Top 5 Most Important Genes:**
1. **blaCTX-M-15** - Extended-spectrum beta-lactamase (25.8% importance)
2. **blaTEM-1** - Beta-lactamase enzyme (12.3%)
3. **aac(3)-IIa** - Aminoglycoside resistance (8.7%)
4. **sul2** - Sulfonamide resistance (7.2%)
5. **qnrS1** - Quinolone resistance (6.1%)

**Interpretation:** Beta-lactamase genes are the primary drivers of ceftriaxone resistance, with aminoglycoside and sulfonamide resistance genes serving as co-indicators.

## 🚀 Deployment

Run locally:
```bash
cd projects/cefixime-resistance-training
streamlit run app.py
```

**App Features:**
- Gene presence/absence input
- Real-time resistance prediction
- Confidence scores
- Feature importance visualization

## 🛠️ Tech Stack

- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-Learn (Random Forest)
- **Visualization:** Plotly (confusion matrix)
- **Evaluation:** Stratified K-Fold CV
- **Deployment:** Streamlit
- **Model Persistence:** Joblib

## 📁 Project Structure

```
cefixime-resistance-training/
├── data/
│   ├── raw/                        # Original NCBI data
│   └── processed/
│       └── dataset_ceftriaxone.csv # Cleaned gene matrix
├── src/
│   ├── preprocessing.py            # Data cleaning
│   ├── train.py                    # Model training pipeline
│   └── make_datasets.py            # Dataset generation
├── models/
│   └── ceftriaxone_model.pkl      # Trained Random Forest
├── results/
│   ├── confusion_matrix.html       # Interactive CM
│   └── feature_importance.csv      # Gene rankings
├── app.py                          # Streamlit deployment
└── README.md
```

## 🧪 Reproducibility

### 1. Install Dependencies
```bash
pip install -r ../../requirements.txt
```

### 2. Preprocess Data
```bash
python src/preprocessing.py
```

### 3. Train Model
```bash
python src/train.py
```

**Expected Output:**
```
✅ Model Accuracy: 95.1%
✅ Sensitivity: 94.2%
✅ Specificity: 96.0%
✅ Model saved to models/ceftriaxone_model.pkl
```

## 💡 Key Learnings

1. **Feature Engineering Not Always Necessary:** Binary gene presence/absence is sufficient - no complex transformations needed
2. **Class Balance Matters:** Balancing classes prevents model bias toward majority class
3. **Medical AI Requires Sensitivity:** In clinical applications, false negatives (missing resistant cases) are more dangerous than false positives
4. **Ensemble Methods Excel:** Random Forest outperformed Logistic Regression and SVM
5. **Cross-Validation is Critical:** 5-fold CV provides robust performance estimates

## 🔮 Future Improvements

- [ ] Multi-antibiotic prediction (cefixime, ciprofloxacin, etc.)
- [ ] Geographic resistance pattern analysis
- [ ] Integration with hospital EHR systems
- [ ] Real-time NCBI data updates
- [ ] SHAP explainability dashboard
- [ ] Multi-species support (beyond *E. coli*)
- [ ] Temporal trend analysis

## 📊 Clinical Impact

**Benefits:**
- ⏱️ **Speed:** Seconds vs 24-48 hours for lab testing
- 💰 **Cost:** Computational prediction vs expensive culture testing
- 🎯 **Accuracy:** 95%+ prediction accuracy
- 🏥 **Decision Support:** Helps clinicians choose effective antibiotics

**Use Case:** Upload bacterial genome → Get instant resistance prediction → Select appropriate antibiotic

---

**Built with ❤️ by Vihaan Kulkarni** | Part of ML-Training Bioinformatics Suite

---

**Author:** Vihaan Kulkarni  
**Date:** 2025-12-14
