# 🧬 Bioinformatics ML Repository

> **Machine Learning Models for Antimicrobial Resistance & Peptide Engineering**

A collection of production-ready ML projects focused on **antimicrobial resistance prediction** and **antimicrobial peptide design**. Built with scikit-learn, Streamlit, and Biopython.

---

## 📋 Projects

### 1. 🛡️ **Ceftriaxone Resistance Predictor** 
*Classification Model for Antibiotic Resistance Detection*

- **Task:** Binary classification (Susceptible vs Resistant)
- **Model:** Random Forest Classifier
- **Accuracy:** 94.9% | Sensitivity: 93.9% | Specificity: 95.9%
- **Data:** 4,383 *E. coli* isolates from NCBI
- **App:** `streamlit run src/app.py`

### 2. 💊 **AI Peptide Dosing Calculator**
*Regression Model for Antimicrobial Peptide Potency Prediction*

- **Task:** MIC (Minimum Inhibitory Concentration) prediction
- **Model:** Random Forest Regressor
- **R² Score:** 0.45 | RMSE: 0.63 log units
- **Data:** 3,143 *E. coli* isolates with MIC values
- **App:** `streamlit run src/app_MIC.py`

---

## 🏥 Biological Context

### Antimicrobial Resistance (AMR)
**Challenge:** Antibiotic-resistant bacteria cause ~1.3M deaths annually (WHO). Traditional lab testing takes 24-48 hours, delaying treatment.

**Solution:** Use genomic markers to **instantly predict resistance** from DNA sequences.

### Antimicrobial Peptides (AMPs)
**Challenge:** Designing potent peptides requires expensive lab screening. Potency varies wildly (MIC: 0.1 - 1000+ µM).

**Solution:** Use machine learning to **predict peptide efficacy** from physicochemical properties, enabling faster design cycles.

---

## � Repository Structure

```
ML-Training/
├── projects/
│   ├── cefixime-resistance-training/    # Antibiotic resistance classifier
│   │   ├── data/
│   │   │   ├── raw/                      # Original NCBI isolates
│   │   │   └── processed/                # Cleaned genotype data
│   │   ├── src/
│   │   │   ├── process.py                # Data preprocessing
│   │   │   └── train.py                  # Model training (RF classifier)
│   │   ├── models/
│   │   │   └── ceftriaxone_model.pkl    # Trained classifier
│   │   └── results/
│   │       ├── confusion_matrix.html     # Interactive CM
│   │       └── feature_importance.csv    # Top resistance genes
│   │
│   └── MIC Regression/                   # Peptide potency regressor
│       ├── data/
│       │   ├── raw/                      # Raw peptide sequences & MIC values
│       │   └── processed/                # Computed physicochemical features
│       ├── src/
│       │   ├── process.py                # Data preprocessing
│       │   └── train.py                  # Model training (RF regressor)
│       ├── models/
│       │   └── mic_predictor.pkl        # Trained regressor
│       └── results/
│           ├── predicted_vs_actual.png   # Predictions visualization
│           └── feature_importance.png    # Top peptide features
│
├── src/
│   ├── app.py                            # Ceftriaxone classifier Streamlit app
│   ├── app_MIC.py                        # MIC regressor Streamlit app
│   └── features.py                       # Biopython feature extraction
│
├── utils/
│   └── model_evaluation.py               # Shared evaluation metrics
│
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/vihaankulkarni29/ML-Training
cd ML-Training

# Install dependencies
pip install -r requirements.txt
```

### Run Applications

**Ceftriaxone Resistance Predictor (Classifier):**
```bash
streamlit run src/app.py
```
Access at `http://localhost:8501`

**AI Peptide Dosing Calculator (Regressor):**
```bash
streamlit run src/app_MIC.py
```
Access at `http://localhost:8501`

---

## 📊 Project 1: Ceftriaxone Resistance Predictor

### Problem Statement
Antibiotic susceptibility testing via culture takes 24-48 hours. Patients with life-threatening infections can't wait. **Goal:** Predict Ceftriaxone resistance instantly from genomic markers.

### Solution
- **Model:** Random Forest Classifier (100 trees, balanced class weights)
- **Data:** 4,383 *E. coli* isolates from NCBI MicroBIGG-E
- **Features:** 352 detected resistance genes/mutations

### Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 94.9% |
| Sensitivity | 93.9% |
| Specificity | 95.9% |
| ROC-AUC | 0.978 |
| Test Set Size | 876 isolates |

### Key Insights
The model independently discovered known resistance mechanisms:
- **blaCTX-M-15** (Extended-Spectrum Beta-Lactamase) - strongest predictor
- **blaCMY-2** (AmpC Cephalosporinase)
- **gyrA_S83L** (Gyrase mutation - fluoroquinolone resistance)

### Biological Mechanism
Beta-lactamase genes encode enzymes that destroy beta-lactam antibiotics (e.g., cephalosporins) before they can bind to bacterial cell walls.

### Files
- Training: `projects/cefixime-resistance-training/src/train.py`
- Model: `projects/cefixime-resistance-training/models/ceftriaxone_model.pkl`
- App: `src/app.py`

---

## 💊 Project 2: AI Peptide Dosing Calculator

### Problem Statement
Antimicrobial peptide (AMP) design is expensive and slow. Wet-lab screening for potency (MIC) takes months. **Goal:** Predict MIC instantly from sequence, enabling computational design cycles.

### Solution
- **Model:** Random Forest Regressor (100 trees)
- **Data:** 3,143 *E. coli* isolates with MIC values (NCBI)
- **Target:** `neg_log_mic_microM` (-log10 of MIC in µM)

### Performance Metrics
| Metric | Value |
|--------|-------|
| R² Score | 0.45 |
| RMSE | 0.63 log units |
| Pearson r | 0.674 |
| p-value | < 0.001 |
| Test Set Size | 629 peptides |

### Interpretation
- RMSE of 0.63 log units = ~4.25x fold-change in actual MIC values
- Model explains 44.6% of variance in test data
- Strong correlation with actual values (p < 0.001)

### Feature Engineering
Computed 7 physicochemical properties using Biopython:
1. **Molecular Weight** - correlates with toxicity vs efficacy
2. **Aromaticity** - aromatic residues enhance membrane interaction
3. **Instability Index** - peptide stability in vivo
4. **Isoelectric Point** - charge affects cellular uptake
5. **GRAVY** (hydrophobicity) - hydrophobic residues improve activity
6. **Length** - longer peptides often more potent but less specific
7. **Positive Charge** - (K + R count) - important for bacterial binding

### Potency Categories
- < 2 µM: 💎 Excellent (highly potent)
- 2-10 µM: ✅ Good (reasonable activity)
- 10-50 µM: ⚠️ Weak (marginal)
- > 50 µM: ❌ Inactive (not viable)

### Files
- Feature extraction: `src/features.py`
- Training: `projects/MIC Regression/src/train.py`
- Model: `projects/MIC Regression/models/mic_predictor.pkl`
- Processed data: `projects/MIC Regression/data/processed/processed_features.csv`
- App: `src/app_MIC.py`

---

## 🔬 Technical Stack

### Data Science
- **Pandas:** Data manipulation & analysis
- **NumPy:** Numerical computations
- **Scikit-Learn:** RandomForest classifiers & regressors
- **Biopython:** Protein sequence analysis (`Bio.SeqUtils.ProtParam`)
- **SciPy:** Statistical tests (Pearson correlation, etc.)

### Visualization
- **Matplotlib:** Static publication-ready plots
- **Plotly:** Interactive HTML charts
- **Kaleido:** PNG export from Plotly

### Deployment
- **Streamlit:** Interactive web apps (no frontend coding)
- **Joblib:** Model persistence (.pkl files)
- **GitHub:** Version control & deployment integration

---

## 🏥 Biological Background

### Antimicrobial Resistance (AMR)

**Global Impact:**
- ~1.3M deaths/year attributable to AMR (WHO, 2022)
- Top 10 global health threat
- Economic cost: $100B+ annually in healthcare

**Genetic Basis (Ceftriaxone Example):**
1. **Enzymatic Inactivation:** blaCTX-M genes produce beta-lactamases that hydrolyze beta-lactam ring
2. **Target Modification:** gyrA mutations alter DNA gyrase binding site
3. **Efflux Pumps:** acrB overexpression exports antibiotics before they act

### Antimicrobial Peptides (AMPs)

**Natural Defense:**
- Found in all life forms (immune system, skin, GI tract)
- Kill bacteria via direct membrane disruption
- Less likely to develop resistance (multiple mechanisms)

**Design Challenge:**
- Potency (MIC) varies 1000-fold (0.1 - 100+ µM)
- Toxicity risk increases with potency
- Design space is massive (20^n for n-length peptides)

**ML Solution:**
- Use physicochemical properties to predict potency
- Enable rational design instead of random screening
- Reduce wet-lab costs & timelines

---

## 📚 Literature & Data Sources

### Antimicrobial Resistance
- **NCBI MicroBIGG-E:** https://microbiggdata.ncbi.nlm.nih.gov/ (genotypes + phenotypes)
- **EUCAST Guidelines:** https://www.eucast.org/ (standard testing methods)
- **CARD Database:** https://card.mcmaster.ca/ (resistance gene annotations)

### Antimicrobial Peptides
- **APD (APD3):** https://aps.unmc.edu/APD/ (AMP database)
- **BioPep:** https://www.bipep.org/ (peptide bioactivity)

### Biopython Feature Extraction
- `ProteinAnalysis` documentation: https://biopython.org/wiki/Documentation

---

## ⚠️ Disclaimers

### Ceftriaxone Predictor
**For research/educational use only.** Not a clinical diagnostic device.
- Always confirm predictions with lab culture + antibiotic susceptibility testing (EUCAST/CLSI)
- Consult clinical microbiology before treatment decisions
- Models trained on specific *E. coli* population; validate locally

### MIC Calculator
**For research/design purposes only.** Not validated for clinical use.
- Predicted MIC is a computational estimate; always validate experimentally
- Model trained on specific data; performance may vary on novel sequences
- Use as design guidance, not final arbiter of peptide efficacy

---

## 🎯 Roadmap

### Q1 2025
- [ ] Multi-organism support (Klebsiella, Pseudomonas)
- [ ] SHAP explainability for individual predictions
- [ ] Confidence intervals for MIC predictions

### Q2 2025
- [ ] REST API for integration with LIS systems
- [ ] Additional antibiotics (fluoroquinolones, aminoglycosides)
- [ ] Uncertainty quantification via Bayesian methods

### Q3 2025
- [ ] Mobile app (iOS/Android) for field deployment
- [ ] Real-time database updates from NCBI
- [ ] Community contribution framework

---

## 👤 Author

**Vihaan Kulkarni** — Bioinformatics & Machine Learning Engineer

---

## 📄 License

MIT License — Free for academic and research use.

---

**Last Updated:** December 17, 2025

**Status:** ✅ Active Development

### Phase 6: Documentation
1. Fill out `README.md` with:
   - Problem statement
   - Key insights (with screenshots)
   - Model metrics
   - Deployment link
2. Use "Problem → Method → Insight → Impact" structure

---

## 📦 Standard Dependencies

Every project includes:
- **Data:** `pandas`, `numpy`
- **Visualization:** `plotly`, `kaleido`
- **Modeling:** `scikit-learn`
- **Explainability:** `shap`
- **Deployment:** `streamlit`

Optional (uncomment in `requirements.txt` if needed):
- **Experiment Tracking:** `mlflow`, `wandb`
- **Deep Learning:** `torch`, `tensorflow`

---

## 💡 Pro Tips

1. **Run baseline first:** Always compare against a simple model
2. **Plotly over Matplotlib:** Interactive charts reveal more insights
3. **Document as you go:** Fill README during the project, not after
4. **Save figures:** Use `fig.write_html()` to preserve interactivity
5. **Version control:** Commit after each major milestone

---

## 🎓 Learning Resources

- [Plotly Documentation](https://plotly.com/python/)
- [SHAP Tutorial](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)

---

## 📊 Portfolio Goals

- ✅ 1 high-quality project per week
- ✅ Every project deployed with Streamlit
- ✅ README formatted for resume/GitHub
- ✅ Interactive visualizations (no static PNGs)
- ✅ Model explainability included

---

**Built by Vihaan Kulkarni**  
*Senior ML Engineer & Data Storyteller*
