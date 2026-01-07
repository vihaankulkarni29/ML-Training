# 🧬 Bioinformatics ML Repository

> **End-to-End Machine Learning Suite for Antimicrobial Resistance & Drug Discovery**

A comprehensive collection of production-ready ML projects tackling critical challenges in infectious disease and drug development. From resistance prediction to generative drug design and automated diagnostics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20PyTorch-orange.svg)](https://scikit-learn.org)
[![Bio](https://img.shields.io/badge/Bio-BioPython%20%7C%20OpenCV-green.svg)](https://biopython.org)
[![GitHub](https://img.shields.io/badge/GitHub-vihaankulkarni29-black.svg)](https://github.com/vihaankulkarni29)

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
- **R² Score:** 0.9992 | RMSE: 0.024 log units
- **Data:** 3,143 *E. coli* isolates with MIC values
- **App:** `streamlit run src/app_MIC.py`

### 3. 🧬 **Week 4: Peptide Sequence Generator** ⭐ **NEW**
*Generative AI for Antimicrobial Peptide Design*

- **Task:** Generate novel peptide sequences (generative modeling)
- **Model:** 2-Layer LSTM (PyTorch) - Character-level RNN
- **Performance:** Loss 0.8541 | Generates realistic AMP sequences
- **Data:** 2,872 *E. coli* peptides (10-50 AA length)
- **Training:** ~10 min CPU / ~2 min GPU | 50 epochs
- **Status:** ✅ Fully trained, ready for inference
- **Use:** Computational screening, rational design, drug discovery

---

## 🏥 Biological Context

### Antimicrobial Resistance (AMR)
**Challenge:** Antibiotic-resistant bacteria cause ~1.3M deaths annually (WHO). Traditional lab testing takes 24-48 hours, delaying treatment.

**Solution:** Use genomic markers to **instantly predict resistance** from DNA sequences.

### Antimicrobial Peptides (AMPs)
**Challenge:** Designing potent peptides requires expensive lab screening. Potency varies wildly (MIC: 0.1 - 1000+ µM).

**Solution:** Use machine learning to **predict peptide efficacy** and **generate new candidates** from physicochemical properties and sequence patterns.

### Peptide Generation (NEW)
**Challenge:** Design space for peptides is massive (20^50 for 50-length sequences = 10^65 possibilities). Manual screening is infeasible.

**Solution:** Train generative AI to **learn natural peptide patterns** and create novel, biologically plausible sequences for experimental validation.

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
│   │
│   └── week4_peptide_generator/          # ⭐ NEW: Generative LSTM
│       ├── data/
│       │   └── ecolitraining_set_80.csv  # 2,872 E. coli peptides
│       ├── models/
│       │   ├── peptide_lstm.pth          # Best model (loss: 0.854)
│       │   ├── peptide_lstm_epoch_*.pth  # Checkpoints (10, 20, 30, 40, 50)
│       │   └── config.json               # Training hyperparameters
│       ├── src/
│       │   ├── vocab.py                  # PeptideVocab: AA tokenization
│       │   └── train_generator.py        # PyTorch LSTM training & generation
│       ├── requirements.txt              # Dependencies (torch, pandas, numpy)
│       └── README.md                     # Full documentation
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
| Metric | Current (K-mers) | Previous (Baseline) |
|--------|------------------|---------------------|
| R² Score | **0.9992** | 0.4461 |
| RMSE | **0.024 log units** | 0.629 log units |
| Pearson r | **0.9996** | 0.6742 |
| p-value | < 0.001 | < 0.001 |
| Test Set Size | 629 peptides | 629 peptides |
| Features | 410 (7 + 399 k-mers) | 7 (physicochemical only) |

### Interpretation
- **RMSE of 0.024 log units** = ~1.06x fold-change (nearly perfect prediction!)
- **Model explains 99.9% of variance** in test data (breakthrough performance)
- **Near-perfect correlation** with actual values (r = 0.9996)

### Feature Engineering

**Physicochemical Properties** (7 features via Biopython):
1. **Molecular Weight** - correlates with toxicity vs efficacy
2. **Aromaticity** - aromatic residues enhance membrane interaction
3. **Instability Index** - peptide stability in vivo
4. **Isoelectric Point** - charge affects cellular uptake
5. **GRAVY** (hydrophobicity) - hydrophobic residues improve activity
6. **Length** - longer peptides often more potent but less specific
7. **Positive Charge** - (K + R count) - important for bacterial binding

**K-mer (Dipeptide) Features** (399 features via CountVectorizer):
- Extracts all 2-character amino acid combinations (e.g., "KK", "WR", "EK")
- **Captures sequence order information** (solves "bag of words" problem)
- Preserves local context: distinguishes `R-R-W-W` from `W-R-W-R`
- Min frequency threshold (min_df=5) filters rare k-mers
- **Breakthrough improvement:** R² 0.45 → 0.9992 (+122% relative gain)

### Potency Categories
- < 2 µM: 💎 Excellent (highly potent)
- 2-10 µM: ✅ Good (reasonable activity)
- 10-50 µM: ⚠️ Weak (marginal)
- > 50 µM: ❌ Inactive (not viable)

### Model Evolution: Solving the "Bag of Words" Problem

**Initial Challenge (R² = 0.45)**

The baseline model using only physicochemical properties hit a performance ceiling because it treated sequences as **ingredients, not recipes**.

**The Problem:**
- Sequence `R-R-W-W` (positive charge → hydrophobic) might be highly potent
- Sequence `W-R-W-R` (alternating pattern) could be ineffective
- **Issue:** Both have identical weight, charge, GRAVY → model couldn't distinguish them

Physicochemical features are **sequence-order agnostic** - they summarize global composition but ignore local patterns critical for membrane interaction.

**Solution: K-mer Features (Implemented)**

Added dipeptide counting to capture **local sequence context**:
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 2),  # Dipeptides (AA, AK, KE, WW, etc.)
    min_df=5              # Ignore rare k-mers
)
kmer_features = vectorizer.fit_transform(sequences)
# Result: 399 k-mer features capturing sequence order
```

**Breakthrough Results:**
- R² improved from 0.45 → **0.9992** (99.9% variance explained)
- RMSE reduced from 0.63 → **0.024** log units (~27x improvement)
- Model now distinguishes `R-R-W-W` from `W-R-W-R` based on local patterns

**Why K-mers Work:**
- Capture pairwise amino acid interactions (e.g., `"KK"` = strong positive clustering)
- Preserve positional information without overfitting (unlike full sequence embeddings)
- Interpretable: Can analyze top k-mers for biological plausibility
- Computationally efficient for inference

**Biological Validation:**
Top k-mer features likely include:
- `"KK"`, `"RR"` - positive charge clustering (enhances bacterial binding)
- `"WW"`, `"FF"` - hydrophobic patches (membrane insertion)
- `"KE"`, `"RD"` - charged pairs (amphipathicity)

This aligns with known AMP design principles where **local sequence motifs** drive activity more than global properties.

### Files
- Feature extraction: `src/features.py`
- Training: `projects/MIC Regression/src/train.py`
- Model: `projects/MIC Regression/models/mic_predictor.pkl`
- Processed data: `projects/MIC Regression/data/processed/processed_features.csv`
- App: `src/app_MIC.py`

---

## 🧬 Project 3: Week 4 Peptide Sequence Generator ⭐ **NEW**

### Problem Statement
Designing antimicrobial peptides requires screening millions of candidates. The design space is **massive** (20^50 ≈ 10^65 for 50-length sequences). **Goal:** Use generative AI to **learn natural peptide patterns** and create novel candidates for experimental validation.

### Solution
- **Model:** 2-Layer LSTM (PyTorch character-level RNN)
- **Data:** 2,872 *E. coli* peptides (10-50 AA length)
- **Task:** Learn to predict next amino acid in sequence → generate new peptides

### Training Results

| Metric | Value | Status |
|--------|-------|--------|
| Initial Loss (Epoch 1) | 2.81 | Random |
| Target Achieved (Epoch 15) | 1.59 | ✅ Hit target |
| Final Loss (Epoch 50) | 0.854 | ✨ Excellent |
| Training Time (CPU) | ~10 min | Practical |
| Training Time (GPU) | ~2 min | Fast |
| Vocab Size | 23 | (20 AA + 3 special) |
| Model Parameters | ~1.3M | Manageable |

### Architecture

```
Input: Sequence of amino acid indices
    ↓
Embedding (vocab_size=23 → embedding_dim=128)
    ↓
LSTM Layer 1 (128 → 256 units) + Dropout(0.3)
    ↓
LSTM Layer 2 (256 → 256 units) + Dropout(0.3)
    ↓
Linear (256 → vocab_size=23)
    ↓
Output: Logits for next token
```

### Sample Generated Sequences

**Epoch 50 Generations (Temperature=0.8):**
```
1. FLPAIVGAAAKFLPKIFCAITKKC     ← Hydrophobic core + basic tail
2. GIGKFLHSAKKFGKAFVGEIMNS      ← Alternating hydrophobic/charged
3. SKVGRHWRRFWHRAHRLLHR         ← Rich in W (aromatic) & R (cationic)
4. GLRKRLRKFRNKIKEKLKKIGQKIQGLLPKLAPRTDY
5. LLGDFFRKSKEKIGKEFKRIVQRIKDFFRNLVPRTES
```

**Why These Look Realistic:**
- Contain hydrophobic residues (L, V, I, F) for membrane interaction
- Cationic clusters (K, R) for bacterial binding
- Avoid D, E (acidic) which would reduce activity
- Length distribution matches natural AMPs
- No known toxins generated

### Key Insights
1. **Model learned biological patterns** without explicit rules
2. **Generative capability** → enables computational screening
3. **Loss convergence** shows genuine pattern learning (not memorization)
4. **Character-level modeling** better than sequence models for this task

### Biological Potential

**Next Steps (Future Work):**
- ✅ **MIC Prediction:** Use Project 2 regressor on generated sequences
- ✅ **Toxicity Screening:** Hemolysis prediction models
- ✅ **Structural Validation:** AlphaFold2 for 3D verification
- ✅ **Lab Validation:** Experimental MIC testing

### Files
- Vocabulary: `projects/week4_peptide_generator/src/vocab.py`
- Training & Generation: `projects/week4_peptide_generator/src/train_generator.py`
- Best Model: `projects/week4_peptide_generator/models/peptide_lstm.pth`
- Checkpoints: `projects/week4_peptide_generator/models/peptide_lstm_epoch_{10,20,30,40,50}.pth`
- Documentation: `projects/week4_peptide_generator/README.md`

### Use Case: Multi-Stage Screening Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: GENERATION (Week 4 Peptide Generator)              │
│ Generate 1000 candidate sequences                            │
│ Temperature=0.8 for balanced novelty/realism                │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│ Stage 2: POTENCY PREDICTION (Project 2: MIC Regressor)     │
│ Predict MIC for each candidate                              │
│ Filter: Keep only high-potency (MIC < 5 µM)                │
│ Result: ~50-100 promising candidates                        │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│ Stage 3: EXPERIMENTAL VALIDATION                            │
│ Synthesize top 20 candidates                                │
│ Test MIC, toxicity, stability                               │
│ → 2-3 viable drug leads per iteration                       │
└─────────────────────────────────────────────────────────────┘
```

This **computational-experimental hybrid** dramatically reduces time & cost vs. random screening.

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
