# 🛡️ Ceftriaxone Resistance Predictor

> **AI-Powered Antibiotic Resistance Prediction from Genomic Data**

**Status:** ✅ **COMPLETE** | Production-Ready Streamlit Deployment

---

## 🏥 The Problem

**Antimicrobial Resistance (AMR)** is a **top 10 global health threat** according to the WHO.

### Current Clinical Challenge
Traditional antibiotic susceptibility testing takes **24-48 hours**:
- Patient with sepsis arrives → Blood culture sent to lab
- Wait 1-2 days → Results returned (often too late)
- Empiric antibiotics given (may be ineffective for resistant strains)
- **Outcome:** Delayed appropriate therapy, increased mortality

### The Opportunity
**Genomic data is available in real-time** from clinical isolates. Machine Learning can instantly predict resistance from genetic markers.

---

## 💡 The Solution

A **Random Forest classifier** that predicts **Ceftriaxone resistance** in *E. coli* from genomic data in **milliseconds**.

**Key Innovation:** The model independently "rediscovered" known resistance mechanisms:
- **blaCTX-M-15** (Extended-Spectrum Beta-Lactamase)
- **blaCMY-2** (AmpC Cephalosporinase)
- **gyrA/parC** (Fluoroquinolone resistance markers)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Sensitivity (Recall)** | 93.9% |
| **Specificity** | 95.9% |
| **Accuracy** | 94.9% |
| **ROC-AUC** | 0.978 |
| **Training Dataset** | 4,383 *E. coli* isolates (NCBI) |
| **Geographic Coverage** | USA, Europe |

### Clinical Significance
- **93.9% Sensitivity:** Catches resistant cases (minimizes false negatives—clinically critical)
- **95.9% Specificity:** Correctly identifies susceptible cases
- **Purpose:** Risk-stratification tool, not replacement for lab testing

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation & Deployment

#### Local Development
```bash
# Clone repository
git clone https://github.com/vihaankulkarni29/ML-Training
cd ML

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

#### Deploy to Streamlit Cloud
1. Push repo to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub repo
4. Deploy with one click
5. Access app at `https://<username>-<appname>.streamlit.app`

---

## 📁 Project Structure

```

---

## 🧬 How to Use the App

### 1. Select Genes
Use the sidebar to choose genes detected in your isolate's genotype.

**Quick Presets:**
- **ESBL Genes:** blaCTX-M-15, blaCTX-M-1, blaTEM-1, blaSHV-1
- **Fluoroquinolone:** gyrA_S83L, parC
- **AmpC:** blaCMY-2, blaDHA-1

### 2. Run Analysis
Click "🚀 Analyze Sample" to generate prediction.

### 3. Interpret Results
- **Resistance Probability:** Confidence of resistance prediction
- **Top Drivers:** Feature importance visualization
- **Clinical Notes:** Recommendations for next steps

### Example
```
Input:  Select blaCTX-M-15, blaTEM-1
Output: ⚠️ RESISTANT (95.2% confidence)
        → Consider alternative antibiotic
        → Confirm with lab testing
```

---

## 🔬 Technical Details

### Model Architecture
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - `n_estimators=100` trees
  - `max_depth=10` (prevent overfitting)
  - `class_weight='balanced'` (handle imbalanced dataset)
  - `min_samples_split=5`
  - `n_jobs=-1` (parallel processing)

### Training Data
- **Source:** NCBI MicroBIGG-E Isolates Browser
- **Organism:** *Escherichia coli*
- **Antibiotic:** Ceftriaxone (3rd-gen cephalosporin)
- **Size:** 4,383 isolates
- **Features:** 352 detected genes/mutations
- **Classes:** Susceptible (51.8%) | Resistant (48.2%)

### Validation Strategy
- **Cross-Validation:** 5-Fold Stratified K-Fold
- **Test Set:** 20% held-out (876 isolates)
- **Metrics:** Sensitivity, Specificity, ROC-AUC (clinical focus)

---

## ⚠️ Disclaimer

**This tool is for research and educational purposes only.**

### Not Suitable For:
- Direct clinical diagnosis without laboratory confirmation
- Replacement of culture-based susceptibility testing
- Use in patient care without MD oversight

### Required Validation:
- Confirm all predictions with lab culture + susceptibility testing (EUCAST/CLSI)
- Consider local resistance epidemiology
- Consult clinical microbiology before treatment decisions

---

## 📚 Biological Background

### Why Predict Resistance from Genes?

**Genetic Basis of Resistance:**
1. **Beta-Lactamase Production:** blaCTX-M, blaTEM, blaCMY genes encode enzymes that hydrolyze beta-lactam antibiotics
2. **Target Modification:** gyrA, parC mutations reduce fluoroquinolone binding
3. **Efflux Pumps:** AcrB overexpression mechanisms

**Advantage:** Genes = DNA fingerprint, available before culture results.

### Known AMR Genes (Top Predictors)
- **blaCTX-M-15:** Most common ESBL globally; strong predictor of cephalosporin resistance
- **blaCMY-2:** AmpC cephalosporinase; inducible resistance
- **gyrA/parC:** Co-selected with beta-lactam resistance in *E. coli*
- **blaTEM-1:** Historic beta-lactamase; often co-occurs with CTX-M

---

## 🛠️ Development & Deployment

### Built With
- **Data Science:** Python, Pandas, Scikit-Learn
- **Visualization:** Matplotlib, Plotly
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud, GitHub

### Model Training
```bash
cd projects/cefixime-resistance-training
python src/train.py
```
Outputs: `models/ceftriaxone_model.pkl`, confusion matrix, feature importance

### Artifact Files Generated
- `models/ceftriaxone_model.pkl` — Trained RandomForest
- `results/confusion_matrix.html` — Plotly confusion matrix
- `results/feature_importance_*.csv` — Top 20 genes
- `results/confusion_matrix_utils.html` — Utils evaluation CM

---

## 📞 Questions & Support

### Model Questions
- **How accurate is this?** 93.9% sensitivity on test set; validate on your local data
- **What if I have missing genes?** Model handles absent genes; pad with zeros
- **Can I use this for other pathogens?** This model is *E. coli*-specific; retraining needed for others

### Technical Issues
- Model not loading? Verify `models/ceftriaxone_model.pkl` exists
- Missing genes in sidebar? Check if model features match your data
- Streamlit error? Reinstall: `pip install --upgrade streamlit`

---

## 📖 Citation & Attribution

**Data Source:**
- NCBI MicroBIGG-E: https://microbiggdata.ncbi.nlm.nih.gov/
- EUCAST Antibiotic Susceptibility Testing Guidelines

**Model Published:**
- Algorithm: Scikit-Learn RandomForestClassifier
- Deployment: Streamlit (open-source)

**Suggested Citation:**
```
Ceftriaxone Resistance Predictor v1.0
Machine Learning model for E. coli resistance prediction
Trained on NCBI isolates, deployed via Streamlit Cloud
```

---

## 🎯 Future Enhancements

- [ ] **Multi-organism support:** Expand to Klebsiella, Pseudomonas
- [ ] **Multi-antibiotic:** Predict resistance to 5+ beta-lactams
- [ ] **SHAP explainability:** Individual prediction breakdowns
- [ ] **Uncertainty quantification:** Confidence intervals for predictions
- [ ] **REST API:** For integration with clinical LIS systems
- [ ] **Mobile app:** iOS/Android version for field deployment

---

## 📄 License

MIT License — Free for research and educational use.

---

**Last Updated:** December 14, 2025

**Status:** ✅ Production Ready

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
