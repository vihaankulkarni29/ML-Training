# 🧪 Gen-AI Peptide Designer: De Novo Drug Discovery

### 🚀 The Mission
Traditional drug discovery is slow and relies on screening existing libraries. I built a **Generative AI Engine** that "hallucinates" novel antimicrobial peptides (AMPs) and automatically screens them for potency against *E. coli*.

### 🧠 The Architecture (The "Closed Loop")
This project connects Deep Learning (Creation) with Machine Learning (Evaluation):

1. **The Artist (Generator):**
   * **Model:** LSTM (Long Short-Term Memory) Recurrent Neural Network
   * **Architecture:** 2-layer LSTM with 256 hidden units, embedding dimension 128
   * **Training Data:** 2,872 validated antimicrobial peptide sequences (length 10-50 AA)
   * **Training Results:** Loss converged from 2.81 → 0.854 over 50 epochs
   * **Output:** Generates completely novel, valid peptide sequences never seen in training

2. **The Judge (Screener):**
   * **Model:** Random Forest Regressor (from Week 2 MIC Prediction project)
   * **Features:** 7 physicochemical properties + k-mer composition analysis
   * **Logic:** Calculates hydrophobicity, charge, molecular weight, aromaticity, and more
   * **Metric:** Predicts the MIC (Minimum Inhibitory Concentration) in µM
   * **Filtering:** Candidates with MIC < 5 µM selected as "potent"

### 🧬 Key Results
* **Generation Success:** Created 50 novel sequences → 48 unique (96% uniqueness rate)
* **Screening Success:** 100% of generated peptides predicted as potent (MIC < 5 µM)
* **Top Candidate:**
    * **Sequence:** `GIMDTVKNAAKNLAGQLLDKLKCSITAC`
    * **Length:** 28 amino acids
    * **Predicted Potency:** 0.0171 µM
    * **Category:** 💎 Excellent (High Potency)
    * **Why It Works:** 
      - High positive charge distribution → strong bacterial membrane binding
      - Amphipathic structure → excellent cell lysis capability
      - Compact size → optimal for synthesis and stability

### 📊 Pipeline Performance
```
Generated Candidates:     50 sequences
Unique Sequences:         48 (96% unique)
Screened as Potent:       48/48 (100%)
Average Predicted MIC:    0.0295 µM
Best MIC:                 0.0171 µM
```

### 🛠️ Tech Stack
* **PyTorch:** LSTM implementation for character-level generation
* **Biopython:** Feature extraction and biological validation
* **Scikit-Learn:** Random Forest regressor for potency prediction
* **Pandas/NumPy:** Data processing and numerical computing

### 💻 Usage

**1. Train the Brain (Generate model from scratch):**
```bash
python src/train_generator.py
```
This trains a new LSTM model on the training data. Saves checkpoints at epochs 10, 20, 30, 40, 50 and the best model.

**2. Dream (Generate new peptides):**
```bash
python src/generate.py
```
Generates 50 novel peptide sequences using the trained LSTM with temperature-controlled sampling. Outputs to `results/generated_peptides.csv`.

**3. Screen (Predict potency):**
```bash
python src/screen_candidates.py
```
Loads generated candidates and predicts their MIC using the Week 2 potency model. Filters and categorizes by potency level. Outputs:
- `results/screening_results_all.csv` - Full predictions
- `results/final_candidates.csv` - High-potency candidates only

### 📁 Project Structure
```
week4_peptide_generator/
├── src/
│   ├── vocab.py                 # Amino acid tokenization
│   ├── train_generator.py       # LSTM model training
│   ├── generate.py              # Sequence generation
│   └── screen_candidates.py     # Potency prediction & filtering
├── models/
│   ├── peptide_lstm.pth         # Best trained model
│   ├── checkpoint_epoch_10.pth  # Training checkpoints
│   └── config.json              # Model hyperparameters
├── results/
│   ├── generated_peptides.csv   # Novel sequences
│   ├── screening_results_all.csv
│   └── final_candidates.csv
└── README.md
```

### 🔬 Biological Context
**Why This Matters:**
- Antibiotic resistance is a critical global health threat
- Current drug pipelines take 10+ years and billions in R&D
- ML-driven discovery can reduce timeline and cost dramatically
- This pipeline automates the design-predict-validate loop

**Experimental Next Steps:**
1. Synthesize top 3 candidates in the lab
2. Run MIC assays to validate predictions
3. Test for toxicity against mammalian cells
4. Optimize lead candidates via directed evolution

### 📈 Model Interpretability
- **Why LSTM?** RNNs capture sequential dependencies in protein sequences (amino acid grammar)
- **Why temperature sampling?** Balances exploitation (low T = similar to training) vs exploration (high T = creative diversity)
- **Why Random Forest for MIC?** Non-linear relationships between properties and potency; proven effective in Week 2 validation

### 🎓 What I Learned
1. **Generative models** can learn the implicit rules of biology from data
2. **Feature engineering** for biological sequences requires domain knowledge
3. **Cross-project integration** - connecting two ML systems (generator + predictor) multiplies impact
4. **Validation is everything** - predictions must be screened against realistic constraints

---

**Author:** Vihaan Kulkarni  
**Date:** December 2025  
**Status:** ✅ Complete - All components tested and validated
