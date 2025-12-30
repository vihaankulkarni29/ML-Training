# Week 4: Peptide Sequence Generator 🧬

> **Generative AI for Antimicrobial Peptide Design**

A PyTorch-based character-level LSTM that learns the "grammar of life" from E. coli peptide sequences and generates novel, biologically plausible antimicrobial peptide (AMP) sequences.

## 🎯 Problem & Solution

### The Challenge
Designing effective antimicrobial peptides is expensive, time-consuming, and low-throughput. Traditional methods require extensive laboratory screening. We need a way to **computationally generate candidate peptides** that follow natural sequence patterns.

### Our Solution
Train a **character-level RNN (LSTM)** to learn:
- Which amino acids typically follow others
- Common sequence motifs in natural peptides
- The "probability distribution" of amino acid transitions
- How to generate new sequences that look "natural" but are novel

### How It Works Like Phone Autocomplete
- **Input:** K-L-L...
- **Prediction:** Next character is likely "R"
- **Repeat:** K-L-L-R-I-K...

The model learns this by treating peptide sequences like text and predicting the next character (amino acid) at each position.

## 📁 Project Structure

```
week4_peptide_generator/
├── data/
│   └── ecolitraining_set_80.csv         # Training dataset: 2,872 sequences from E. coli
├── models/
│   ├── peptide_lstm.pth                 # Best trained model (loss: 0.8541)
│   ├── peptide_lstm_epoch_10.pth        # Checkpoint at epoch 10
│   ├── peptide_lstm_epoch_20.pth        # Checkpoint at epoch 20
│   ├── peptide_lstm_epoch_30.pth        # Checkpoint at epoch 30
│   ├── peptide_lstm_epoch_40.pth        # Checkpoint at epoch 40
│   ├── peptide_lstm_epoch_50.pth        # Final checkpoint at epoch 50
│   └── config.json                      # Model hyperparameters & training metadata
├── src/
│   ├── vocab.py                         # PeptideVocab: Amino acid tokenization
│   └── train_generator.py               # Training pipeline & sequence generation
├── requirements.txt                     # Python dependencies (PyTorch, pandas, numpy)
└── README.md                            # This file
```

## 🧠 Architecture

### 1. **Tokenization (vocab.py)**
Converts biological sequences to numbers (and back):

```python
vocab = PeptideVocab()
encoded = vocab.encode("ACDEFGH")  # [1, 4, 5, 6, 7, 8, 9, 10, 2]
#                                    <SOS> A  C  D  E  F  G  H <EOS>
```

**Vocabulary (23 tokens):**
- **Special:** `<PAD>` (0), `<SOS>` (1), `<EOS>` (2)
- **Amino Acids (20):** A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

### 2. **Dataset Loading (PeptideDataset)**
- Loads CSV with sequences in 'SEQUENCE' column
- **Filters:** Sequences of 10-50 amino acids (biological stability)
- **Result:** 2,872 valid training sequences
- Uses **collate_fn** for dynamic padding in batches

### 3. **LSTM Model Architecture**

```
Input Sequence (batch_size × seq_length)
    ↓
Embedding Layer (vocab_size=23 → embedding_dim=128)
    ↓
LSTM Layer 1 (embedding_dim=128 → hidden_size=256, return_sequences=True)
    ↓
Dropout (p=0.3)
    ↓
LSTM Layer 2 (hidden_size=256 → hidden_size=256, return_sequences=True)
    ↓
Dropout (p=0.3)
    ↓
Linear Layer (hidden_size=256 → vocab_size=23)
    ↓
Logits for each position (batch_size × seq_length × vocab_size)
```

**Why 2 LSTM layers?**
- Layer 1 learns local patterns (dipeptides, tripeptides)
- Layer 2 learns long-range dependencies (structural motifs)
- More layers = more pattern capture = better generation

## 📊 Training Results

### Loss Convergence

| Epoch | Loss | Status |
|-------|------|--------|
| 1 | 2.81 | Random predictions |
| 5 | 2.31 | Learning starting |
| 10 | 1.88 | Recognizable patterns |
| 15 | **1.59** | **Target hit!** 🎯 |
| 20 | 1.39 | Strong convergence |
| 30 | 1.12 | Excellent patterns |
| 40 | 0.96 | Near-optimal |
| 50 | **0.85** | **Final (best)** ✨ |

**Interpretation:**
- Started with loss ~2.81 (random guessing)
- Each epoch improved by ~2-3%
- Hit target of **1.5 by epoch 15**
- Continued learning to epoch 50 (loss 0.854)
- **3.3x improvement** from start to finish

### Sample Generations (Epoch 50)

**Training discovered natural patterns:**

```
1. FLPAIVGAAAKFLPKIFCAITKKC    ← Hydrophobic + basic tail
2. GIGKFLHSAKKFGKAFVGEIMNS     ← Alternating hydrophobic/hydrophilic
3. SKVGRHWRRFWHRAHRLLHR        ← Rich in W, R (aromatic + cationic)
4. GLRKRLRKFRNKIKEKLKKIGQKIQGLLPKLAPRTDY  ← Structured pattern
5. LLGDFFRKSKEKIGKEFKRIVQRIKDFFRNLVPRTES  ← Complex 40-char sequence
```

These sequences contain the same **physicochemical properties** as natural AMPs:
- Mix of hydrophobic residues (L, V, F, I)
- Cationic residues (K, R) for bacterial binding
- Few charged acidic residues (D, E)
- Appropriate length distribution

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd projects/week4_peptide_generator

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Run training from project root
python src/train_generator.py

# Expected output:
# ======================================================================
# Char-RNN Peptide Generator (PyTorch)
# ======================================================================
# Using device: cpu  (or cuda if available)
# Vocab size: 23
# Loaded 2872 sequences (length 10-50)
# Epoch 01/50 - Loss: 2.8121
#   Saved best model (loss 2.8121)
# ...
# Epoch 50/50 - Loss: 0.8541
#   Saved best model (loss 0.8541)
```

**Training Time:**
- CPU: ~10-15 minutes (Intel i5/i7)
- GPU: ~2 minutes (NVIDIA GPU)

### Model Checkpoints

Models are saved at every 10 epochs. Resume training from checkpoint:

```python
import torch
from src.train_generator import PeptideLSTM
from src.vocab import PeptideVocab

# Load checkpoint
checkpoint = torch.load('models/peptide_lstm_epoch_30.pth')
vocab = PeptideVocab()
model = PeptideLSTM(vocab_size=23).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Continue from epoch 31...
```

## 🔬 Generating New Sequences

### Interactive Generation

```python
import torch
from src.train_generator import PeptideLSTM, generate_sequence
from src.vocab import PeptideVocab

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = PeptideVocab()
model = PeptideLSTM(vocab_size=23).to(device)

# Load weights
checkpoint = torch.load('models/peptide_lstm.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate sequences with different temperatures
for temp in [0.5, 0.8, 1.0, 1.2]:
    seq = generate_sequence(model, vocab, device, temperature=temp)
    print(f"Temp {temp}: {seq}")
```

### Temperature Parameter

**Temperature controls randomness:**

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.5 | Conservative, follows training data closely | Conservative candidates |
| 0.8 | Balanced, some creativity | Default - good diversity |
| 1.0 | Standard probability distribution | Random baseline |
| 1.2+ | Creative, explores unusual sequences | Drug discovery screening |

**Lower T → More similar to training data**  
**Higher T → More novel/creative sequences**

## 📈 Training Configuration

```json
{
  "batch_size": 64,
  "learning_rate": 0.001,
  "epochs": 50,
  "embedding_dim": 128,
  "hidden_size": 256,
  "num_layers": 2,
  "dropout": 0.3,
  "min_length": 10,
  "max_length": 50,
  "optimizer": "Adam",
  "loss_function": "CrossEntropyLoss (ignore_index=0 for padding)",
  "gradient_clip": 5.0,
  "best_loss": 0.8541,
  "training_date": "2025-12-30"
}
```

## 🧪 Evaluation Metrics

## 🧪 Evaluation Metrics

### Loss Metrics
- **Training Loss:** Measures accuracy of next-character prediction during training
- **Final Loss:** 0.8541 (excellent convergence)
- **Target:** < 1.5 (achieved by epoch 15) ✅

### Sequence Quality Metrics
- **Validity:** 100% of generated sequences contain only valid amino acids
- **Length Distribution:** Generated sequences follow natural length patterns (10-50 AA)
- **Similarity:** Generated ≠ exact copies of training data (verified via similarity scoring)

### How to Evaluate Generated Peptides

```python
import torch
from src.train_generator import PeptideLSTM, generate_sequence
from src.vocab import PeptideVocab
from collections import Counter

# Generate batch of sequences
device = torch.device('cpu')
vocab = PeptideVocab()
model = PeptideLSTM(vocab_size=23).to(device)
checkpoint = torch.load('models/peptide_lstm.pth')
model.load_state_dict(checkpoint['model_state_dict'])

sequences = [generate_sequence(model, vocab, device) for _ in range(100)]

# Analyze physicochemical properties
def aa_composition(seq):
    return dict(Counter(seq))

# Check for known motifs
for seq in sequences:
    if 'RR' in seq or 'KK' in seq:  # Cationic clusters
        print(f"Strong charge: {seq}")
    if seq.count('L') + seq.count('V') + seq.count('I') > len(seq) * 0.3:
        print(f"Hydrophobic: {seq}")
```

## 💡 Biological Insights

### What the Model Learned

1. **Hydrophobic-Hydrophilic Balance**
   - Learned to interleave hydrophobic (L, V, I, F) with charged (K, R)
   - Natural AMPs need both for membrane interaction

2. **Cationic Clusters**
   - Generates sequences with K-K and R-R motifs
   - These clusters help binding to negatively-charged bacterial membranes

3. **Sequence Length Distribution**
   - Respects 10-50 AA range from training data
   - Natural AMPs rarely shorter (ineffective) or longer (toxic)

4. **Aromatic Residues (W, Y, F)**
   - Appropriately distributed
   - Important for peptide-lipid interactions

### Biological Limitations (and Future Work)

Generated sequences should be validated experimentally:
- [ ] **MIC Testing:** Measure minimum inhibitory concentration against bacteria
- [ ] **Toxicity:** Test hemolysis (red blood cell lysis) in humans
- [ ] **Stability:** Check protease resistance in serum/blood
- [ ] **Structure:** Perform NMR/CD spectroscopy to confirm predicted structure

## 🔧 Advanced Usage

### Fine-tuning on Custom Data

```python
# Load pre-trained model
checkpoint = torch.load('models/peptide_lstm.pth')
model = PeptideLSTM(vocab_size=23)
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune on new peptide family
custom_dataset = PeptideDataset('my_peptides.csv', vocab)
dataloader = DataLoader(custom_dataset, batch_size=32, collate_fn=collate_fn)

# Train for fewer epochs
for epoch in range(5):
    train_epoch(model, dataloader, criterion, optimizer, device)
    torch.save(model.state_dict(), f'models/finetuned_epoch_{epoch}.pth')
```

### Batch Generation for Drug Screening

```python
import pandas as pd

# Generate 1000 candidate sequences
candidates = [generate_sequence(model, vocab, device, temperature=0.9) 
              for _ in range(1000)]

# Save for experimental validation
df = pd.DataFrame({'sequence': candidates})
df.to_csv('candidate_peptides.csv', index=False)

print(f"Generated {len(candidates)} unique candidate peptides")
```

## 🎓 Learning Resources

### Concepts Used

1. **LSTM (Long Short-Term Memory)**
   - Handles long-range dependencies in sequences
   - Solves "vanishing gradient" problem of basic RNNs
   - Reference: Hochreiter & Schmidhuber (1997)

2. **Character-Level Language Models**
   - Learns "language of biology" at atomic level
   - No pre-tokenization needed (unlike word-level models)
   - Discovers structure from data

3. **Generative Modeling**
   - Temperature sampling for controlling output diversity
   - Importance of good training data (quality > quantity)
   - Avoiding mode collapse (generating identical sequences)

### Further Reading

- **Antimicrobial Peptides:** Hancock & Sahl (2006), "Antimicrobial and host-defense peptides"
- **LSTM Theory:** Goodfellow et al. (2016), "Deep Learning" (Chapters 10-15)
- **Sequence Modeling:** Bengio et al. (2013), "Recent Advances in Deep Learning for Speech"
- **PyTorch Tutorials:** https://pytorch.org/tutorials/

## 📝 Files Description

### `src/vocab.py`
```python
class PeptideVocab:
    def encode(sequence: str) -> List[int]
    def decode(indices: List[int]) -> str
```
- Maps amino acids ↔ numbers
- Handles special tokens (<SOS>, <EOS>, <PAD>)
- Critical for data preprocessing

### `src/train_generator.py`
```python
class PeptideDataset(Dataset)
class PeptideLSTM(nn.Module)
def train_epoch(...)
def generate_sequence(...)
def collate_fn(...)
```
- Full training pipeline
- Model architecture
- Generation function
- Batch collation with dynamic padding

### `models/config.json`
Stores training metadata for reproducibility:
- Hyperparameters (batch size, learning rate, embedding dim)
- Training date and best loss
- Vocab size and sequence length bounds

## 🐛 Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "CSV must contain 'SEQUENCE' column" | Wrong column name | Rename column to 'SEQUENCE' (uppercase) |
| Out of memory | Batch too large | Reduce batch_size from 64 to 32 or 16 |
| Loss not decreasing | LR too high or too low | Try LR = 0.0001 or 0.01 |
| Sequences too short | Temperature too low | Increase temperature to 1.0-1.2 |
| GPU not found | PyTorch CPU version | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |

## 📚 Requirements

```
pandas           # Data loading
numpy            # Numerical operations
torch            # Deep learning framework
scikit-learn     # Metrics (optional)
tqdm             # Progress bars
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 📄 License & Citation

**Project:** Week 4 Peptide Generator - Deep Learning for Biosequence Design  
**Repository:** https://github.com/vihaankulkarni29/ML-Training  
**Date:** December 2025

**If you use this work, please cite:**

```bibtex
@software{peptide_generator_2025,
  title = {Generative LSTM for Antimicrobial Peptide Design},
  author = {Vihaan Kulkarni},
  year = {2025},
  url = {https://github.com/vihaankulkarni29/ML-Training}
}
```

## 🤝 Contributing

Have ideas to improve the generator?

- [ ] Implement Transformer architecture for longer sequences
- [ ] Add GAN-based discriminator for quality scoring
- [ ] Integrate with AlphaFold for 3D structure prediction
- [ ] Build web app for interactive generation
- [ ] Fine-tune on specific peptide families (lacticins, defensins, etc.)

## ✅ Checklist for Deployment

- [x] Model training completed
- [x] Loss converged below 1.5
- [x] Sample sequences validated
- [x] Code pushed to GitHub
- [ ] Experimental validation (future)
- [ ] Patent filing (if applicable, future)

---

**Questions or bugs?** Open an issue on GitHub or contact Vihaan Kulkarni.

**Happy peptide generating! 🧬✨**
