# Week 4: Peptide Generator 🧬

An LSTM-based deep learning model to generate novel peptide sequences based on E. coli training data.

## 🎯 Project Overview

This project uses a character-level LSTM (Long Short-Term Memory) neural network to learn patterns from existing peptide sequences and generate new, biologically plausible peptide sequences.

## 📁 Project Structure

```
week4_peptide_generator/
├── data/
│   └── ecolitraining_set_80.csv    # Training dataset from Week 2
├── models/
│   ├── peptide_lstm.keras          # Trained LSTM model
│   ├── training_history.png        # Training/validation curves
│   └── config.json                 # Model configuration
├── src/
│   ├── vocab.py                    # Amino acid vocabulary & tokenization
│   └── train_generator.py          # LSTM model & training pipeline
├── requirements.txt
└── README.md
```

## 🧠 How It Works

### 1. **Vocabulary (vocab.py)**
Converts amino acid sequences into numerical representations:
- **20 Standard Amino Acids**: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- **Special Tokens**: 
  - `<PAD>`: Padding for sequences of different lengths
  - `<START>`: Beginning of sequence
  - `<END>`: End of sequence
  - `<UNK>`: Unknown amino acid

### 2. **LSTM Generator (train_generator.py)**
A deep learning model that learns peptide sequence patterns:
- **Embedding Layer**: Maps amino acids to dense vectors
- **2 LSTM Layers**: Each with 256 units to capture sequential patterns
- **Dropout Layers**: Prevent overfitting (30% dropout rate)
- **Output Layer**: Predicts next amino acid in sequence

### 3. **Generation Process**
1. Model is trained on existing E. coli peptide sequences
2. Learns probability distributions for amino acid transitions
3. Generates new sequences by sampling from learned distributions
4. Uses temperature parameter to control randomness

## 🚀 Getting Started

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training the Model

1. **Navigate to the src directory:**
```bash
cd src
```

2. **Run the training script:**
```bash
python train_generator.py
```

### What Happens During Training

1. ✅ Loads peptide sequences from `data/ecolitraining_set_80.csv`
2. ✅ Converts sequences to numerical format using vocabulary
3. ✅ Splits data into training (80%) and validation (20%)
4. ✅ Trains LSTM model for up to 100 epochs
5. ✅ Saves best model based on validation loss
6. ✅ Generates 10 sample peptide sequences
7. ✅ Saves training history plots and configuration

## 📊 Model Architecture

```
Input: Sequence of amino acid indices
  ↓
Embedding Layer (128 dimensions)
  ↓
LSTM Layer 1 (256 units) + Dropout (0.3)
  ↓
LSTM Layer 2 (256 units) + Dropout (0.3)
  ↓
Dense Layer (Softmax activation)
  ↓
Output: Probability distribution over amino acids
```

## 🎛️ Configuration

Key parameters in `train_generator.py`:

```python
EMBEDDING_DIM = 128      # Dimension of amino acid embeddings
LSTM_UNITS = 256         # Number of units in each LSTM layer
BATCH_SIZE = 64          # Samples per training batch
EPOCHS = 100             # Maximum training epochs
VALIDATION_SPLIT = 0.2   # 20% data for validation
```

## 📈 Training Callbacks

- **ModelCheckpoint**: Saves best model during training
- **EarlyStopping**: Stops training if no improvement for 15 epochs
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

## 🔬 Generating New Sequences

After training, the model generates sequences by:

1. Starting with `<START>` token
2. Predicting next amino acid based on current sequence
3. Sampling from probability distribution (controlled by temperature)
4. Repeating until `<END>` token or max length reached

**Temperature parameter:**
- `< 1.0`: More conservative, follows training data closely
- `= 1.0`: Balanced sampling
- `> 1.0`: More creative, explores diverse sequences

## 📊 Example Output

```
Sample 1: MKLVNRASPL
Sample 2: ACDEFGHIKL
Sample 3: TGKPLMNQRS
...
```

## 🔍 Model Evaluation

Training generates:
- **Training History Plot**: Shows loss and accuracy curves
- **Config File**: Records model parameters and training date
- **Sample Sequences**: 10 generated peptides for quick validation

## 💡 Use Cases

- **Drug Discovery**: Generate novel therapeutic peptides
- **Protein Engineering**: Design new protein sequences
- **Biotechnology**: Create peptides with specific properties
- **Research**: Study sequence-function relationships

## 🛠️ Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce `BATCH_SIZE` or `LSTM_UNITS`

**Issue**: Model not converging
- **Solution**: Increase training data or adjust learning rate

**Issue**: Generated sequences too similar
- **Solution**: Increase temperature parameter (try 1.2-1.5)

## 📚 Dependencies

- **TensorFlow**: Deep learning framework for LSTM
- **NumPy**: Numerical operations
- **Pandas**: Data loading and manipulation
- **Matplotlib**: Visualization of training progress

## 🎓 Learning Objectives

- ✅ Understand sequence-to-sequence learning
- ✅ Implement LSTM for text generation
- ✅ Work with biological sequence data
- ✅ Apply deep learning to bioinformatics

## 🔮 Future Enhancements

- [ ] Add attention mechanism for longer sequences
- [ ] Implement variational autoencoder (VAE) for controlled generation
- [ ] Add sequence property prediction (e.g., antimicrobial activity)
- [ ] Fine-tune on specific peptide families
- [ ] Create web interface for interactive generation

## 📝 Notes

- Model training time depends on dataset size and hardware
- GPU recommended for faster training (CPU works but slower)
- Generated sequences should be validated for biological relevance

## 🤝 Contributing

Feel free to experiment with:
- Different model architectures
- Alternative sequence representations
- Various generation strategies
- Additional training data

---

**Created**: December 2025  
**Week**: 4 - Deep Learning for Sequence Generation  
**Dataset**: E. coli Peptide Training Set (Week 2)
