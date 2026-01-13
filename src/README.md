# DeepG2P - Deep Genotype-to-Phenotype Model

> **1D Convolutional Neural Network for Antimicrobial Resistance Prediction from Mass Spectrometry Data**

A ResNet-inspired deep learning architecture that predicts antimicrobial resistance patterns directly from MALDI-TOF mass spectrometry signals.

---

## 🏗️ Architecture

### **ResNet-1D with Residual Blocks**

```
Input (6000×1) 
    ↓
Conv1D (kernel=7, stride=2) → BatchNorm → ReLU → MaxPool
    ↓
[ResidualBlock × 2] (64 channels, stride=1)
    ↓
[ResidualBlock × 2] (128 channels, stride=2)
    ↓
[ResidualBlock × 2] (256 channels, stride=2)
    ↓
[ResidualBlock × 2] (512 channels, stride=2)
    ↓
Global Average Pooling → Dropout (0.5) → FC → Sigmoid
    ↓
Output (10 antibiotics)
```

### **Residual Block**
```
Input
  ├─→ Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm
  │                                                ↓
  └────────────────────────────────────────────→ Add → ReLU
                                                  ↓
                                               Output
```

---

## 📂 Files

### **model.py**
Core model implementation:
- `ResidualBlock`: Convolutional block with skip connections
- `DeepG2P`: Main ResNet-1D architecture
- `create_deepg2p_model()`: Factory function for different model sizes

**Key Features:**
- Flexible input dimensions (default: 6000×1)
- Multi-label classification (10 antibiotics)
- He/Xavier weight initialization
- Feature map extraction for interpretability

**Model Sizes:**
```python
# Small: 32 base channels, 2-2-2-2 blocks, 0.3 dropout
model = create_deepg2p_model(model_size='small')  # ~500K params

# Medium: 64 base channels, 2-2-2-2 blocks, 0.5 dropout
model = create_deepg2p_model(model_size='medium')  # ~2M params

# Large: 64 base channels, 3-4-6-3 blocks, 0.5 dropout
model = create_deepg2p_model(model_size='large')  # ~5M params
```

### **train.py**
Comprehensive training pipeline:

**Components:**
1. **DRIAMSDataset**: Custom PyTorch Dataset for .npy files
2. **Loss**: BCEWithLogitsLoss with automatic pos_weight (handles class imbalance)
3. **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
4. **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
5. **Metrics**: AUPRC, AUROC, Loss
6. **Logging**: TensorBoard + console output

**Features:**
- Automatic class imbalance handling
- Best model checkpointing (`models/best_model.pth`)
- Periodic checkpoints every 5 epochs
- Training configuration saved to JSON
- Progress bars with live metrics

---

## 🚀 Usage

### **Training**

```bash
# Basic training with default parameters
python src/train.py

# Custom training
python src/train.py \
  --train-features data/processed/X_train.npy \
  --train-labels data/processed/y_train.npy \
  --val-features data/processed/X_val.npy \
  --val-labels data/processed/y_val.npy \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --model-size medium \
  --num-antibiotics 10
```

### **Monitor Training**

```bash
# Launch TensorBoard
tensorboard --logdir results/logs

# View at http://localhost:6006
```

### **Model Inference**

```python
import torch
from model import create_deepg2p_model

# Load model
model = create_deepg2p_model(num_antibiotics=10)
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
x = torch.randn(1, 1, 6000)  # Single spectrum
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)

print(f"Resistance probabilities: {probs[0].numpy()}")
```

---

## 📊 Training Pipeline

### **Data Flow**
```
.npy files (X, y)
    ↓
DRIAMSDataset (PyTorch Dataset)
    ↓
DataLoader (batch_size=32, shuffle=True)
    ↓
DeepG2P Model
    ↓
BCEWithLogitsLoss (pos_weight for imbalance)
    ↓
AdamW Optimizer (lr=1e-4)
    ↓
Metrics: AUPRC, AUROC
    ↓
Best Model → models/best_model.pth
```

### **Class Imbalance Handling**

The training pipeline automatically calculates `pos_weight` for BCEWithLogitsLoss:

```python
pos_weight = (# negative samples) / (# positive samples)
```

This penalizes false negatives more heavily for rare resistance cases.

### **Expected Output**

```
======================================================================
DeepG2P Training Pipeline - Antimicrobial Resistance Prediction
======================================================================

🖥️  Device: cuda
   GPU: NVIDIA GeForce RTX 3080
   Memory: 10.00 GB

📂 Loading datasets...
   Training samples: 8000
   Validation samples: 2000

⚖️  Calculating class weights for imbalanced data...
   Class imbalance ratio: 15.23:1 (negative:positive)

🏗️  Building DeepG2P model (size: medium)...
   Total parameters: 2,147,850
   Trainable parameters: 2,147,850

🚀 Starting training for 20 epochs...
   Batch size: 32
   Learning rate: 0.0001
   Optimizer: AdamW
   Loss: BCEWithLogitsLoss (pos_weight=15.23)

======================================================================
Epoch 1/20
======================================================================
Epoch 1 [Train]: 100%|████████| 250/250 [00:45<00:00, 5.5it/s, loss=0.2134]
Epoch 1 [Val]:   100%|████████| 63/63 [00:08<00:00, 7.8it/s, loss=0.1892]

📊 Epoch 1 Summary:
   Train Loss: 0.2247 | Train AUPRC: 0.7234
   Val Loss:   0.1965 | Val AUPRC:   0.7812 | Val AUROC: 0.8456
   ✅ New best model saved! (Val Loss: 0.1965)
...
```

---

## 🔬 Model Details

### **Input Format**
- **Shape**: `(batch_size, 1, 6000)`
- **Type**: `torch.FloatTensor`
- **Data**: MALDI-TOF mass spectrometry intensities (6000 m/z bins)

### **Output Format**
- **Shape**: `(batch_size, 10)`
- **Type**: `torch.FloatTensor` (logits)
- **Range**: `[0, 1]` after sigmoid
- **Interpretation**: Probability of resistance for each antibiotic

### **Antibiotics Predicted**
1. Ceftriaxone
2. Ciprofloxacin
3. Cefixime
4. Ampicillin
5. Gentamicin
6. Trimethoprim
7. Tetracycline
8. Chloramphenicol
9. Azithromycin
10. Meropenem

---

## 📈 Performance Metrics

### **AUPRC (Area Under Precision-Recall Curve)**
- Primary metric for imbalanced multi-label classification
- More informative than AUROC for rare events
- **Target**: >0.80 per antibiotic

### **AUROC (Area Under ROC Curve)**
- Secondary metric for overall discrimination
- **Target**: >0.85 per antibiotic

### **BCEWithLogitsLoss**
- Combines sigmoid + BCE for numerical stability
- Uses pos_weight to handle class imbalance
- **Target**: <0.15 validation loss

---

## 💾 Model Checkpoints

### **Saved Files**

```
models/
├── best_model.pth           # Best model (lowest val loss)
├── final_model.pth          # Final epoch model
├── checkpoint_epoch_5.pth   # Checkpoint at epoch 5
├── checkpoint_epoch_10.pth  # Checkpoint at epoch 10
├── checkpoint_epoch_15.pth  # Checkpoint at epoch 15
└── checkpoint_epoch_20.pth  # Checkpoint at epoch 20

results/
├── logs/                    # TensorBoard logs
│   └── events.out.tfevents.*
└── training_config.json     # Training hyperparameters
```

### **Checkpoint Format**

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'loss': float,
    'timestamp': str (ISO format)
}
```

---

## 🎯 Design Rationale

### **Why ResNet-1D?**
1. **Skip connections** prevent vanishing gradients in deep networks
2. **1D convolutions** naturally handle sequential mass spec data
3. **Global average pooling** reduces overfitting vs fully connected layers
4. **Proven architecture** adapted from image classification (ResNet-18)

### **Why BCEWithLogitsLoss?**
1. Numerically stable (log-sum-exp trick)
2. Multi-label friendly (independent binary predictions)
3. pos_weight handles class imbalance without resampling

### **Why AdamW?**
1. Adaptive learning rates per parameter
2. Weight decay decoupled from gradient updates
3. Better generalization than Adam

---

## 📚 References

1. **ResNet**: He et al. (2016) - "Deep Residual Learning for Image Recognition"
2. **DRIAMS**: Weis et al. (2020) - "Direct Antimicrobial Resistance Prediction from MALDI-TOF Mass Spectra"
3. **BCEWithLogits**: PyTorch Documentation - Numerically stable binary cross-entropy

---

## 🛠️ Development

### **Adding New Antibiotics**

1. Update `num_antibiotics` parameter in model creation
2. Prepare labels with correct dimensions
3. Retrain model

```python
model = create_deepg2p_model(num_antibiotics=15)  # 5 new antibiotics
```

### **Extending Architecture**

```python
# Add more residual blocks
model = DeepG2P(
    num_blocks=[3, 4, 6, 3],  # ResNet-34 configuration
    base_channels=64
)
```

### **Custom Loss Functions**

```python
# Focal loss for extreme imbalance
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

---

## ✅ Best Practices

1. **Monitor class balance**: Check pos_weight during training
2. **Use validation set**: Never evaluate on training data
3. **Track AUPRC**: More informative than accuracy for imbalanced data
4. **Save checkpoints**: Enables recovery from crashes
5. **Log hyperparameters**: Save training_config.json for reproducibility
6. **Use TensorBoard**: Visualize training curves in real-time

---

## 🚨 Troubleshooting

### **CUDA Out of Memory**
```bash
# Reduce batch size
python src/train.py --batch-size 16

# Use smaller model
python src/train.py --model-size small
```

### **Poor Validation Performance**
- Check class balance (pos_weight should be >1 for rare events)
- Increase training epochs
- Add data augmentation (noise, shifts)
- Use larger model

### **Training Too Slow**
- Use GPU (`device='cuda'`)
- Increase batch size
- Reduce num_workers if CPU-bound
- Use mixed precision training (future feature)

---

## 📄 License

MIT License - See repository root for details.

---

**Author**: Vihaan Kulkarni  
**Contact**: [GitHub](https://github.com/vihaankulkarni29)  
**Date**: January 2026
