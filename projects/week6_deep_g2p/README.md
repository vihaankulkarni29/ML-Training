# Week 6 – Deep G2P

Deep learning pipeline for genotype-to-phenotype (G2P) prediction using large antimicrobial resistance datasets (e.g., DRIAMS). The layout favors scalability (chunked loading, simple configs) and reproducibility (clear scripts, explicit requirements).

## Structure
- `data/raw` — source DRIAMS dumps (large; do not commit).
- `data/processed` — cleaned numpy arrays (`.npy`) for model input.
- `src/data_loader.py` — streaming/feeder utilities to handle large arrays.
- `src/model.py` — 1D-CNN / ResNet-style backbones.
- `src/train.py` — training loop with logging/checkpoint hooks.
- `src/evaluate.py` — evaluation script (AUPRC, ROC-AUC, confusion matrix).
- `notebooks/exploratory_analysis.ipynb` — quick EDA and sanity checks.
- `requirements.txt` — pinned minimal dependencies.

## Quickstart
1) Place raw DRIAMS data under `data/raw/`.
2) Convert to numpy arrays (features + labels) and save to `data/processed/`.
3) Run training:
   ```bash
   python -m src.train --train data/processed/X_train.npy --train-labels data/processed/y_train.npy \
     --val data/processed/X_val.npy --val-labels data/processed/y_val.npy --out-dir results/
   ```
4) Evaluate a checkpoint:
   ```bash
   python -m src.evaluate --checkpoint results/best.pt --test data/processed/X_test.npy --test-labels data/processed/y_test.npy
   ```

## Notes
- Keep `.npy` arrays in channel-first shape expected by the model (e.g., `[N, C, L]`).
- Prefer balanced sampling or class weighting for skewed resistance labels.
- Track seeds and configs for reproducibility.
