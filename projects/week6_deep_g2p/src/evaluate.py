"""Evaluation script for trained G2P models."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

from .data_loader import build_dataloader
from .model import get_model


def predict(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="predict", leave=False):
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            preds.append(prob.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(in_channels=args.in_channels, num_classes=args.num_classes, architecture=args.architecture)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    loader = build_dataloader(args.test, args.test_labels, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    probs, targets = predict(model, loader, device)

    # Binary classification metrics.
    ap = average_precision_score(targets, probs)
    roc = roc_auc_score(targets, probs)
    preds = (probs >= args.threshold).astype(int)
    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, digits=4)

    print(f"AUPRC: {ap:.4f}\nROC-AUC: {roc:.4f}\nConfusion matrix:\n{cm}\n\nClassification report:\n{report}")

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "probs.npy", probs)
    np.save(out_path / "targets.npy", targets)
    np.save(out_path / "preds.npy", preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test", required=True, help="Path to X_test.npy")
    parser.add_argument("--test-labels", required=True, help="Path to y_test.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--architecture", choices=["resnet", "cnn"], default="resnet")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
