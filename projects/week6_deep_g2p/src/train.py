"""Training loop for G2P models."""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

from .data_loader import build_dataloader, create_splits
from .model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(in_channels=args.in_channels, num_classes=args.num_classes, architecture=args.architecture)
    model.to(device)

    full_loader = build_dataloader(args.train, args.train_labels, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_idx, val_idx, _ = create_splits(args.train, args.train_labels)  # uses labels file
    train_subset = Subset(full_loader.dataset, train_idx)
    val_subset = Subset(full_loader.dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, Path(args.out_dir) / "best.pt")

    save_checkpoint(model, Path(args.out_dir) / "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to X_train.npy")
    parser.add_argument("--train-labels", required=True, help="Path to y_train.npy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--architecture", choices=["resnet", "cnn"], default="resnet")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
