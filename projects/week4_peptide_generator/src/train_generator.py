"""
LSTM-based peptide generator using PyTorch.
Trains a character-level RNN to generate antimicrobial peptide sequences.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vocab import PeptideVocab


class PeptideDataset(Dataset):
    """Dataset that tokenizes peptide sequences with PeptideVocab."""

    def __init__(self, csv_path: str, vocab: PeptideVocab, min_length: int = 10, max_length: int = 50):
        self.vocab = vocab
        self.min_length = min_length
        self.max_length = max_length
        df = pd.read_csv(csv_path)
        if 'SEQUENCE' not in df.columns:
            raise ValueError("CSV must contain 'SEQUENCE' column")

        allowed = set(vocab.vocab)
        self.sequences = []
        for seq in df['SEQUENCE'].dropna():
            seq = str(seq).upper().strip()
            if min_length <= len(seq) <= max_length and all(ch in allowed for ch in seq):
                self.sequences.append(seq)

        print(f"Loaded {len(self.sequences)} sequences (length {min_length}-{max_length})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.sequences[idx])
        return torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch):
    batch = sorted(batch, key=len, reverse=True)
    return pad_sequence(batch, batch_first=True, padding_value=0)


class PeptideLSTM(nn.Module):
    """Two-layer LSTM for next-token prediction."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_sequence(model, vocab, device, max_length: int = 50, temperature: float = 1.0):
    model.eval()
    with torch.no_grad():
        idx = vocab.char_to_idx['<SOS>']
        sequence = [idx]
        hidden = model.init_hidden(1, device)

        for _ in range(max_length):
            x = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            idx = torch.multinomial(probs, 1).item()
            if idx == vocab.char_to_idx['<EOS>']:
                break
            sequence.append(idx)

    return vocab.decode(sequence)


def main():
    config = {
        'data_path': Path('data/ecolitraining_set_80.csv'),
        'model_dir': Path('models'),
        'model_path': Path('models/peptide_lstm.pth'),
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 50,
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'min_length': 10,
        'max_length': 50,
    }

    print("=" * 70)
    print("Char-RNN Peptide Generator (PyTorch)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab = PeptideVocab()
    print(f"Vocab size: {vocab.vocab_size}")

    dataset = PeptideDataset(
        csv_path=config['data_path'],
        vocab=vocab,
        min_length=config['min_length'],
        max_length=config['max_length'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = PeptideLSTM(
        vocab_size=vocab.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    config['model_dir'].mkdir(exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, config['epochs'] + 1):
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch:02d}/{config['epochs']} - Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config}, config['model_path'])
            print(f"  Saved best model (loss {loss:.4f})")

        if epoch % 10 == 0:
            ckpt_path = config['model_dir'] / f"peptide_lstm_epoch_{epoch}.pth"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config}, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")
            print("  Sample generations:")
            for i in range(3):
                print(f"    {i+1}. {generate_sequence(model, vocab, device, temperature=0.8)}")

    config_path = config['model_dir'] / 'config.json'
    with open(config_path, 'w') as f:
        out_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        out_cfg['best_loss'] = best_loss
        out_cfg['training_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        json.dump(out_cfg, f, indent=4)

    print("=" * 70)
    print(f"Training finished. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {config['model_path']}")
    print(f"Config saved to: {config_path}")
    print("Sample sequences:")
    for i in range(5):
        print(f"  {i+1}. {generate_sequence(model, vocab, device, temperature=0.8)}")
    print("=" * 70)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
