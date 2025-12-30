"""
Inference script for generating novel antimicrobial peptide sequences.
Loads trained LSTM model and generates diverse peptide candidates.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch

from train_generator import PeptideLSTM
from vocab import PeptideVocab


# ============================================================================
# Generation Functions
# ============================================================================

class PeptideGenerator:
    """Wrapper for peptide sequence generation from trained LSTM."""

    def __init__(self, model_path: str = "models/peptide_lstm.pth", device: Optional[str] = None):
        """
        Initialize generator with trained model and vocabulary.

        Args:
            model_path: Path to trained model checkpoint
            device: torch device ('cuda' or 'cpu'). Auto-detected if None.
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load vocabulary
        self.vocab = PeptideVocab()

        # Load model (weights_only=False for compatibility)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = PeptideLSTM(
            vocab_size=self.vocab.vocab_size,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    def generate_sequence(
        self,
        start_char: str = "<SOS>",
        max_length: int = 50,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate a single peptide sequence using temperature sampling.

        Args:
            start_char: Starting character (amino acid or <SOS>)
            max_length: Maximum sequence length
            temperature: Sampling temperature
                - < 1.0: Conservative (follows training data)
                - 1.0: Standard probability distribution
                - > 1.0: Creative (explores unusual sequences)

        Returns:
            Generated peptide sequence (amino acids only, no special tokens)
        """
        self.model.eval()

        with torch.no_grad():
            # Initialize with start token
            if start_char in self.vocab.char_to_idx:
                current_idx = self.vocab.char_to_idx[start_char]
            else:
                current_idx = self.vocab.char_to_idx["<SOS>"]

            sequence = [current_idx]
            hidden = self.model.init_hidden(1, self.device)

            # Generate sequence one token at a time
            for step in range(max_length):
                # Prepare input
                x = torch.tensor([[current_idx]], dtype=torch.long, device=self.device)

                # Forward pass
                logits, hidden = self.model(x, hidden)

                # Apply temperature
                logits = logits[0, -1, :] / temperature

                # Get probabilities
                probs = torch.softmax(logits, dim=0)

                # Sample next token
                current_idx = torch.multinomial(probs, 1).item()

                # Check for end token
                if current_idx == self.vocab.char_to_idx["<EOS>"]:
                    break

                sequence.append(current_idx)

            # Decode sequence (removes special tokens)
            return self.vocab.decode(sequence)

    def generate_batch(
        self,
        num_sequences: int = 50,
        temperature: float = 0.8,
        start_chars: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Generate a batch of peptide sequences.

        Args:
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            start_chars: List of starting characters (cycles through if provided)
            verbose: Print progress

        Returns:
            List of generated sequences
        """
        sequences = []

        if start_chars is None:
            start_chars = ["<SOS>"]

        for i in range(num_sequences):
            start_char = start_chars[i % len(start_chars)]
            seq = self.generate_sequence(
                start_char=start_char,
                temperature=temperature,
            )
            sequences.append(seq)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_sequences} sequences")

        return sequences

    def save_sequences(self, sequences: List[str], output_path: str = "results/generated_peptides.csv") -> None:
        """
        Save generated sequences to CSV file.

        Args:
            sequences: List of peptide sequences
            output_path: Path to output CSV file
        """
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get unique sequences
        unique_sequences = list(dict.fromkeys(sequences))  # Preserves order

        # Write to CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "length", "generation_index"])

            for idx, seq in enumerate(unique_sequences):
                writer.writerow([seq, len(seq), idx])

        print(f"\n✓ Saved {len(unique_sequences)} unique sequences to {output_path}")

    def analyze_batch(self, sequences: List[str]) -> Dict:
        """
        Compute statistics on generated sequences.

        Args:
            sequences: List of peptide sequences

        Returns:
            Dictionary of statistics
        """
        from collections import Counter

        if not sequences:
            return {}

        lengths = [len(seq) for seq in sequences]
        aa_counts = Counter()

        for seq in sequences:
            aa_counts.update(seq)

        stats = {
            "total_generated": len(sequences),
            "unique_sequences": len(set(sequences)),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "amino_acid_distribution": dict(aa_counts),
        }

        return stats

    def print_stats(self, stats: Dict) -> None:
        """Pretty print generation statistics."""
        print("\n" + "=" * 70)
        print("GENERATION STATISTICS")
        print("=" * 70)
        print(f"Total Generated:     {stats['total_generated']}")
        print(f"Unique Sequences:    {stats['unique_sequences']}")
        print(f"Uniqueness Rate:     {stats['unique_sequences'] / stats['total_generated'] * 100:.1f}%")
        print(f"\nLength Statistics:")
        print(f"  Average Length:    {stats['avg_length']:.1f} AA")
        print(f"  Min Length:        {stats['min_length']} AA")
        print(f"  Max Length:        {stats['max_length']} AA")
        print(f"\nAmino Acid Distribution (Top 10):")

        aa_dist = stats["amino_acid_distribution"]
        sorted_aa = sorted(aa_dist.items(), key=lambda x: x[1], reverse=True)[:10]

        for aa, count in sorted_aa:
            pct = count / sum(aa_dist.values()) * 100
            print(f"  {aa}: {count:4d} ({pct:5.1f}%)")

        print("=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main generation pipeline."""
    print("\n" + "=" * 70)
    print("PEPTIDE SEQUENCE GENERATOR - Inference Pipeline")
    print("=" * 70 + "\n")

    # Initialize generator
    print("1. Loading trained model...")
    generator = PeptideGenerator(model_path="models/peptide_lstm.pth")

    # Generate sequences with different temperatures
    print("\n2. Generating sequences (Temperature=0.8)...")
    sequences = generator.generate_batch(
        num_sequences=50,
        temperature=0.8,
        verbose=True,
    )

    # Compute statistics
    print("\n3. Analyzing generated sequences...")
    stats = generator.analyze_batch(sequences)
    generator.print_stats(stats)

    # Save results
    print("\n4. Saving results...")
    generator.save_sequences(sequences, output_path="results/generated_peptides.csv")

    # Display top 5 sequences
    print("\n5. Top 5 Generated Sequences:")
    print("-" * 70)
    for i, seq in enumerate(sequences[:5], 1):
        # Basic analysis
        hydrophobic_aa = "LVIF"
        charged_aa = "KR"
        hydrophobic_pct = sum(seq.count(aa) for aa in hydrophobic_aa) / len(seq) * 100 if seq else 0
        charged_pct = sum(seq.count(aa) for aa in charged_aa) / len(seq) * 100 if seq else 0

        print(
            f"{i}. {seq:<45} | Len: {len(seq):2d} | "
            f"Hydrophobic: {hydrophobic_pct:5.1f}% | Charged: {charged_pct:5.1f}%"
        )

    print("-" * 70)

    # Temperature comparison
    print("\n6. Temperature Comparison (5 sequences each):")
    print("-" * 70)

    for temp in [0.5, 0.8, 1.0, 1.2]:
        print(f"\nTemperature = {temp}:")
        temp_seqs = generator.generate_batch(
            num_sequences=5,
            temperature=temp,
            verbose=False,
        )
        for i, seq in enumerate(temp_seqs, 1):
            print(f"  {i}. {seq}")

    print("-" * 70)
    print("\n✓ Generation complete!")
    print(f"✓ Results saved to: results/generated_peptides.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
