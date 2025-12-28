"""
Vocabulary module for converting amino acids to numeric indices and vice versa.
This module handles the tokenization of peptide sequences for the LSTM model.
"""

import numpy as np
from typing import List, Dict


class AminoAcidVocabulary:
    """
    A vocabulary class to map amino acids to numeric indices and back.
    Includes special tokens for padding, start, and end of sequences.
    """
    
    def __init__(self):
        """Initialize vocabulary with 20 standard amino acids plus special tokens."""
        # Standard 20 amino acids
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'  # Padding token
        self.START_TOKEN = '<START>'  # Start of sequence
        self.END_TOKEN = '<END>'  # End of sequence
        self.UNK_TOKEN = '<UNK>'  # Unknown amino acid
        
        # Build vocabulary
        self.special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        self.vocab = self.special_tokens + self.amino_acids
        
        # Create mappings
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.vocab)}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_to_idx.items()}
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert an amino acid sequence to a list of indices.
        
        Args:
            sequence: String of amino acids (e.g., 'ACDEFGH')
            add_special_tokens: Whether to add START and END tokens
            
        Returns:
            List of integer indices
        """
        encoded = []
        
        if add_special_tokens:
            encoded.append(self.aa_to_idx[self.START_TOKEN])
        
        for aa in sequence:
            if aa in self.aa_to_idx:
                encoded.append(self.aa_to_idx[aa])
            else:
                # Unknown amino acid
                encoded.append(self.aa_to_idx[self.UNK_TOKEN])
        
        if add_special_tokens:
            encoded.append(self.aa_to_idx[self.END_TOKEN])
        
        return encoded
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Convert a list of indices back to an amino acid sequence.
        
        Args:
            indices: List of integer indices
            remove_special_tokens: Whether to remove special tokens from output
            
        Returns:
            String of amino acids
        """
        sequence = []
        
        for idx in indices:
            if idx in self.idx_to_aa:
                aa = self.idx_to_aa[idx]
                
                if remove_special_tokens and aa in self.special_tokens:
                    continue
                    
                sequence.append(aa)
        
        return ''.join(sequence)
    
    def encode_batch(self, sequences: List[str], max_length: int = None, 
                     add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode a batch of sequences with padding.
        
        Args:
            sequences: List of amino acid sequences
            max_length: Maximum sequence length (will pad/truncate to this length)
            add_special_tokens: Whether to add START and END tokens
            
        Returns:
            2D numpy array of shape (batch_size, max_length)
        """
        encoded_sequences = [self.encode(seq, add_special_tokens) for seq in sequences]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_sequences)
        
        # Pad sequences
        padded = np.full((len(sequences), max_length), self.aa_to_idx[self.PAD_TOKEN], dtype=np.int32)
        
        for i, seq in enumerate(encoded_sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def get_pad_idx(self) -> int:
        """Return the index of the padding token."""
        return self.aa_to_idx[self.PAD_TOKEN]
    
    def get_start_idx(self) -> int:
        """Return the index of the start token."""
        return self.aa_to_idx[self.START_TOKEN]
    
    def get_end_idx(self) -> int:
        """Return the index of the end token."""
        return self.aa_to_idx[self.END_TOKEN]


if __name__ == "__main__":
    # Example usage
    vocab = AminoAcidVocabulary()
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Vocabulary: {vocab.vocab}")
    print()
    
    # Test encoding
    test_sequence = "ACDEFGHIKLMNPQRSTVWY"
    encoded = vocab.encode(test_sequence)
    print(f"Original sequence: {test_sequence}")
    print(f"Encoded: {encoded}")
    
    # Test decoding
    decoded = vocab.decode(encoded)
    print(f"Decoded: {decoded}")
    print()
    
    # Test batch encoding
    sequences = ["ACE", "FGHIKL", "MNPQR"]
    batch = vocab.encode_batch(sequences, max_length=10)
    print(f"Batch encoded shape: {batch.shape}")
    print(f"Batch:\n{batch}")
