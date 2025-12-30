"""
Vocabulary class for Amino Acid tokenization for a Generative LSTM model.
"""


class PeptideVocab:
    """Handles tokenization of peptide sequences."""
    
    def __init__(self):
        """Initialize vocabulary with 20 standard amino acids and special tokens."""
        # Standard 20 amino acids
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>']
        
        # Build vocabulary: special tokens first, then amino acids
        self.vocab = special_tokens + list(amino_acids)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Vocabulary size
        self.vocab_size = len(self.vocab)
    
    def encode(self, sequence):
        """
        Encode a peptide sequence to indices with SOS and EOS tokens.
        
        Args:
            sequence: String of amino acids (e.g., "MKA")
            
        Returns:
            List of integers with <SOS> at start and <EOS> at end
        """
        encoded = [self.char_to_idx['<SOS>']]
        
        for char in sequence:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        
        encoded.append(self.char_to_idx['<EOS>'])
        
        return encoded
    
    def decode(self, indices):
        """
        Decode a list of indices back to a peptide sequence.
        
        Args:
            indices: List of integers
            
        Returns:
            String of amino acids (special tokens removed)
        """
        sequence = []
        
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                # Skip special tokens
                if char not in ['<PAD>', '<SOS>', '<EOS>']:
                    sequence.append(char)
        
        return ''.join(sequence)


if __name__ == "__main__":
    # Test the vocabulary
    vocab = PeptideVocab()
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Vocabulary: {vocab.vocab}\n")
    
    # Test encoding/decoding
    test_sequence = "AKL"
    encoded = vocab.encode(test_sequence)
    decoded = vocab.decode(encoded)
    
    print(f"Original: {test_sequence}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    print(f"\nMatch: {test_sequence == decoded}")
