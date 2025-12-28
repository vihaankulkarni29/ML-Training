"""
LSTM-based Peptide Sequence Generator
This script trains a character-level LSTM to generate novel peptide sequences
based on the E. coli training dataset.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import json

from vocab import AminoAcidVocabulary


class PeptideDataLoader:
    """Load and preprocess peptide sequences for LSTM training."""
    
    def __init__(self, data_path, vocab, max_length=None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file containing peptide sequences
            vocab: AminoAcidVocabulary instance
            max_length: Maximum sequence length (None to use longest sequence)
        """
        self.vocab = vocab
        self.data_path = data_path
        self.max_length = max_length
        
    def load_sequences(self, sequence_column='Sequence'):
        """
        Load peptide sequences from CSV file.
        
        Args:
            sequence_column: Name of column containing sequences
            
        Returns:
            List of peptide sequences
        """
        df = pd.read_csv(self.data_path)
        
        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in dataset")
        
        sequences = df[sequence_column].dropna().tolist()
        sequences = [str(seq).upper() for seq in sequences]
        
        # Filter out sequences with invalid characters
        valid_sequences = []
        for seq in sequences:
            if all(aa in self.vocab.amino_acids or aa == ' ' for aa in seq):
                valid_sequences.append(seq.replace(' ', ''))
        
        print(f"Loaded {len(valid_sequences)} valid sequences from {len(sequences)} total")
        
        return valid_sequences
    
    def prepare_training_data(self, sequences):
        """
        Prepare input and target sequences for LSTM training.
        
        Args:
            sequences: List of peptide sequences
            
        Returns:
            X: Input sequences (encoded)
            y: Target sequences (one-hot encoded)
            max_length: Maximum sequence length used
        """
        if self.max_length is None:
            # Add 2 for START and END tokens
            self.max_length = max(len(seq) for seq in sequences) + 2
        
        # Encode all sequences
        encoded_sequences = self.vocab.encode_batch(
            sequences, 
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        # Create input (X) and target (y) pairs
        # X: sequence[:-1], y: sequence[1:]
        X = encoded_sequences[:, :-1]
        y = encoded_sequences[:, 1:]
        
        # Convert y to one-hot encoding
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=self.vocab.vocab_size)
        
        print(f"Training data shape: X={X.shape}, y={y_one_hot.shape}")
        
        return X, y_one_hot, self.max_length


class PeptideLSTM:
    """LSTM model for peptide sequence generation."""
    
    def __init__(self, vocab_size, max_length, embedding_dim=128, lstm_units=256):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            max_length: Maximum sequence length
            embedding_dim: Dimension of embedding layer
            lstm_units: Number of LSTM units per layer
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_model(self):
        """Build the LSTM architecture."""
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length - 1,
                name='embedding'
            ),
            
            # First LSTM layer with return sequences
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                name='lstm_1'
            ),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Second LSTM layer
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                name='lstm_2'
            ),
            layers.Dropout(0.3, name='dropout_2'),
            
            # Dense layer for each timestep
            layers.TimeDistributed(
                layers.Dense(self.vocab_size, activation='softmax'),
                name='output'
            )
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=64, model_save_path='models/peptide_lstm.keras'):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences (one-hot)
            X_val: Validation input sequences (optional)
            y_val: Validation target sequences (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            model_save_path: Path to save the best model
            
        Returns:
            Training history
        """
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def generate_sequence(self, vocab, start_token=None, max_length=50, temperature=1.0):
        """
        Generate a new peptide sequence.
        
        Args:
            vocab: AminoAcidVocabulary instance
            start_token: Starting token index (uses START token if None)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated peptide sequence as string
        """
        if start_token is None:
            start_token = vocab.get_start_idx()
        
        # Start with the start token
        sequence = [start_token]
        
        for _ in range(max_length):
            # Prepare input (pad to model's expected length)
            x = np.zeros((1, self.max_length - 1))
            x[0, :len(sequence)] = sequence[:self.max_length - 1]
            
            # Predict next token
            predictions = self.model.predict(x, verbose=0)[0, len(sequence) - 1]
            
            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))
            
            # Sample next token
            next_token = np.random.choice(len(predictions), p=predictions)
            
            # Stop if END token is generated
            if next_token == vocab.get_end_idx():
                break
            
            sequence.append(next_token)
        
        # Decode and return
        return vocab.decode(sequence, remove_special_tokens=True)


def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    # Configuration
    DATA_PATH = '../data/ecolitraining_set_80.csv'
    MODEL_SAVE_PATH = '../models/peptide_lstm.keras'
    EMBEDDING_DIM = 128
    LSTM_UNITS = 256
    BATCH_SIZE = 64
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    print("=" * 60)
    print("LSTM Peptide Generator Training")
    print("=" * 60)
    
    # Initialize vocabulary
    print("\n1. Initializing vocabulary...")
    vocab = AminoAcidVocabulary()
    print(f"   Vocabulary size: {vocab.vocab_size}")
    
    # Load data
    print("\n2. Loading and preparing data...")
    data_loader = PeptideDataLoader(DATA_PATH, vocab)
    sequences = data_loader.load_sequences()
    X, y, max_length = data_loader.prepare_training_data(sequences)
    
    # Train/validation split
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Max sequence length: {max_length}")
    
    # Build model
    print("\n3. Building LSTM model...")
    lstm = PeptideLSTM(
        vocab_size=vocab.vocab_size,
        max_length=max_length,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS
    )
    lstm.build_model()
    lstm.compile_model()
    lstm.summary()
    
    # Train model
    print("\n4. Training model...")
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Plot training history
    print("\n5. Saving training history...")
    plot_training_history(history, '../models/training_history.png')
    
    # Save training config
    config = {
        'vocab_size': vocab.vocab_size,
        'max_length': max_length,
        'embedding_dim': EMBEDDING_DIM,
        'lstm_units': LSTM_UNITS,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('../models/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Generate sample sequences
    print("\n6. Generating sample peptide sequences...")
    print("-" * 60)
    for i in range(10):
        seq = lstm.generate_sequence(vocab, temperature=0.8)
        print(f"   Sample {i+1}: {seq}")
    print("-" * 60)
    
    print("\n✓ Training complete!")
    print(f"   Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Training history saved to: ../models/training_history.png")
    print(f"   Config saved to: ../models/config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
