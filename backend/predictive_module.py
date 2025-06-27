
from sklearn.model_selection import train_test_split
import os
os.environ["KERAS_BACKEND"] = "torch"
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
import matplotlib.pyplot as plt
import keras
from keras import layers, optimizers, callbacks
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# Set random seeds for reproducibility
np.random.seed(42)

class UserBehaviorDataProcessor:
    def __init__(self):
        self.token_to_idx = {
            'PageView': 0,
            'AddToCart': 1,
            'Buy': 2,
            'Favorite': 3,
            'PAD': 4  # Padding token
        }
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        self.pad_token = self.token_to_idx['PAD']

    def prepare_sequences(self, dataframe):
        """
        Convert sequences to input/target pairs for next-token prediction
        """
        X, y = [], []

        for idx in range(len(dataframe)):
            sequence = dataframe.iloc[idx]  # Assuming sequence is in column 1

            # Convert tokens to indices
            sequence_idx = [self.token_to_idx[action] for action in sequence if action in self.token_to_idx]

            # Skip very short sequences
            if len(sequence_idx) <= 1:
                continue

            # Create input/target pairs (proper next-token prediction)
            input_seq = sequence_idx[:-1]    # All but last token
            target_seq = sequence_idx[1:]     # All but first token

            X.append(input_seq)
            y.append(target_seq)

        return X, y

    def pad_sequences(self, sequences, maxlen=None):
        """Pad sequences to same length"""
        return pad_sequences(sequences, maxlen=maxlen, padding='post', value=self.pad_token)

def create_lstm_model(vocab_size, embedding_dim=128, lstm_units=128, dropout=0.3):
    """
    Create a simple LSTM model for sequence prediction
    """
    model = keras.Sequential([
        # Embedding layer
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,  # Automatically handle padding
            name='embedding'
        ),

        # LSTM layers
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='lstm_1'
        ),

        # Dense layer for prediction
        layers.Dense(vocab_size, activation='softmax', name='output')
    ])

    return model

def create_callbacks(patience=3):
    """
    Create training callbacks
    """
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # Save best model
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callback_list

def train_model(sequence_train, sequence_val, epochs=10, batch_size=32):
    """
    Main training function
    """
    print("Preparing data...")

    # Initialize data processor
    processor = UserBehaviorDataProcessor()

    # Prepare training data
    X_train, y_train = processor.prepare_sequences(sequence_train)
    X_val, y_val = processor.prepare_sequences(sequence_val)

    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")

    # Find max sequence length
    max_len = max(max(len(seq) for seq in X_train), max(len(seq) for seq in X_val))
    print(f"Max sequence length: {max_len}")

    # Pad sequences
    X_train_pad = processor.pad_sequences(X_train, maxlen=max_len)
    y_train_pad = processor.pad_sequences(y_train, maxlen=max_len)
    X_val_pad = processor.pad_sequences(X_val, maxlen=max_len)
    y_val_pad = processor.pad_sequences(y_val, maxlen=max_len)

    print(f"Training data shape: {X_train_pad.shape}")
    print(f"Training target shape: {y_train_pad.shape}")

    # Create model
    print("\nCreating model...")
    model = create_lstm_model(
        vocab_size=processor.vocab_size,
        embedding_dim=64,
        lstm_units=64,
        dropout=0.3
    )

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Create callbacks
    callback_list = create_callbacks(patience=3)

    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train_pad, y_train_pad,
        validation_data=(X_val_pad, y_val_pad),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )

    return model, history, processor

def evaluate_model(model, processor, sequence_test):
    """
    Evaluate model on test set with multiple metrics
    """
    print("Evaluating on test set...")

    # Prepare test data
    X_test, y_test = processor.prepare_sequences(sequence_test)

    # Find max length (should be same as training)
    max_len = max(len(seq) for seq in X_test)

    # Pad sequences
    X_test_pad = processor.pad_sequences(X_test, maxlen=max_len)
    y_test_pad = processor.pad_sequences(y_test, maxlen=max_len)

    # Evaluate
    loss, accuracy = model.evaluate(X_test_pad, y_test_pad, verbose=0)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get predictions
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Flatten predictions and true values, excluding padding
    y_true_flat = []
    y_pred_flat = []
    for true_seq, pred_seq in zip(y_test_pad, y_pred):
        for true_token, pred_token in zip(true_seq, pred_seq):
            if true_token != processor.pad_token:
                y_true_flat.append(true_token)
                y_pred_flat.append(pred_token)

    # Calculate additional metrics
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted')
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted')
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')

    print(f"Test Precision (weighted): {precision:.4f}")
    print(f"Test Recall (weighted): {recall:.4f}")
    print(f"Test F1-score (weighted): {f1:.4f}")

    # Print classification report for more detailed analysis
    print("\nClassification Report:")
    print(classification_report(y_true_flat, y_pred_flat, target_names=list(processor.token_to_idx.keys())[:-1])) # Exclude PAD token

    return loss, accuracy, precision, recall, f1

def predict_next_token(model, processor, sequence, top_k=3):
    """
    Predict next tokens for a given sequence
    """
    # Convert sequence to indices
    sequence_idx = [processor.token_to_idx[token] for token in sequence if token in processor.token_to_idx]

    # Pad sequence
    sequence_pad = processor.pad_sequences([sequence_idx])

    # Predict
    predictions = model.predict(sequence_pad, verbose=0)

    # Get predictions for last token
    last_pred = predictions[0, len(sequence_idx)-1, :]

    # Get top-k predictions
    top_indices = np.argsort(last_pred)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        token = processor.idx_to_token[idx]
        prob = last_pred[idx]
        results.append((token, prob))

    return results


# Simple one-liner training (if you just want to run it quickly)
def quick_train(sequence_train, sequence_val, sequence_test=None):
    """
    Ultra-simple training function
    """
    model, history, processor = train_model(sequence_train, sequence_val)

    if sequence_test is not None:
        evaluate_model(model, processor, sequence_test)

    return model, history, processor

model_path = 'Trained_Model/best_model.keras'
loaded_model = keras.models.load_model(model_path)
processor = UserBehaviorDataProcessor()

def predict_next(sequence, top_k=4):
    results = predict_next_token(loaded_model, processor, sequence, top_k)
    return results