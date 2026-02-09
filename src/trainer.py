"""
Training pipeline for models.
"""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from .models import DenseClassifier, Autoencoder, RBM
from .data_utils import generate_fake_data, split_and_scale


class Trainer:
    """Main training class"""
    
    def __init__(self, model_type='dense', config=None, output_dir='results'):
        """
        Initialize trainer.
        
        Args:
            model_type (str): Type of model ('dense', 'autoencoder', 'rbm')
            config (dict): Configuration dictionary
            output_dir (str): Directory to save results
        """
        self.model_type = model_type
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        if model_type == 'dense':
            self.model = DenseClassifier(
                input_dim=self.config.get('input_dim', 100),
                hidden_units=self.config.get('hidden_units', 128),
                num_classes=self.config.get('num_classes', 2)
            )
            self.model.compile()
        elif model_type == 'autoencoder':
            self.model = Autoencoder(
                input_dim=self.config.get('input_dim', 100),
                encoding_dim=self.config.get('encoding_dim', 32)
            )
            self.model.compile()
        elif model_type == 'rbm':
            self.model = RBM(
                visible_units=self.config.get('input_dim', 100),
                hidden_units=self.config.get('hidden_units', 32),
                learning_rate=self.config.get('learning_rate', 0.01)
            )
        
        self.history = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def train(self, X_train, Y_train=None, X_val=None, Y_val=None, epochs=100, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            Y_train (np.ndarray): Training labels (for supervised models)
            X_val (np.ndarray): Validation features
            Y_val (np.ndarray): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
        
        Returns:
            dict: Training history
        """
        if self.model_type == 'dense':
            return self._train_dense(X_train, Y_train, X_val, Y_val, epochs, batch_size)
        elif self.model_type == 'autoencoder':
            return self._train_autoencoder(X_train, X_val, epochs, batch_size)
        elif self.model_type == 'rbm':
            return self._train_rbm(X_train, epochs, batch_size)
    
    def _train_dense(self, X_train, Y_train, X_val, Y_val, epochs, batch_size):
        """Train dense classifier"""
        validation_data = None
        if X_val is not None and Y_val is not None:
            validation_data = (X_val, Y_val)
        
        history = self.model.get_model().fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self.history = history.history
        return self.history
    
    def _train_autoencoder(self, X_train, X_val, epochs, batch_size):
        """Train autoencoder"""
        validation_data = X_val if X_val is not None else None
        
        history = self.model.get_autoencoder().fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val) if validation_data is not None else None,
            verbose=1
        )
        
        self.history = history.history
        return self.history
    
    def _train_rbm(self, X_train, epochs, batch_size):
        """Train RBM"""
        num_batches = (X_train.shape[0] + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, X_train.shape[0])
                batch = X_train[start:end]
                self.model.train_batch(batch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
        
        return {'status': 'completed'}
    
    def save_model(self, name=None):
        """
        Save trained model.
        
        Args:
            name (str): Model name
        """
        if name is None:
            name = f"{self.model_type}_{self.timestamp}"
        
        model_path = os.path.join(self.output_dir, f"{name}.h5")
        
        if self.model_type in ['dense', 'autoencoder']:
            if self.model_type == 'dense':
                self.model.get_model().save(model_path)
            else:
                self.model.get_autoencoder().save(model_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def save_history(self, name=None):
        """
        Save training history.
        
        Args:
            name (str): History name
        """
        if name is None:
            name = f"{self.model_type}_history_{self.timestamp}"
        
        history_path = os.path.join(self.output_dir, f"{name}.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                serializable_history[key] = [float(v) if isinstance(v, np.ndarray) else v for v in value]
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        return history_path


def main():
    """Example training script"""
    # Configuration
    config = {
        'input_dim': 100,
        'hidden_units': 128,
        'num_classes': 2
    }
    
    # Generate sample data
    X, Y = generate_fake_data(num_samples=1000)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # Train model
    trainer = Trainer(model_type='dense', config=config)
    print("Training model...")
    trainer.train(X_train, Y_train, X_test, Y_test, epochs=10)
    
    # Save results
    trainer.save_model('simple_classifier')
    trainer.save_history('simple_classifier')
    
    # Evaluate
    model = trainer.model.get_model()
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
