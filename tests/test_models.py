"""
Unit tests for models module.
"""

import pytest
import numpy as np
from src.models import DenseClassifier, Autoencoder, RBM


class TestDenseClassifier:
    """Test DenseClassifier"""
    
    def setup_method(self):
        """Setup for each test"""
        self.classifier = DenseClassifier(
            input_dim=100,
            hidden_units=128,
            num_classes=2
        )
    
    def test_model_creation(self):
        """Test model is created successfully"""
        model = self.classifier.get_model()
        assert model is not None
    
    def test_model_structure(self):
        """Test model has correct structure"""
        model = self.classifier.get_model()
        assert len(model.layers) == 3  # Dense, Dropout, Dense
    
    def test_forward_pass(self):
        """Test forward pass with random input"""
        model = self.classifier.get_model()
        X = np.random.random((10, 100)).astype(np.float32)
        output = model.predict(X, verbose=0)
        
        assert output.shape == (10, 2)
        assert np.all((output >= 0) & (output <= 1))  # Probabilities


class TestAutoencoder:
    """Test Autoencoder"""
    
    def setup_method(self):
        """Setup for each test"""
        self.autoencoder = Autoencoder(
            input_dim=100,
            encoding_dim=32
        )
    
    def test_encoder_creation(self):
        """Test encoder is created"""
        encoder = self.autoencoder.get_encoder()
        assert encoder is not None
    
    def test_encoder_output_shape(self):
        """Test encoder output shape"""
        encoder = self.autoencoder.get_encoder()
        X = np.random.random((10, 100)).astype(np.float32)
        encoded = encoder.predict(X, verbose=0)
        
        assert encoded.shape == (10, 32)
    
    def test_autoencoder_output_shape(self):
        """Test autoencoder reconstructs original shape"""
        model = self.autoencoder.get_autoencoder()
        X = np.random.random((10, 100)).astype(np.float32)
        reconstructed = model.predict(X, verbose=0)
        
        assert reconstructed.shape == X.shape


class TestRBM:
    """Test RBM"""
    
    def setup_method(self):
        """Setup for each test"""
        self.rbm = RBM(
            visible_units=100,
            hidden_units=32,
            learning_rate=0.01
        )
    
    def test_rbm_creation(self):
        """Test RBM is created"""
        assert self.rbm is not None
        assert self.rbm.weights.shape == (100, 32)
    
    def test_forward_pass(self):
        """Test forward pass"""
        v = np.random.random((10, 100))
        h_prob, h_sample = self.rbm.forward(v)
        
        assert h_prob.shape == (10, 32)
        assert h_sample.shape == (10, 32)
    
    def test_backward_pass(self):
        """Test backward pass"""
        h = np.random.random((10, 32))
        v_prob, v_sample = self.rbm.backward(h)
        
        assert v_prob.shape == (10, 100)
        assert v_sample.shape == (10, 100)


if __name__ == '__main__':
    pytest.main([__file__])
