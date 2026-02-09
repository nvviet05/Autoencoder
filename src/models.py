"""
Model definitions for classification, autoencoder, and RBM.
"""

import tensorflow as tf
import numpy as np


class DenseClassifier:
    """Simple Dense Neural Network for binary classification"""
    
    def __init__(self, input_dim=100, hidden_units=128, num_classes=2):
        """
        Initialize dense classifier.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_units (int): Number of units in hidden layer
            num_classes (int): Number of output classes
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build model architecture"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=self.hidden_units,
                activation='relu',
                input_shape=(self.input_dim,),
                name='hidden_layer'
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                units=self.num_classes,
                activation='softmax',
                name='output_layer'
            )
        ])
        return model
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy'):
        """Compile model"""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the Keras model"""
        return self.model


class Autoencoder:
    """Autoencoder for unsupervised feature learning"""
    
    def __init__(self, input_dim=100, encoding_dim=32):
        """
        Initialize autoencoder.
        
        Args:
            input_dim (int): Input dimension
            encoding_dim (int): Latent space dimension
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder, self.decoder, self.autoencoder = self._build_model()
    
    def _build_model(self):
        """Build encoder, decoder, and full autoencoder"""
        # Encoder
        input_data = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(64, activation='relu')(input_data)
        encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        encoder = tf.keras.Model(input_data, encoded)
        
        # Decoder
        encoded_input = tf.keras.Input(shape=(self.encoding_dim,))
        decoded = tf.keras.layers.Dense(64, activation='relu')(encoded_input)
        decoded = tf.keras.layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        decoder = tf.keras.Model(encoded_input, decoded)
        
        # Full autoencoder
        autoencoder = tf.keras.Model(input_data, decoder(encoder(input_data)))
        
        return encoder, decoder, autoencoder
    
    def compile(self, optimizer='adam'):
        """Compile autoencoder"""
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='mse'
        )
    
    def get_autoencoder(self):
        """Return the autoencoder model"""
        return self.autoencoder
    
    def get_encoder(self):
        """Return the encoder model"""
        return self.encoder


class RBM:
    """Restricted Boltzmann Machine"""
    
    def __init__(self, visible_units=100, hidden_units=32, learning_rate=0.01):
        """
        Initialize RBM.
        
        Args:
            visible_units (int): Number of visible units
            hidden_units (int): Number of hidden units
            learning_rate (float): Learning rate for training
        """
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.01, (visible_units, hidden_units))
        self.visible_bias = np.zeros(visible_units)
        self.hidden_bias = np.zeros(hidden_units)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, v):
        """Forward pass: visible to hidden"""
        h_prob = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        h_sample = (h_prob > np.random.random(h_prob.shape)).astype(np.float32)
        return h_prob, h_sample
    
    def backward(self, h):
        """Backward pass: hidden to visible"""
        v_prob = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        v_sample = (v_prob > np.random.random(v_prob.shape)).astype(np.float32)
        return v_prob, v_sample
    
    def train_batch(self, v):
        """Train on a batch using Contrastive Divergence"""
        # Positive phase
        h_prob_pos, h_sample_pos = self.forward(v)
        
        # Negative phase
        v_prob_neg, v_sample_neg = self.backward(h_sample_pos)
        h_prob_neg, _ = self.forward(v_sample_neg)
        
        # Update weights and biases
        positive = np.dot(v.T, h_prob_pos)
        negative = np.dot(v_sample_neg.T, h_prob_neg)
        
        self.weights += self.learning_rate * (positive - negative) / v.shape[0]
        self.visible_bias += self.learning_rate * np.mean(v - v_sample_neg, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(h_prob_pos - h_prob_neg, axis=0)


def get_dense_model(input_dim=100, hidden_units=128, num_classes=2):
    """Factory function to create a dense classifier"""
    classifier = DenseClassifier(input_dim, hidden_units, num_classes)
    classifier.compile()
    return classifier.get_model()


def build_autoencoder(input_dim=100, encoding_dim=32):
    """Factory function to create an autoencoder"""
    ae = Autoencoder(input_dim, encoding_dim)
    ae.compile()
    return ae.get_autoencoder()
