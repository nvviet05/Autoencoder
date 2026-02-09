"""
Inference utilities for making predictions with trained models.
"""

import numpy as np
import tensorflow as tf


def predict(model, X, return_probabilities=False):
    """
    Make predictions with the model.
    
    Args:
        model: Trained Keras model
        X (np.ndarray): Input features
        return_probabilities (bool): Return probabilities instead of class labels
    
    Returns:
        np.ndarray: Predictions
    """
    predictions = model.predict(X)
    
    if return_probabilities:
        return predictions
    else:
        return np.argmax(predictions, axis=1)


def predict_single(model, x, return_probability=False):
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained Keras model
        x (np.ndarray): Input features (1D array)
        return_probability (bool): Return probability instead of class label
    
    Returns:
        int or float: Prediction or probability
    """
    x = np.reshape(x, (1, -1))
    prediction = model.predict(x, verbose=0)
    
    if return_probability:
        return prediction[0]
    else:
        return np.argmax(prediction[0])


def batch_predict(model, X, batch_size=32):
    """
    Make predictions in batches (memory efficient).
    
    Args:
        model: Trained Keras model
        X (np.ndarray): Input features
        batch_size (int): Batch size
    
    Returns:
        np.ndarray: Predictions
    """
    predictions = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)
    
    return np.vstack(predictions)


def load_model(model_path):
    """
    Load a saved Keras model.
    
    Args:
        model_path (str): Path to saved model
    
    Returns:
        model: Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def inference_pipeline(model_path, data_path, output_path=None):
    """
    Complete inference pipeline.
    
    Args:
        model_path (str): Path to saved model
        data_path (str): Path to input data
        output_path (str): Path to save predictions (optional)
    
    Returns:
        np.ndarray: Predictions
    """
    # Load model and data
    model = load_model(model_path)
    X = np.load(data_path)
    
    # Make predictions
    predictions = predict(model, X)
    
    # Save predictions if output path provided
    if output_path:
        np.save(output_path, predictions)
        print(f"Predictions saved to {output_path}")
    
    return predictions


class Predictor:
    """Predictor class for managing inference"""
    
    def __init__(self, model_path):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path (str): Path to saved model
        """
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def predict(self, X):
        """Make predictions"""
        return predict(self.model, X)
    
    def predict_single(self, x):
        """Predict single sample"""
        return predict_single(self.model, x)
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence scores.
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            tuple: (predictions, confidences)
        """
        probabilities = self.model.predict(X)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences


def main():
    """Example inference script"""
    # Generate sample data
    X_test = np.random.random((10, 100)).astype(np.float32)
    
    # Make predictions (requires a trained model)
    # predictor = Predictor('models/trained_model.h5')
    # predictions = predictor.predict(X_test)
    # print("Predictions:", predictions)
    
    print("Inference module ready. Use Predictor class or inference_pipeline function.")


if __name__ == '__main__':
    main()
