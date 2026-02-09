"""
Data utilities for loading, preprocessing, and generating data.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_fake_data(num_samples=1000, input_dim=100, num_classes=2, seed=42):
    """
    Generate synthetic data for testing and development.
    
    Args:
        num_samples (int): Number of samples
        input_dim (int): Feature dimension
        num_classes (int): Number of classes
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X, Y) feature matrix and labels
    """
    np.random.seed(seed)
    X = np.random.random((num_samples, input_dim)).astype(np.float32)
    Y = np.random.randint(num_classes, size=(num_samples, 1)).astype(np.int32)
    return X, Y


def load_data(filepath, split_ratio=0.2, scale=True):
    """
    Load data from file.
    
    Args:
        filepath (str): Path to data file (.npy, .csv, etc.)
        split_ratio (float): Train-test split ratio
        scale (bool): Whether to scale features
    
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    if filepath.endswith('.npy'):
        X = np.load(filepath)
    elif filepath.endswith('.csv'):
        data = np.loadtxt(filepath, delimiter=',')
        X = data[:, :-1]
        Y = data[:, -1]
        return split_and_scale(X, Y, split_ratio, scale)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return X


def split_and_scale(X, Y, split_ratio=0.2, scale=True):
    """
    Split data into train/test and optionally scale.
    
    Args:
        X (np.ndarray): Feature matrix
        Y (np.ndarray): Labels
        split_ratio (float): Test split ratio
        scale (bool): Whether to scale features
    
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=split_ratio, random_state=42
    )
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


def save_data(X, Y, output_dir, prefix='data'):
    """
    Save data to disk.
    
    Args:
        X (np.ndarray): Feature matrix
        Y (np.ndarray): Labels
        output_dir (str): Output directory
        prefix (str): Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{prefix}_X.npy'), X)
    np.save(os.path.join(output_dir, f'{prefix}_Y.npy'), Y)
    print(f"Data saved to {output_dir}")


def create_batches(X, Y, batch_size=32, shuffle=True):
    """
    Create mini-batches for training.
    
    Args:
        X (np.ndarray): Feature matrix
        Y (np.ndarray): Labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
    
    Yields:
        tuple: (batch_X, batch_Y)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield X[batch_indices], Y[batch_indices]
