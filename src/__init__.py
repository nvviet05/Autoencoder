"""
Neural Network Classification Package
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .models import get_dense_model, build_autoencoder
from .data_utils import generate_fake_data, load_data
from .trainer import Trainer
from .inference import predict

__all__ = [
    'get_dense_model',
    'build_autoencoder',
    'generate_fake_data',
    'load_data',
    'Trainer',
    'predict'
]
