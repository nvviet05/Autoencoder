# Neural Network Classification with Autoencoder & RBM

A standard AI research repository for deep learning experiments with neural networks, autoencoders, and Restricted Boltzmann Machines (RBM).

## Project Structure

```
├── src/                      # Source code
│   ├── __init__.py
│   ├── models.py            # Model definitions
│   ├── data_utils.py        # Data loading & preprocessing
│   ├── trainer.py           # Training pipeline
│   └── inference.py         # Prediction & inference
├── data/                     # Dataset directory
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_autoencoder_rbm.ipynb
├── models/                  # Saved model weights
│   └── .gitkeep
├── configs/                 # Configuration files
│   ├── default_config.yaml
│   └── experiment_config.yaml
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_models.py
├── results/                 # Output results, logs, plots
│   └── .gitkeep
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Matplotlib
- PyYAML

## Installation

```bash
# Clone repository
git clone <repository-url>
cd neuron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/trainer.py --config configs/default_config.yaml
```

### Inference

```bash
python src/inference.py --model models/trained_model.h5 --input data/sample.npy
```

### Notebooks

Explore experiments in Jupyter:

```bash
jupyter notebook notebooks/
```

## Models

- **Dense Neural Network**: Simple feedforward network for classification
- **Autoencoder**: Unsupervised learning for feature extraction
- **Restricted Boltzmann Machine**: Generative model for deep belief networks

## Results

Training results, metrics, and visualizations are saved in `results/` directory.

## Author

Nguyen Van Viet

## License

MIT License
