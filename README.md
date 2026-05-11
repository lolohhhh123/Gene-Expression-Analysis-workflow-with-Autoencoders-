# Gene Expression Analysis with Autoencoders

A comprehensive workflow for gene expression data analysis using various autoencoder architectures for dimensionality reduction and feature extraction.

## Features
- Preprocess data to divide test/train dataset
- Multiple autoencoder architectures:
  - Basic Autoencoder
  - Variational Autoencoder (VAE)
  - Denoising Autoencoder
  - Sparse Autoencoder
- Memory-efficient training with GPU support
- Automatic hyperparameter tuning
- Random Forest feature importance analysis
- Z-score Analysis
- Comprehensive logging and error handling

## Installation

```bash
# Clone the repository
git clone https://github.com/lolohhhh123/Gene-Expression-Analysis-workflow-with-Autoencoders-.git
cd gene-autoencoder

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Quick Start

Place your data following the structure:
.
├── train/
│ ├── sample1_train.xlsx
│ └── ...
├── test/
│ ├── sample1_test.xlsx
│ └── ...
└── Autoencoder.py

# Run with defaults
python Autoencoder.py

# Or Run with customize:
python Autoencoder.py --models autoencoder vae --embedding_dims 100 200 --epochs 30
