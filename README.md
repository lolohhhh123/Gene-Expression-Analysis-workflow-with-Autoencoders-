# Gene Expression Analysis with DAE-Lin and Autoencoders

A comprehensive workflow for gene expression data analysis using DAE-Lin or other various autoencoder architectures for dimensionality reduction and feature extraction.

## Features
- Preprocess data to divide test/train dataset
- Proposed DAE-Lin architectures
- Multiple autoencoder architectures:
  - Basic Autoencoder
  - Variational Autoencoder (VAE)
  - Denoising Autoencoder
  - Sparse Autoencoder
- Memory-efficient training with GPU support (could also be run in CPU)
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
├── original/
│ ├── sample1.csv
│ └── ...
├── DAE-Lin.py
└── Autoencoder.py

# Run with defaults
python Autoencoder.py
python DAE-Lin.py --input_dir ./original

# Or Run with customize:
python Autoencoder.py --models autoencoder vae --embedding_dims 100 200 --epochs 30
python DAE-Lin.py --input_dir ./original --output_dir ./DAE_Lin_results --latent_dim 32 --hidden_dims 128,64 --epochs 200 --lr 1e-3 --alpha 1.0 --beta 1e-4 --patience 15 --device cpu
