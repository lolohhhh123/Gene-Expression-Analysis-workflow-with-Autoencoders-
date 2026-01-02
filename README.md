# Gene Expression Analysis with Autoencoders

A comprehensive workflow for gene expression data analysis using various autoencoder architectures for dimensionality reduction and feature extraction.

## Features

- Multiple autoencoder architectures:
  - Basic Autoencoder
  - Variational Autoencoder (VAE)
  - Denoising Autoencoder
  - Sparse Autoencoder
- Memory-efficient training with GPU support
- Automatic hyperparameter tuning
- Random Forest feature importance analysis
- Comprehensive logging and error handling
- Z-score Analysis

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
# Run the main training script
python autoencoder_gene_analysis.py

# With custom configuration
python autoencoder_gene_analysis.py --config config/custom_config.yaml

# Z-score Analysis
After running the autoencoder analysis, use the Z-score module to further analyze feature importance:
```bash
python zscore_analysis.py --input data/results --output analysis/zscore_results

# Quick analysis
python -c "from utils.zscore_calculator import quick_zscore_analysis; quick_zscore_analysis('data/results')"

# Integrated pipeline
python integrated_analysis.py data/results
