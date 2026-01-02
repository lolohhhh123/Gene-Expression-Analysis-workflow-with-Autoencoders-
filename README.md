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
## Original Data Preprocessing (Without Synthetic Data)
For datasets that don't have synthetic data, use the original-only preprocessing module. Others please use quick_original_preprocess.py to divide the data into train and test dataset according to the test-ratio

# Process a single original file
python scripts/quick_original_preprocess.py single data/original/sample1.csv --output processed_original

# Process all files in a directory
python scripts/quick_original_preprocess.py directory data/original --output processed_original --test-ratio 0.3 --validate

# Using the main module
python data_preprocessing_original_only.py --input data/original --output processed_original --test-ratio 0.3 --validate

# Run the main training script
python autoencoder_gene_analysis.py

# With custom configuration
python autoencoder_gene_analysis.py --config config/custom_config.yaml

# Z-score Analysis
After running the autoencoder analysis, use the Z-score module to further analyze feature importance:
python zscore_analysis.py --input data/results --output analysis/zscore_results

# Quick analysis
python -c "from utils.zscore_calculator import quick_zscore_analysis; quick_zscore_analysis('data/results')"

# Integrated pipeline
python integrated_analysis.py data/results
