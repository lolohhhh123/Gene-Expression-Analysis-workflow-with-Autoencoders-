"""
Data loading and preprocessing utilities for gene expression data.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Optional
import yaml


class GeneDataLoader:
    """Load and preprocess gene expression data."""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
    
    def load_data_pair(self, sample_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data for a given sample."""
        train_path = os.path.join(
            self.data_config['input_dir'],
            self.data_config['train_dir'],
            f"{sample_name}_train.xlsx"
        )
        test_path = os.path.join(
            self.data_config['input_dir'],
            self.data_config['test_dir'],
            f"{sample_name}_test.xlsx"
        )
        
        train_data = pd.read_excel(train_path, index_col=0)
        test_data = pd.read_excel(test_path, index_col=0)
        
        return train_data, test_data
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing steps to data."""
        config = self.data_config['preprocessing']
        
        # Drop NaN values
        if config['drop_na']:
            data = data.dropna(axis=0, how='any')
        
        # Replace NaN with specified value
        data = data.replace('nan', config['replace_nan'])
        
        # Convert to numpy array
        data_array = data.values.astype(np.float32)
        
        # Normalize
        data_array = self._normalize(data_array, config['normalization_method'])
        
        # Transpose if needed
        if config['transpose']:
            data_array = data_array.T
        
        return data_array
    
    def _normalize(self, data: np.ndarray, method: str) -> np.ndarray:
        """Normalize data using specified method."""
        if method == 'max':
            return data / (data.max(axis=1, keepdims=True) + 1e-7)
        elif method == 'minmax':
            data_min = data.min(axis=1, keepdims=True)
            data_range = data.max(axis=1, keepdims=True) - data_min
            return (data - data_min) / (data_range + 1e-7)
        elif method == 'zscore':
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            return (data - mean) / (std + 1e-7)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_gene_index_mapping(self, gene_ids: np.ndarray) -> Dict[str, int]:
        """Create mapping from gene IDs to indices."""
        return {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
