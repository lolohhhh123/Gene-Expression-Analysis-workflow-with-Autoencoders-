"""
Data Preprocessing Module for Gene Expression Analysis

This module handles data loading, merging, and train-test splitting
for gene expression datasets.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')


def setup_preprocessing_logger(output_dir: Path = None) -> logging.Logger:
    """Setup logger for preprocessing operations."""
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if output directory is provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "data_preprocessing.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def load_dataframe(file_path: Path, file_type: str = "") -> Optional[pd.DataFrame]:
    """
    Load data from CSV or Excel file with proper index handling.
    
    Args:
        file_path: Path to data file
        file_type: Description of file type for logging
        
    Returns:
        Loaded DataFrame or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
        
        logger.info(f"Loaded {file_type} from {file_path.name}: {df.shape}")
        
        # Handle index column
        if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
            df = df.set_index(df.columns[0])
            logger.debug(f"Set column '{df.columns[0]}' as index")
        else:
            # Use first column as index (assuming it's gene names)
            df = df.set_index(df.columns[0])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None


def deduplicate_dataframe(df: pd.DataFrame, file_type: str = "") -> pd.DataFrame:
    """
    Handle duplicate indices by averaging duplicate rows.
    
    Args:
        df: Input DataFrame
        file_type: Description for logging
        
    Returns:
        DataFrame with unique indices
    """
    logger = logging.getLogger(__name__)
    
    if df.index.duplicated().any():
        duplicate_count = df.index.duplicated().sum()
        logger.warning(f"{file_type} contains {duplicate_count} duplicate indices, averaging duplicates")
        
        # Average duplicate rows
        df_dedup = df.groupby(df.index).mean()
        
        logger.info(f"Deduplicated {file_type}: {df.shape} -> {df_dedup.shape}")
        return df_dedup
    
    return df


def find_control_ad_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify Control and AD columns in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (control_columns, ad_columns)
    """
    # Look for columns starting with 'Control' or 'AD'
    control_patterns = ['Control', 'control', 'CTRL', 'ctrl', 'Normal', 'normal']
    ad_patterns = ['AD', 'ad', 'Alzheimer', 'alzheimer', 'Disease', 'disease']
    
    control_cols = []
    ad_cols = []
    
    for col in df.columns:
        col_str = str(col)
        
        # Check for control patterns
        if any(pattern in col_str for pattern in control_patterns):
            control_cols.append(col)
        # Check for AD patterns
        elif any(pattern in col_str for pattern in ad_patterns):
            ad_cols.append(col)
        else:
            # Try to infer from column structure
            if 'control' in col_str.lower():
                control_cols.append(col)
            elif 'ad' in col_str.lower() or 'alzheimer' in col_str.lower():
                ad_cols.append(col)
    
    return control_cols, ad_cols


def generate_column_names(base_name: str, start_idx: int, count: int) -> List[str]:
    """
    Generate standardized column names.
    
    Args:
        base_name: Base name for columns (e.g., 'Control', 'AD')
        start_idx: Starting index number
        count: Number of columns to generate
        
    Returns:
        List of column names
    """
    return [f"{base_name}.{i}" for i in range(start_idx, start_idx + count)]


def merge_datasets(original_df: pd.DataFrame,
                   synthetic_control_df: pd.DataFrame,
                   synthetic_ad_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merge original and synthetic datasets.
    
    Args:
        original_df: Original dataset
        synthetic_control_df: Synthetic control dataset
        synthetic_ad_df: Synthetic AD dataset
        
    Returns:
        Merged DataFrame or None if merge fails
    """
    logger = logging.getLogger(__name__)
    
    # Find common indices
    common_indices = (
        original_df.index
        .intersection(synthetic_control_df.index)
        .intersection(synthetic_ad_df.index)
    )
    
    if len(common_indices) == 0:
        logger.error("No common indices found between datasets")
        return None
    
    logger.info(f"Found {len(common_indices)} common indices")
    
    # Select common rows
    original_common = original_df.loc[common_indices]
    control_common = synthetic_control_df.loc[common_indices]
    ad_common = synthetic_ad_df.loc[common_indices]
    
    # Merge datasets
    merged_df = pd.concat(
        [original_common, control_common, ad_common],
        axis=1
    )
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    
    return merged_df


def split_dataset(merged_df: pd.DataFrame,
                  test_ratio: float = 0.5,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.
    
    Args:
        merged_df: Merged dataset
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger = logging.getLogger(__name__)
    
    np.random.seed(random_state)
    
    # Get all columns
    all_columns = merged_df.columns.tolist()
    
    # Shuffle columns
    shuffled_columns = np.random.permutation(all_columns)
    
    # Calculate split point
    split_point = int(len(shuffled_columns) * (1 - test_ratio))
    
    # Split columns
    train_columns = shuffled_columns[:split_point]
    test_columns = shuffled_columns[split_point:]
    
    # Create train and test datasets
    train_df = merged_df[train_columns]
    test_df = merged_df[test_columns]
    
    logger.info(f"Train set: {train_df.shape} ({len(train_columns)} columns)")
    logger.info(f"Test set: {test_df.shape} ({len(test_columns)} columns)")
    logger.info(f"Split ratio: {train_df.shape[1]}:{test_df.shape[1]} "
                f"(train:test)")
    
    return train_df, test_df


def save_datasets(train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  output_dir: Path,
                  sample_name: str,
                  format: str = 'xlsx') -> None:
    """
    Save train and test datasets to files.
    
    Args:
        train_df: Training dataset
        test_df: Testing dataset
        output_dir: Output directory
        sample_name: Base name for output files
        format: Output format ('xlsx' or 'csv')
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'xlsx':
        train_path = output_dir / f"{sample_name}_train.xlsx"
        test_path = output_dir / f"{sample_name}_test.xlsx"
        
        train_df.to_excel(train_path)
        test_df.to_excel(test_path)
        
        logger.info(f"Saved training data to: {train_path}")
        logger.info(f"Saved testing data to: {test_path}")
        
    elif format.lower() == 'csv':
        train_path = output_dir / f"{sample_name}_train.csv"
        test_path = output_dir / f"{sample_name}_test.csv"
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        logger.info(f"Saved training data to: {train_path}")
        logger.info(f"Saved testing data to: {test_path}")
    
    else:
        logger.error(f"Unsupported format: {format}")
        raise ValueError(f"Unsupported format: {format}")


def process_sample(original_path: Path,
                   synthetic_ad_path: Path,
                   synthetic_control_path: Path,
                   train_output_dir: Path,
                   test_output_dir: Path,
                   sample_name: str,
                   test_ratio: float = 0.5,
                   random_state: int = 42) -> bool:
    """
    Process a single sample through the preprocessing pipeline.
    
    Args:
        original_path: Path to original CSV file
        synthetic_ad_path: Path to synthetic AD file
        synthetic_control_path: Path to synthetic control file
        train_output_dir: Directory for training output
        test_output_dir: Directory for testing output
        sample_name: Name of the sample
        test_ratio: Proportion for test split
        random_state: Random seed
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing sample: {sample_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Load datasets
        logger.info("Loading datasets...")
        original_df = load_dataframe(original_path, "Original")
        synthetic_ad_df = load_dataframe(synthetic_ad_path, "Synthetic AD")
        synthetic_control_df = load_dataframe(synthetic_control_path, "Synthetic Control")
        
        if original_df is None or synthetic_ad_df is None or synthetic_control_df is None:
            logger.error("Failed to load one or more datasets")
            return False
        
        # Deduplicate datasets
        logger.info("Deduplicating datasets...")
        original_df = deduplicate_dataframe(original_df, "Original")
        synthetic_ad_df = deduplicate_dataframe(synthetic_ad_df, "Synthetic AD")
        synthetic_control_df = deduplicate_dataframe(synthetic_control_df, "Synthetic Control")
        
        # Identify column types in original data
        control_cols, ad_cols = find_control_ad_columns(original_df)
        logger.info(f"Found {len(control_cols)} Control columns, {len(ad_cols)} AD columns in original data")
        
        # Generate new column names for synthetic data
        synthetic_ad_cols = generate_column_names(
            "AD", len(ad_cols) + 1, synthetic_ad_df.shape[1]
        )
        synthetic_control_cols = generate_column_names(
            "Control", len(control_cols) + 1, synthetic_control_df.shape[1]
        )
        
        # Rename synthetic data columns
        synthetic_ad_df.columns = synthetic_ad_cols
        synthetic_control_df.columns = synthetic_control_cols
        
        logger.info(f"Renamed {len(synthetic_ad_cols)} AD columns, "
                   f"{len(synthetic_control_cols)} Control columns")
        
        # Merge datasets
        logger.info("Merging datasets...")
        merged_df = merge_datasets(original_df, synthetic_control_df, synthetic_ad_df)
        
        if merged_df is None:
            logger.error("Failed to merge datasets")
            return False
        
        # Sort columns (Control first, then AD)
        control_cols_all = [col for col in merged_df.columns if 'Control' in col]
        ad_cols_all = [col for col in merged_df.columns if 'AD' in col]
        
        # Sort by the number after the dot
        control_cols_all.sort(key=lambda x: int(x.split('.')[1]) if '.' in x else 0)
        ad_cols_all.sort(key=lambda x: int(x.split('.')[1]) if '.' in x else 0)
        
        merged_df = merged_df[control_cols_all + ad_cols_all]
        
        # Split dataset
        logger.info("Splitting dataset...")
        train_df, test_df = split_dataset(
            merged_df, test_ratio=test_ratio, random_state=random_state
        )
        
        # Save datasets
        logger.info("Saving datasets...")
        save_datasets(
            train_df, test_df, train_output_dir, sample_name, format='xlsx'
        )
        save_datasets(
            train_df, test_df, test_output_dir, sample_name, format='xlsx'
        )
        
        # Save column information
        save_column_info(
            sample_name, train_output_dir,
            train_columns=train_df.columns.tolist(),
            test_columns=test_df.columns.tolist(),
            control_columns=control_cols_all,
            ad_columns=ad_cols_all
        )
        
        logger.info(f"Successfully processed {sample_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {sample_name}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def save_column_info(sample_name: str,
                     output_dir: Path,
                     train_columns: List[str],
                     test_columns: List[str],
                     control_columns: List[str],
                     ad_columns: List[str]) -> None:
    """
    Save column information for a sample.
    
    Args:
        sample_name: Name of the sample
        output_dir: Output directory
        train_columns: List of training columns
        test_columns: List of testing columns
        control_columns: List of control columns
        ad_columns: List of AD columns
    """
    info_dir = output_dir / "column_info"
    info_dir.mkdir(exist_ok=True)
    
    # Save column lists
    column_info = {
        'train_columns': train_columns,
        'test_columns': test_columns,
        'control_columns': control_columns,
        'ad_columns': ad_columns,
        'total_columns': len(train_columns) + len(test_columns)
    }
    
    # Save as JSON
    import json
    json_path = info_dir / f"{sample_name}_columns.json"
    with open(json_path, 'w') as f:
        json.dump(column_info, f, indent=2)
    
    # Save as CSV for easy viewing
    columns_df = pd.DataFrame({
        'column_type': ['train'] * len(train_columns) + ['test'] * len(test_columns),
        'column_name': train_columns + test_columns,
        'condition': [
            'Control' if 'Control' in col else 'AD'
            for col in train_columns + test_columns
        ]
    })
    
    csv_path = info_dir / f"{sample_name}_columns.csv"
    columns_df.to_csv(csv_path, index=False)


def run_preprocessing_pipeline(original_dir: Path,
                               synthetic_dir: Path,
                               train_output_dir: Path,
                               test_output_dir: Path,
                               test_ratio: float = 0.5,
                               random_state: int = 42) -> Dict[str, bool]:
    """
    Run preprocessing pipeline on all samples in directory.
    
    Args:
        original_dir: Directory with original CSV files
        synthetic_dir: Directory with synthetic XLSX files
        train_output_dir: Directory for training output
        test_output_dir: Directory for testing output
        test_ratio: Proportion for test split
        random_state: Random seed
        
    Returns:
        Dictionary with processing results for each sample
    """
    logger = logging.getLogger(__name__)
    
    # Setup directories
    for directory in [train_output_dir, test_output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Find original CSV files
    original_files = list(original_dir.glob("*.csv"))
    
    if not original_files:
        logger.error(f"No CSV files found in {original_dir}")
        return {}
    
    logger.info(f"Found {len(original_files)} CSV files in {original_dir}")
    
    # Process each file
    results = {}
    
    for original_file in original_files:
        sample_name = original_file.stem
        
        # Construct expected synthetic file paths
        synthetic_ad_file = synthetic_dir / f"{sample_name}_synthetic_AD.xlsx"
        synthetic_control_file = synthetic_dir / f"{sample_name}_synthetic_Control.xlsx"
        
        # Check if synthetic files exist
        if not synthetic_ad_file.exists():
            logger.warning(f"Synthetic AD file not found: {synthetic_ad_file}")
            results[sample_name] = False
            continue
        
        if not synthetic_control_file.exists():
            logger.warning(f"Synthetic Control file not found: {synthetic_control_file}")
            results[sample_name] = False
            continue
        
        # Process sample
        success = process_sample(
            original_file,
            synthetic_ad_file,
            synthetic_control_file,
            train_output_dir,
            test_output_dir,
            sample_name,
            test_ratio,
            random_state
        )
        
        results[sample_name] = success
    
    # Create summary report
    create_summary_report(results, train_output_dir)
    
    return results


def create_summary_report(results: Dict[str, bool], output_dir: Path) -> None:
    """
    Create a summary report of preprocessing results.
    
    Args:
        results: Dictionary of processing results
        output_dir: Output directory for report
    """
    logger = logging.getLogger(__name__)
    
    successful = [sample for sample, success in results.items() if success]
    failed = [sample for sample, success in results.items() if not success]
    
    summary = {
        'total_samples': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(results) if len(results) > 0 else 0,
        'successful_samples': successful,
        'failed_samples': failed
    }
    
    # Save summary as JSON
    import json
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary as CSV
    summary_df = pd.DataFrame([
        {
            'sample': sample,
            'status': 'SUCCESS' if success else 'FAILED',
            'original_file': f"{sample}.csv",
            'synthetic_ad_file': f"{sample}_synthetic_AD.xlsx",
            'synthetic_control_file': f"{sample}_synthetic_Control.xlsx"
        }
        for sample, success in results.items()
    ])
    
    csv_path = output_dir / "preprocessing_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.2%}")
    
    if failed:
        logger.info("\nFailed samples:")
        for sample in failed:
            logger.info(f"  - {sample}")


class DataPreprocessor:
    """
    Main class for data preprocessing operations.
    """
    
    def __init__(self,
                 original_dir: Path,
                 synthetic_dir: Path,
                 output_dir: Path,
                 test_ratio: float = 0.5,
                 random_state: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            original_dir: Directory with original CSV files
            synthetic_dir: Directory with synthetic XLSX files
            output_dir: Base output directory
            test_ratio: Proportion for test split
            random_state: Random seed
        """
        self.original_dir = Path(original_dir)
        self.synthetic_dir = Path(synthetic_dir)
        self.output_dir = Path(output_dir)
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Setup output directories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.log_dir = self.output_dir / "logs"
        
        # Setup logger
        self.logger = setup_preprocessing_logger(self.log_dir)
    
    def run(self) -> Dict[str, bool]:
        """
        Run the preprocessing pipeline.
        
        Returns:
            Dictionary of processing results
        """
        self.logger.info("Starting data preprocessing pipeline")
        self.logger.info(f"Original data directory: {self.original_dir}")
        self.logger.info(f"Synthetic data directory: {self.synthetic_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Test ratio: {self.test_ratio}")
        
        results = run_preprocessing_pipeline(
            self.original_dir,
            self.synthetic_dir,
            self.train_dir,
            self.test_dir,
            self.test_ratio,
            self.random_state
        )
        
        return results
    
    def validate_outputs(self) -> Dict[str, Dict]:
        """
        Validate the generated train and test files.
        
        Returns:
            Dictionary with validation results for each sample
        """
        self.logger.info("Validating output files...")
        
        validation_results = {}
        
        # Get all processed samples
        train_files = list(self.train_dir.glob("*_train.xlsx"))
        
        for train_file in train_files:
            sample_name = train_file.stem.replace('_train', '')
            test_file = self.test_dir / f"{sample_name}_test.xlsx"
            
            validation = {
                'train_file_exists': train_file.exists(),
                'test_file_exists': test_file.exists(),
                'train_shape': None,
                'test_shape': None,
                'column_overlap': None,
                'validation_passed': False
            }
            
            if train_file.exists() and test_file.exists():
                try:
                    # Load files
                    train_df = pd.read_excel(train_file, index_col=0)
                    test_df = pd.read_excel(test_file, index_col=0)
                    
                    validation['train_shape'] = train_df.shape
                    validation['test_shape'] = test_df.shape
                    
                    # Check for column overlap (should be none)
                    train_cols = set(train_df.columns)
                    test_cols = set(test_df.columns)
                    column_overlap = train_cols.intersection(test_cols)
                    
                    validation['column_overlap'] = len(column_overlap)
                    
                    # Check that all columns are unique
                    if len(column_overlap) == 0:
                        validation['validation_passed'] = True
                        self.logger.info(f"✓ {sample_name}: Validation passed")
                    else:
                        self.logger.warning(f"✗ {sample_name}: Column overlap found: {column_overlap}")
                
                except Exception as e:
                    self.logger.error(f"✗ {sample_name}: Validation error: {str(e)}")
            
            validation_results[sample_name] = validation
        
        # Save validation results
        validation_df = pd.DataFrame.from_dict(validation_results, orient='index')
        validation_path = self.output_dir / "validation_results.csv"
        validation_df.to_csv(validation_path)
        
        self.logger.info(f"Validation results saved to {validation_path}")
        
        return validation_results


def main():
    """Command-line interface for data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess gene expression data for autoencoder training'
    )
    
    parser.add_argument('--original', '-o', type=str, required=True,
                       help='Directory with original CSV files')
    parser.add_argument('--synthetic', '-s', type=str, required=True,
                       help='Directory with synthetic XLSX files')
    parser.add_argument('--output', '-d', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--test-ratio', '-t', type=float, default=0.5,
                       help='Test set ratio (default: 0.5)')
    parser.add_argument('--random-state', '-r', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate outputs after preprocessing')
    
    args = parser.parse_args()
    
    # Create and run preprocessor
    preprocessor = DataPreprocessor(
        original_dir=args.original,
        synthetic_dir=args.synthetic,
        output_dir=args.output,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # Run preprocessing
    results = preprocessor.run()
    
    # Validate if requested
    if args.validate:
        validation_results = preprocessor.validate_outputs()
        
        # Print summary
        successful = sum(1 for v in validation_results.values() 
                        if v.get('validation_passed', False))
        total = len(validation_results)
        
        print(f"\nValidation complete: {successful}/{total} samples passed validation")
    
    return results


if __name__ == "__main__":
    main()
