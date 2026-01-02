"""
Original Data Preprocessing Module for Gene Expression Analysis

This module handles data loading and train-test splitting for 
original gene expression datasets without synthetic data.
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import logging
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')


class OriginalDataPreprocessor:
    """
    Preprocess original gene expression data without synthetic data.
    """
    
    def __init__(self,
                 input_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 test_ratio: float = 0.3,
                 random_state: int = 42,
                 log_level: str = "INFO"):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing original CSV files
            output_dir: Base output directory
            test_ratio: Proportion of data for testing (default: 0.3)
            random_state: Random seed for reproducibility
            log_level: Logging level
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Setup directories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.log_dir = self.output_dir / "logs"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        for directory in [self.train_dir, self.test_dir, 
                         self.log_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Track processing statistics
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'processed_samples': [],
            'failed_samples': []
        }
    
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        level = log_levels.get(log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "original_preprocessing.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_original_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and preprocess original CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        try:
            self.logger.info(f"Loading file: {file_path.name}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if first column is index column
            if df.columns[0] in ['Unnamed: 0', '', 'index', 'Index']:
                df = df.set_index(df.columns[0])
                self.logger.debug(f"Set '{df.columns[0]}' as index")
            else:
                # Assume first column contains gene names
                df = df.set_index(df.columns[0])
            
            self.logger.info(f"Loaded data shape: {df.shape}")
            
            # Handle duplicate indices
            if df.index.duplicated().any():
                self.logger.warning(f"Found duplicate indices, averaging duplicates")
                df = self.deduplicate_dataframe(df)
            
            # Basic quality checks
            if df.empty:
                self.logger.error(f"DataFrame is empty after loading")
                return None
            
            if df.isnull().values.any():
                null_count = df.isnull().sum().sum()
                self.logger.warning(f"Found {null_count} null values, filling with 0")
                df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def deduplicate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate indices by averaging.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        if df.index.duplicated().any():
            original_shape = df.shape
            df_dedup = df.groupby(df.index).mean()
            self.logger.info(f"Deduplicated: {original_shape} -> {df_dedup.shape}")
            return df_dedup
        return df
    
    def identify_sample_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify different sample types based on column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (control_columns, ad_columns, other_columns)
        """
        control_patterns = [
            'Control', 'control', 'CTRL', 'ctrl', 
            'Normal', 'normal', 'Healthy', 'healthy'
        ]
        
        ad_patterns = [
            'AD', 'ad', 'Alzheimer', 'alzheimer',
            'Disease', 'disease', 'Patient', 'patient'
        ]
        
        control_cols = []
        ad_cols = []
        other_cols = []
        
        for col in df.columns:
            col_str = str(col)
            
            # Check for control patterns
            if any(pattern.lower() in col_str.lower() for pattern in control_patterns):
                control_cols.append(col)
            # Check for AD patterns
            elif any(pattern.lower() in col_str.lower() for pattern in ad_patterns):
                ad_cols.append(col)
            else:
                other_cols.append(col)
        
        return control_cols, ad_cols, other_cols
    
    def create_stratified_split(self, 
                               control_cols: List[str],
                               ad_cols: List[str],
                               other_cols: List[str],
                               test_ratio: float) -> Tuple[List[str], List[str]]:
        """
        Create stratified split preserving sample type distribution.
        
        Args:
            control_cols: Control sample columns
            ad_cols: AD sample columns
            other_cols: Other sample columns
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_columns, test_columns)
        """
        np.random.seed(self.random_state)
        
        train_cols = []
        test_cols = []
        
        # Split each category separately to preserve distribution
        for category_name, category_cols in [
            ('Control', control_cols),
            ('AD', ad_cols),
            ('Other', other_cols)
        ]:
            if not category_cols:
                continue
            
            # Shuffle columns
            shuffled = np.random.permutation(category_cols)
            
            # Calculate split point
            split_idx = int(len(shuffled) * (1 - test_ratio))
            
            # Split
            cat_train = shuffled[:split_idx]
            cat_test = shuffled[split_idx:]
            
            train_cols.extend(cat_train)
            test_cols.extend(cat_test)
            
            self.logger.debug(
                f"{category_name}: {len(cat_train)} train, "
                f"{len(cat_test)} test"
            )
        
        # Shuffle final lists to mix categories
        np.random.shuffle(train_cols)
        np.random.shuffle(test_cols)
        
        return train_cols, test_cols
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info(f"Splitting dataset with test ratio: {self.test_ratio}")
        
        # Identify sample types
        control_cols, ad_cols, other_cols = self.identify_sample_types(df)
        
        self.logger.info(
            f"Sample types - Control: {len(control_cols)}, "
            f"AD: {len(ad_cols)}, Other: {len(other_cols)}"
        )
        
        # Create stratified split
        train_cols, test_cols = self.create_stratified_split(
            control_cols, ad_cols, other_cols, self.test_ratio
        )
        
        # Create train and test datasets
        train_df = df[train_cols]
        test_df = df[test_cols]
        
        self.logger.info(f"Train set shape: {train_df.shape}")
        self.logger.info(f"Test set shape: {test_df.shape}")
        
        # Calculate split statistics
        total_samples = len(train_cols) + len(test_cols)
        train_ratio = len(train_cols) / total_samples
        
        self.logger.info(
            f"Split: {len(train_cols)} train ({train_ratio:.1%}), "
            f"{len(test_cols)} test ({self.test_ratio:.1%})"
        )
        
        return train_df, test_df, train_cols, test_cols
    
    def save_dataset(self, 
                    df: pd.DataFrame, 
                    file_path: Path,
                    format: str = 'xlsx') -> bool:
        """
        Save dataset to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('xlsx' or 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'xlsx':
                df.to_excel(file_path)
            elif format.lower() == 'csv':
                df.to_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved dataset to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {str(e)}")
            return False
    
    def save_metadata(self, 
                     sample_name: str,
                     original_shape: Tuple[int, int],
                     train_shape: Tuple[int, int],
                     test_shape: Tuple[int, int],
                     train_cols: List[str],
                     test_cols: List[str],
                     control_cols: List[str],
                     ad_cols: List[str],
                     other_cols: List[str]) -> None:
        """
        Save metadata about the preprocessing.
        
        Args:
            sample_name: Name of the sample
            original_shape: Shape of original data
            train_shape: Shape of training data
            test_shape: Shape of testing data
            train_cols: Training column names
            test_cols: Testing column names
            control_cols: Control column names
            ad_cols: AD column names
            other_cols: Other column names
        """
        metadata = {
            'sample_name': sample_name,
            'processing_date': datetime.now().isoformat(),
            'test_ratio': self.test_ratio,
            'random_state': self.random_state,
            'shapes': {
                'original': original_shape,
                'train': train_shape,
                'test': test_shape
            },
            'column_counts': {
                'total_original': original_shape[1],
                'total_train': train_shape[1],
                'total_test': test_shape[1],
                'control': len(control_cols),
                'ad': len(ad_cols),
                'other': len(other_cols)
            },
            'split_ratios': {
                'train': train_shape[1] / original_shape[1],
                'test': test_shape[1] / original_shape[1]
            }
        }
        
        # Save as JSON
        json_path = self.metadata_dir / f"{sample_name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save column lists as CSV
        columns_df = pd.DataFrame({
            'sample_type': ['train'] * len(train_cols) + ['test'] * len(test_cols),
            'column_name': train_cols + test_cols,
            'condition': [
                self._get_condition(col, control_cols, ad_cols)
                for col in train_cols + test_cols
            ]
        })
        
        csv_path = self.metadata_dir / f"{sample_name}_columns.csv"
        columns_df.to_csv(csv_path, index=False)
        
        self.logger.debug(f"Metadata saved for {sample_name}")
    
    def _get_condition(self, 
                      column: str,
                      control_cols: List[str],
                      ad_cols: List[str]) -> str:
        """Determine condition for a column."""
        if column in control_cols:
            return 'control'
        elif column in ad_cols:
            return 'ad'
        else:
            return 'other'
    
    def process_single_file(self, file_path: Path) -> bool:
        """
        Process a single original data file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        sample_name = file_path.stem
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {sample_name}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Load data
            df = self.load_original_data(file_path)
            if df is None:
                self.logger.error(f"Failed to load {sample_name}")
                return False
            
            original_shape = df.shape
            
            # Identify sample types before splitting
            control_cols, ad_cols, other_cols = self.identify_sample_types(df)
            
            # Split dataset
            train_df, test_df, train_cols, test_cols = self.split_dataset(df)
            
            # Save datasets
            train_file = self.train_dir / f"{sample_name}_train.xlsx"
            test_file = self.test_dir / f"{sample_name}_test.xlsx"
            
            if not self.save_dataset(train_df, train_file):
                return False
            
            if not self.save_dataset(test_df, test_file):
                return False
            
            # Save metadata
            self.save_metadata(
                sample_name,
                original_shape,
                train_df.shape,
                test_df.shape,
                train_cols,
                test_cols,
                control_cols,
                ad_cols,
                other_cols
            )
            
            # Update statistics
            self.processing_stats['successful'] += 1
            self.processing_stats['processed_samples'].append({
                'sample': sample_name,
                'original_shape': original_shape,
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'control_samples': len(control_cols),
                'ad_samples': len(ad_cols)
            })
            
            self.logger.info(f"✓ Successfully processed {sample_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {sample_name}: {str(e)}")
            self.processing_stats['failed'] += 1
            self.processing_stats['failed_samples'].append({
                'sample': sample_name,
                'error': str(e)
            })
            return False
    
    def run_batch_processing(self) -> Dict:
        """
        Process all CSV files in the input directory.
        
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info("Starting batch processing of original data")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Test ratio: {self.test_ratio}")
        
        # Find all CSV files
        csv_files = list(self.input_dir.glob("*.csv"))
        self.processing_stats['total_files'] = len(csv_files)
        
        if not csv_files:
            self.logger.error(f"No CSV files found in {self.input_dir}")
            return self.processing_stats
        
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        # Process each file
        for i, file_path in enumerate(csv_files, 1):
            self.logger.info(f"\nProcessing file {i}/{len(csv_files)}")
            self.process_single_file(file_path)
        
        # Create summary report
        self.create_summary_report()
        
        return self.processing_stats
    
    def create_summary_report(self) -> None:
        """Create a summary report of the batch processing."""
        total = self.processing_stats['total_files']
        successful = self.processing_stats['successful']
        failed = self.processing_stats['failed']
        
        success_rate = successful / total if total > 0 else 0
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'test_ratio': self.test_ratio,
            'random_state': self.random_state,
            'total_files': total,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'successful_samples': [
                s['sample'] for s in self.processing_stats['processed_samples']
            ],
            'failed_samples': [
                s['sample'] for s in self.processing_stats['failed_samples']
            ]
        }
        
        # Save summary as JSON
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed statistics as CSV
        if self.processing_stats['processed_samples']:
            detailed_df = pd.DataFrame(self.processing_stats['processed_samples'])
            detailed_path = self.output_dir / "detailed_statistics.csv"
            detailed_df.to_csv(detailed_path, index=False)
        
        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total files: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Success rate: {success_rate:.1%}")
        
        if failed > 0:
            self.logger.info("\nFailed samples:")
            for sample in self.processing_stats['failed_samples']:
                self.logger.info(f"  - {sample['sample']}: {sample['error']}")
    
    def validate_outputs(self) -> Dict[str, Dict]:
        """
        Validate the generated train and test files.
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("\nValidating output files...")
        
        validation_results = {}
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
                'row_consistency': None,
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
                    
                    # Check row consistency
                    train_rows = set(train_df.index)
                    test_rows = set(test_df.index)
                    row_consistency = train_rows == test_rows
                    
                    validation['row_consistency'] = row_consistency
                    
                    # Validate
                    if len(column_overlap) == 0 and row_consistency:
                        validation['validation_passed'] = True
                        self.logger.info(f"✓ {sample_name}: Validation passed")
                    else:
                        self.logger.warning(f"✗ {sample_name}: Validation issues")
                        if len(column_overlap) > 0:
                            self.logger.warning(f"  Column overlap: {column_overlap}")
                        if not row_consistency:
                            self.logger.warning(f"  Row mismatch: {len(train_rows)} vs {len(test_rows)}")
                
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
    """Command-line interface for original data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess original gene expression data (no synthetic data)'
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing CSV files')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--test-ratio', '-t', type=float, default=0.3,
                       help='Test set ratio (default: 0.3)')
    parser.add_argument('--random-state', '-r', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate outputs after preprocessing')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Create and run preprocessor
    preprocessor = OriginalDataPreprocessor(
        input_dir=args.input,
        output_dir=args.output,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        log_level=args.log_level
    )
    
    # Run preprocessing
    stats = preprocessor.run_batch_processing()
    
    # Validate if requested
    if args.validate:
        validation_results = preprocessor.validate_outputs()
        
        # Calculate validation success rate
        successful = sum(1 for v in validation_results.values() 
                        if v.get('validation_passed', False))
        total = len(validation_results)
        
        if total > 0:
            print(f"\nValidation: {successful}/{total} samples passed ({successful/total:.1%})")
    
    return stats


if __name__ == "__main__":
    main()
