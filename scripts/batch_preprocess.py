#!/usr/bin/env python3
"""
Batch preprocessing script for multiple datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_preprocessing import DataPreprocessor
import yaml
import pandas as pd


def load_config(config_path: Path):
    """Load preprocessing configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_info(input_dirs, output_base):
    """Create dataset information file."""
    datasets = []
    
    for dataset_name, paths in input_dirs.items():
        original_dir = Path(paths['original'])
        synthetic_dir = Path(paths['synthetic'])
        output_dir = Path(output_base) / dataset_name
        
        # Count files
        original_files = list(original_dir.glob("*.csv"))
        synthetic_ad_files = list(synthetic_dir.glob("*_synthetic_AD.xlsx"))
        
        datasets.append({
            'dataset_name': dataset_name,
            'original_dir': str(original_dir),
            'synthetic_dir': str(synthetic_dir),
            'output_dir': str(output_dir),
            'original_files': len(original_files),
            'synthetic_ad_files': len(synthetic_ad_files),
            'status': 'pending'
        })
    
    return pd.DataFrame(datasets)


def process_datasets(config_path: Path, dataset_config_path: Path):
    """Process multiple datasets based on configuration."""
    
    # Load configurations
    config = load_config(config_path)
    dataset_config = load_config(dataset_config_path)
    
    # Create dataset info
    dataset_info = create_dataset_info(
        dataset_config['datasets'],
        dataset_config['output_base']
    )
    
    print("Dataset processing plan:")
    print(dataset_info[['dataset_name', 'original_files', 'synthetic_ad_files', 'status']])
    print("\n" + "="*60)
    
    results = {}
    
    for _, row in dataset_info.iterrows():
        print(f"\nProcessing dataset: {row['dataset_name']}")
        print("-" * 40)
        
        # Create preprocessor
        preprocessor = DataPreprocessor(
            original_dir=row['original_dir'],
            synthetic_dir=row['synthetic_dir'],
            output_dir=row['output_dir'],
            test_ratio=config['preprocessing']['splitting']['test_ratio'],
            random_state=config['preprocessing']['splitting']['random_state']
        )
        
        # Run preprocessing
        try:
            dataset_results = preprocessor.run()
            
            # Validate
            if config['validation']['enabled']:
                preprocessor.validate_outputs()
            
            # Update status
            success_count = sum(1 for v in dataset_results.values() if v)
            total_count = len(dataset_results)
            
            results[row['dataset_name']] = {
                'success': success_count,
                'total': total_count,
                'success_rate': success_count / total_count if total_count > 0 else 0
            }
            
            print(f"  ✓ Completed: {success_count}/{total_count} samples")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results[row['dataset_name']] = {
                'success': 0,
                'total': 0,
                'success_rate': 0,
                'error': str(e)
            }
    
    # Create results summary
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'dataset'
    
    output_base = Path(dataset_config['output_base'])
    results_path = output_base / "batch_processing_results.csv"
    results_df.to_csv(results_path)
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_path}")
    print("\nSummary:")
    print(results_df[['success', 'total', 'success_rate']])
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch preprocessing for multiple datasets'
    )
    
    parser.add_argument('--config', '-c', type=str, default='config/preprocessing_config.yaml',
                       help='Preprocessing configuration file')
    parser.add_argument('--datasets', '-d', type=str, default='config/datasets.yaml',
                       help='Dataset configuration file')
    
    args = parser.parse_args()
    
    process_datasets(
        Path(args.config),
        Path(args.datasets)
    )
