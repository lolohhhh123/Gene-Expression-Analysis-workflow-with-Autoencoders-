#!/usr/bin/env python3
"""
Quick preprocessing for single dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_preprocessing import process_sample


def quick_preprocess(original_file: str,
                     synthetic_ad_file: str,
                     synthetic_control_file: str,
                     output_dir: str = "./processed",
                     sample_name: str = None):
    """
    Quick preprocessing for a single sample.
    
    Args:
        original_file: Path to original CSV file
        synthetic_ad_file: Path to synthetic AD file
        synthetic_control_file: Path to synthetic control file
        output_dir: Output directory
        sample_name: Name for the sample (default: stem of original file)
    """
    original_path = Path(original_file)
    synthetic_ad_path = Path(synthetic_ad_file)
    synthetic_control_path = Path(synthetic_control_file)
    output_dir = Path(output_dir)
    
    if sample_name is None:
        sample_name = original_path.stem
    
    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    print(f"Processing: {sample_name}")
    print(f"Original: {original_path}")
    print(f"Synthetic AD: {synthetic_ad_path}")
    print(f"Synthetic Control: {synthetic_control_path}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    success = process_sample(
        original_path,
        synthetic_ad_path,
        synthetic_control_path,
        train_dir,
        test_dir,
        sample_name,
        test_ratio=0.5,
        random_state=42
    )
    
    if success:
        print(f"\n✓ Successfully processed {sample_name}")
        print(f"  Training data: {train_dir}/{sample_name}_train.xlsx")
        print(f"  Testing data: {test_dir}/{sample_name}_test.xlsx")
    else:
        print(f"\n✗ Failed to process {sample_name}")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Quick preprocessing for single sample'
    )
    
    parser.add_argument('original', type=str,
                       help='Original CSV file')
    parser.add_argument('synthetic_ad', type=str,
                       help='Synthetic AD XLSX file')
    parser.add_argument('synthetic_control', type=str,
                       help='Synthetic Control XLSX file')
    parser.add_argument('--output', '-o', type=str, default='./processed',
                       help='Output directory')
    parser.add_argument('--name', '-n', type=str,
                       help='Sample name (default: based on original filename)')
    
    args = parser.parse_args()
    
    quick_preprocess(
        args.original,
        args.synthetic_ad,
        args.synthetic_control,
        args.output,
        args.name
    )
