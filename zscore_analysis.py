"""
Log Z-score Calculation for Gene Feature Importance Analysis

This module calculates log-normalized Z-scores for gene feature importance
based on Random Forest analysis outputs.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


def setup_logging(log_dir: Path = None):
    """Setup logging configuration."""
    if log_dir:
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "zscore_analysis.log"
    else:
        log_file = None
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_gene_importance_data(directory_path: Path) -> Optional[pd.DataFrame]:
    """
    Load all gene importance CSV files from directory.
    
    Args:
        directory_path: Path to directory containing CSV files
        
    Returns:
        Combined DataFrame with gene importance data or None if no data
    """
    logger = logging.getLogger(__name__)
    
    # Get all CSV files
    csv_files = list(directory_path.glob("*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in the specified directory")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    all_data = []
    failed_files = []
    
    for file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            logger.debug(f"File {file.name} columns: {df.columns.tolist()}")
            
            # Check data columns
            if len(df.columns) < 2:
                logger.warning(f"File {file.name} has insufficient columns, skipping")
                failed_files.append((file.name, "insufficient columns"))
                continue
            
            # Identify gene names and importance columns
            # Assuming first column is gene names, second is feature importance
            gene_col = df.columns[0]
            importance_col = df.columns[1]
            
            # Create clean DataFrame
            clean_df = pd.DataFrame({
                'gene_names': df[gene_col].astype(str).str.strip(),
                'feature_importance': pd.to_numeric(df[importance_col], errors='coerce')
            })
            
            # Remove NaN values and zeros
            clean_df = clean_df.dropna(subset=['feature_importance'])
            clean_df = clean_df[clean_df['feature_importance'] > 0]
            
            if len(clean_df) == 0:
                logger.warning(f"File {file.name} has no valid data after cleaning")
                failed_files.append((file.name, "no valid data"))
                continue
            
            # Add metadata
            clean_df['source_file'] = file.stem
            clean_df['sample_name'] = extract_sample_name(file.stem)
            
            all_data.append(clean_df)
            logger.info(f"Processed {file.name}: {len(clean_df)} valid rows")
            
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
            failed_files.append((file.name, str(e)))
            continue
    
    if not all_data:
        logger.error("No valid data was successfully read from any file")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Log summary statistics
    logger.info(f"Total combined data: {len(combined_data)} rows")
    logger.info(f"Number of unique genes: {combined_data['gene_names'].nunique()}")
    logger.info(f"Number of source files: {combined_data['source_file'].nunique()}")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        save_failed_files_log(failed_files, directory_path)
    
    return combined_data


def extract_sample_name(filename: str) -> str:
    """
    Extract sample name from filename.
    
    Args:
        filename: Input filename
        
    Returns:
        Extracted sample name
    """
    # Remove common suffixes
    suffixes = ['_characteristics', '_weights', '_loss', '_train', '_test']
    for suffix in suffixes:
        if suffix in filename:
            filename = filename.split(suffix)[0]
    
    # Remove autoencoder model types
    model_types = ['_autoencoder', '_vae', '_denoising', '_sparse']
    for model_type in model_types:
        if model_type in filename:
            filename = filename.replace(model_type, '')
    
    return filename


def save_failed_files_log(failed_files: List[Tuple], directory_path: Path):
    """Save log of failed file processing attempts."""
    log_file = directory_path / "failed_files_log.csv"
    failed_df = pd.DataFrame(failed_files, columns=['filename', 'error'])
    failed_df.to_csv(log_file, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Failed files log saved to {log_file}")


def calculate_log_zscore(combined_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculate log Z-scores based on log-normal distribution.
    
    Args:
        combined_data: Combined DataFrame with gene importance data
        
    Returns:
        DataFrame with calculated Z-scores or None if calculation fails
    """
    logger = logging.getLogger(__name__)
    
    if combined_data is None or len(combined_data) == 0:
        logger.error("No data provided for Z-score calculation")
        return None
    
    # Apply logarithmic transformation
    logger.info("Applying logarithmic transformation...")
    combined_data['log_importance'] = np.log(combined_data['feature_importance'])
    
    try:
        # Method 1: Z-score based on individual log values
        global_mean_log = np.mean(combined_data['log_importance'])
        global_std_log = np.std(combined_data['log_importance'], ddof=1)
        
        logger.info(f"Global mean of log importance: {global_mean_log:.4f}")
        logger.info(f"Global std of log importance: {global_std_log:.4f}")
        
        # Calculate individual Z-scores
        combined_data['log_z_score_individual'] = (
            (combined_data['log_importance'] - global_mean_log) / global_std_log
        )
        
        # Method 2: Z-score based on gene mean log values
        gene_log_means = combined_data.groupby('gene_names')['log_importance'].agg(
            ['mean', 'std', 'count']
        ).reset_index()
        gene_log_means.columns = ['gene_names', 'mean_log_importance', 'std_log_importance', 'gene_count']
        
        # Calculate Z-score for gene means
        mean_of_means = np.mean(gene_log_means['mean_log_importance'])
        std_of_means = np.std(gene_log_means['mean_log_importance'], ddof=1)
        gene_log_means['log_z_score_gene_mean'] = (
            (gene_log_means['mean_log_importance'] - mean_of_means) / std_of_means
        )
        
        # Merge back with original data
        combined_data = pd.merge(
            combined_data,
            gene_log_means[['gene_names', 'log_z_score_gene_mean']],
            on='gene_names',
            how='left'
        )
        
        # Create summary statistics for each gene
        summary_stats = combined_data.groupby('gene_names').agg({
            'log_z_score_individual': ['mean', 'std'],
            'log_z_score_gene_mean': 'first',
            'feature_importance': ['count', 'mean', 'std', 'min', 'max'],
            'log_importance': ['mean', 'std'],
            'source_file': lambda x: ', '.join(sorted(set(x))),
            'sample_name': lambda x: ', '.join(sorted(set(x)))
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = [
            'zscore_individual_mean', 'zscore_individual_std',
            'zscore_gene_mean',
            'count', 'importance_mean', 'importance_std', 'importance_min', 'importance_max',
            'log_mean', 'log_std',
            'source_files', 'samples'
        ]
        
        result_df = summary_stats.reset_index()
        
        # Calculate additional statistics
        result_df['importance_cv'] = (result_df['importance_std'] / result_df['importance_mean']).round(4)
        result_df['log_cv'] = (result_df['log_std'] / result_df['log_mean']).round(4)
        
        # Sort by absolute Z-score (gene mean method)
        result_df['abs_zscore'] = np.abs(result_df['zscore_gene_mean'])
        result_df = result_df.sort_values('abs_zscore', ascending=False)
        
        logger.info("Z-score calculation completed successfully")
        logger.info(f"Processed {len(result_df)} unique genes")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating log Z-scores: {str(e)}")
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return None


def create_distribution_plots(combined_data: pd.DataFrame, 
                            result_df: pd.DataFrame, 
                            savepath: Path):
    """
    Create distribution visualization plots.
    
    Args:
        combined_data: Raw combined data
        result_df: Results DataFrame with Z-scores
        savepath: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    savepath.mkdir(exist_ok=True)
    
    try:
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Original feature importance distribution
        axes[0, 0].hist(combined_data['feature_importance'], 
                       bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Original Feature Importance Distribution')
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Log-transformed distribution
        axes[0, 1].hist(combined_data['log_importance'], 
                       bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Log-Transformed Distribution')
        axes[0, 1].set_xlabel('log(Feature Importance)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. QQ plot for normality check
        stats.probplot(combined_data['log_importance'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot for Normality Check')
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        
        # 4. Gene mean Z-score distribution
        axes[1, 1].hist(result_df['zscore_gene_mean'], 
                       bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=1.96, color='green', linestyle=':', alpha=0.5)
        axes[1, 1].axvline(x=-1.96, color='green', linestyle=':', alpha=0.5)
        axes[1, 1].set_title('Gene Mean Z-score Distribution')
        axes[1, 1].set_xlabel('Z-score (Gene Means)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_path = savepath / "distribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create gene ranking plot
        top_genes = result_df.head(20).copy()
        plt.figure(figsize=(12, 8))
        
        # Color by significance
        colors = []
        for score in top_genes['zscore_gene_mean']:
            if abs(score) >= 2.58:
                colors.append('darkred' if score > 0 else 'darkblue')  # p < 0.01
            elif abs(score) >= 1.96:
                colors.append('red' if score > 0 else 'blue')  # p < 0.05
            else:
                colors.append('lightcoral' if score > 0 else 'lightblue')
        
        bars = plt.barh(range(len(top_genes)), top_genes['zscore_gene_mean'], 
                       color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_genes['zscore_gene_mean'])):
            plt.text(score, i, f'{score:.2f}', 
                    va='center', ha='left' if score > 0 else 'right',
                    fontweight='bold')
        
        plt.yticks(range(len(top_genes)), top_genes['gene_names'])
        plt.xlabel('Log Z-score (Based on Gene Means)')
        plt.title('Top 20 Genes by Absolute Z-score')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        plt.axvline(x=1.96, color='green', linestyle=':', alpha=0.5, label='p=0.05')
        plt.axvline(x=-1.96, color='green', linestyle=':', alpha=0.5)
        plt.axvline(x=2.58, color='red', linestyle=':', alpha=0.5, label='p=0.01')
        plt.axvline(x=-2.58, color='red', linestyle=':', alpha=0.5)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        rank_plot_path = savepath / "gene_ranking.png"
        plt.savefig(rank_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create sample-wise heatmap of top genes
        create_sample_heatmap(combined_data, result_df, savepath)
        
        logger.info(f"Plots saved to {savepath}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")


def create_sample_heatmap(combined_data: pd.DataFrame, 
                         result_df: pd.DataFrame, 
                         savepath: Path):
    """
    Create heatmap showing gene importance across samples.
    
    Args:
        combined_data: Raw combined data
        result_df: Results DataFrame with Z-scores
        savepath: Directory to save plots
    """
    top_genes = result_df.head(30)['gene_names'].tolist()
    
    # Pivot data for heatmap
    heatmap_data = combined_data[combined_data['gene_names'].isin(top_genes)]
    heatmap_data = heatmap_data.pivot_table(
        index='gene_names',
        columns='sample_name',
        values='log_importance',
        aggfunc='mean'
    )
    
    # Sort rows by Z-score
    gene_order = result_df[result_df['gene_names'].isin(top_genes)]
    gene_order = gene_order.set_index('gene_names').loc[top_genes]
    heatmap_data = heatmap_data.loc[gene_order.index]
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, 
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'log(Feature Importance)'},
                square=True,
                linewidths=0.5)
    plt.title('Top 30 Genes - Log Feature Importance Across Samples')
    plt.tight_layout()
    heatmap_path = savepath / "sample_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()


def validate_lognormal_fit(combined_data: pd.DataFrame, savepath: Path) -> pd.DataFrame:
    """
    Validate lognormal distribution fit for each dataset.
    
    Args:
        combined_data: Combined data with source_file column
        savepath: Directory to save validation results
        
    Returns:
        DataFrame with validation statistics
    """
    logger = logging.getLogger(__name__)
    
    datasets = combined_data['source_file'].unique()
    fit_results = []
    
    for dataset in datasets:
        data = combined_data[combined_data['source_file'] == dataset]['feature_importance']
        
        if len(data) < 8:  # Minimum sample size for reliable tests
            fit_results.append({
                'dataset': dataset,
                'n_genes': len(data),
                'shapiro_statistic': np.nan,
                'shapiro_p_value': np.nan,
                'normality': 'insufficient_data',
                'skewness': np.nan,
                'kurtosis': np.nan,
                'mean_log': np.nan,
                'std_log': np.nan
            })
            continue
        
        # Apply log transformation
        log_data = np.log(data)
        
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(log_data)
        
        # Additional normality tests
        k2_stat, k2_p = stats.normaltest(log_data)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(log_data)
        kurtosis = stats.kurtosis(log_data)
        
        # Estimate lognormal distribution parameters
        mean_log = np.mean(log_data)
        std_log = np.std(log_data, ddof=1)
        
        # Determine normality status
        if shapiro_p > 0.05 and k2_p > 0.05:
            normality = 'normal'
        elif shapiro_p > 0.05 or k2_p > 0.05:
            normality = 'borderline'
        else:
            normality = 'non-normal'
        
        fit_results.append({
            'dataset': dataset,
            'n_genes': len(data),
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'dagostino_k2_stat': k2_stat,
            'dagostino_k2_p': k2_p,
            'normality': normality,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean_log': mean_log,
            'std_log': std_log
        })
    
    # Create validation DataFrame
    validation_df = pd.DataFrame(fit_results)
    validation_path = savepath / "lognormal_validation.csv"
    validation_df.to_csv(validation_path, index=False)
    
    # Summary statistics
    normal_count = validation_df[validation_df['normality'] == 'normal'].shape[0]
    borderline_count = validation_df[validation_df['normality'] == 'borderline'].shape[0]
    
    logger.info(f"Lognormal validation: {normal_count} normal, {borderline_count} borderline")
    logger.info(f"Validation results saved to {validation_path}")
    
    return validation_df


def run_zscore_analysis(input_dir: Path, output_dir: Path) -> Tuple[Optional[pd.DataFrame], 
                                                                  Optional[pd.DataFrame]]:
    """
    Main function to run complete Z-score analysis.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save results
        
    Returns:
        Tuple of (result_df, validation_df) or (None, None) on failure
    """
    logger = logging.getLogger(__name__)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting Z-score analysis")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    combined_data = load_gene_importance_data(input_dir)
    if combined_data is None:
        logger.error("Failed to load data. Exiting.")
        return None, None
    
    # Save raw combined data
    raw_data_path = output_dir / "raw_combined_data.csv"
    combined_data.to_csv(raw_data_path, index=False)
    logger.info(f"Raw combined data saved to {raw_data_path}")
    
    # Step 2: Calculate Z-scores
    result_df = calculate_log_zscore(combined_data)
    if result_df is None:
        logger.error("Failed to calculate Z-scores. Exiting.")
        return None, None
    
    # Step 3: Save results
    result_path = output_dir / "gene_zscore_results.csv"
    result_df.to_csv(result_path, index=False)
    logger.info(f"Results saved to {result_path}")
    
    # Step 4: Create summary report
    create_summary_report(result_df, output_dir)
    
    # Step 5: Create visualizations
    create_distribution_plots(combined_data, result_df, output_dir)
    
    # Step 6: Validate lognormal fit
    validation_df = validate_lognormal_fit(combined_data, output_dir)
    
    logger.info("Z-score analysis completed successfully")
    
    return result_df, validation_df


def create_summary_report(result_df: pd.DataFrame, output_dir: Path):
    """
    Create a summary report of the analysis.
    
    Args:
        result_df: Results DataFrame
        output_dir: Directory to save report
    """
    summary = {
        'total_genes': len(result_df),
        'genes_with_multiple_observations': (result_df['count'] > 1).sum(),
        'mean_zscore': result_df['zscore_gene_mean'].mean(),
        'std_zscore': result_df['zscore_gene_mean'].std(),
        'genes_above_2_sd': (abs(result_df['zscore_gene_mean']) > 1.96).sum(),
        'genes_above_3_sd': (abs(result_df['zscore_gene_mean']) > 2.58).sum(),
        'top_gene': result_df.iloc[0]['gene_names'],
        'top_zscore': result_df.iloc[0]['zscore_gene_mean'],
        'min_importance': result_df['importance_mean'].min(),
        'max_importance': result_df['importance_mean'].max(),
        'mean_importance': result_df['importance_mean'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Also save as markdown for easier reading
    md_path = output_dir / "analysis_summary.md"
    with open(md_path, 'w') as f:
        f.write("# Z-score Analysis Summary\n\n")
        f.write("## Key Statistics\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"- **{key}**: {value:.4f}\n")
            else:
                f.write(f"- **{key}**: {value}\n")
        
        f.write("\n## Top 10 Genes\n\n")
        f.write("| Gene | Z-score | Count | Mean Importance |\n")
        f.write("|------|---------|-------|----------------|\n")
        for _, row in result_df.head(10).iterrows():
            f.write(f"| {row['gene_names']} | {row['zscore_gene_mean']:.4f} | "
                   f"{row['count']} | {row['importance_mean']:.6f} |\n")


def main():
    """Command-line interface for Z-score analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate log Z-scores for gene feature importance'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing CSV files')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--log', '-l', type=str, default=None,
                       help='Log directory (default: output directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    log_dir = Path(args.log) if args.log else output_dir / "logs"
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "zscore_analysis.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Run analysis
    try:
        result_df, validation_df = run_zscore_analysis(input_dir, output_dir)
        
        if result_df is not None:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"\nResults saved to: {output_dir}")
            print(f"\nTop 5 genes by absolute Z-score:")
            print(result_df[['gene_names', 'zscore_gene_mean', 'count', 
                           'importance_mean']].head().to_string(index=False))
        else:
            print("\nAnalysis failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
