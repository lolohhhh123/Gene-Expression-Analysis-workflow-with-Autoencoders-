"""
Integrated analysis pipeline combining autoencoder and Z-score analysis.
"""

from pathlib import Path
import pandas as pd
from utils.zscore_calculator import ZScoreCalculator


def run_integrated_analysis(autoencoder_dir: Path, output_dir: Path = None):
    """
    Run complete analysis pipeline: Autoencoder → Z-score → Integration.
    
    Args:
        autoencoder_dir: Directory with autoencoder results
        output_dir: Output directory (optional)
    """
    output_dir = output_dir or autoencoder_dir / "integrated_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("INTEGRATED GENE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Step 1: Run Z-score analysis
    print("\n1. Running Z-score analysis...")
    calculator = ZScoreCalculator(output_dir)
    zscore_results = calculator.calculate_from_directory(autoencoder_dir)
    
    if zscore_results is None:
        print("No valid data found for Z-score analysis")
        return
    
    print(f"   ✓ Analyzed {len(zscore_results)} genes")
    
    # Step 2: Integrate with autoencoder results
    print("\n2. Integrating with autoencoder results...")
    calculator.integrate_with_autoencoder_results(autoencoder_dir)
    
    # Step 3: Create summary
    print("\n3. Creating summary report...")
    create_summary_report(zscore_results, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # Show top results
    print("\nTOP SIGNIFICANT GENES:")
    top_genes = zscore_results.head(10)
    for i, (_, row) in enumerate(top_genes.iterrows(), 1):
        significance = "***" if abs(row['z_score']) > 2.58 else "**" if abs(row['z_score']) > 1.96 else "*"
        print(f"{i:2d}. {row['gene']:20s} Z={row['z_score']:6.2f} {significance}")
    
    return zscore_results


def create_summary_report(zscore_results: pd.DataFrame, output_dir: Path):
    """Create a comprehensive summary report."""
    summary = {
        'total_genes': len(zscore_results),
        'high_significance': (abs(zscore_results['z_score']) > 2.58).sum(),
        'medium_significance': ((abs(zscore_results['z_score']) > 1.96) & 
                               (abs(zscore_results['z_score']) <= 2.58)).sum(),
        'mean_zscore': zscore_results['z_score'].mean(),
        'std_zscore': zscore_results['z_score'].std(),
        'top_gene': zscore_results.iloc[0]['gene'],
        'top_zscore': zscore_results.iloc[0]['z_score']
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "analysis_summary.csv", index=False)
    
    # Save full results
    zscore_results.to_csv(output_dir / "full_zscore_results.csv", index=False)
    
    # Save gene lists by significance
    for threshold, label in [(2.58, "high_significance"), 
                            (1.96, "medium_significance")]:
        genes = zscore_results[abs(zscore_results['z_score']) > threshold]
        if len(genes) > 0:
            genes[['gene', 'z_score']].to_csv(
                output_dir / f"genes_{label}.csv", index=False
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrated autoencoder and Z-score analysis'
    )
    parser.add_argument('input', type=str, 
                       help='Directory with autoencoder results')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory (default: input/integrated_analysis)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "integrated_analysis"
    
    run_integrated_analysis(input_dir, output_dir)
