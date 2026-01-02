"""
Simplified Z-score calculator for integration with autoencoder project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class ZScoreCalculator:
    """Simple Z-score calculator for gene importance analysis."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("./results")
        self.results_dir.mkdir(exist_ok=True)
    
    def calculate_from_directory(self, input_dir: Path) -> Optional[pd.DataFrame]:
        """Calculate Z-scores from CSV files in directory."""
        import_files = list(input_dir.glob("*_characteristics.csv"))
        
        if not import_files:
            print("No characteristic files found")
            return None
        
        all_data = []
        
        for file in import_files:
            try:
                df = pd.read_csv(file)
                
                # Extract relevant columns
                if len(df.columns) >= 2:
                    gene_col = df.columns[0]
                    imp_col = df.columns[1]
                    
                    data = pd.DataFrame({
                        'gene': df[gene_col],
                        'importance': pd.to_numeric(df[imp_col], errors='coerce')
                    }).dropna()
                    
                    if len(data) > 0:
                        data['source'] = file.stem
                        all_data.append(data)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not all_data:
            return None
        
        # Combine and calculate
        combined = pd.concat(all_data, ignore_index=True)
        result = self._calculate_zscore(combined)
        
        # Save results
        output_file = self.results_dir / "gene_zscore_summary.csv"
        result.to_csv(output_file, index=False)
        
        # Save top genes
        top_genes = result.head(50)
        top_genes.to_csv(self.results_dir / "top_genes_zscore.csv", index=False)
        
        return result
    
    def _calculate_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Z-scores for combined data."""
        # Log transformation
        data['log_imp'] = np.log(data['importance'])
        
        # Gene-level aggregation
        gene_stats = data.groupby('gene').agg({
            'log_imp': ['mean', 'std', 'count'],
            'importance': 'mean'
        }).round(4)
        
        gene_stats.columns = ['log_mean', 'log_std', 'count', 'imp_mean']
        gene_stats = gene_stats.reset_index()
        
        # Calculate Z-score
        overall_mean = gene_stats['log_mean'].mean()
        overall_std = gene_stats['log_mean'].std()
        
        gene_stats['z_score'] = (gene_stats['log_mean'] - overall_mean) / overall_std
        gene_stats['abs_z_score'] = abs(gene_stats['z_score'])
        
        # Sort and add ranking
        gene_stats = gene_stats.sort_values('abs_z_score', ascending=False)
        gene_stats['rank'] = range(1, len(gene_stats) + 1)
        
        return gene_stats[['rank', 'gene', 'z_score', 'abs_z_score', 
                          'count', 'imp_mean', 'log_mean', 'log_std']]
    
    def integrate_with_autoencoder_results(self, autoencoder_dir: Path):
        """Integrate Z-scores with autoencoder feature importance."""
        zscore_results = self.calculate_from_directory(autoencoder_dir)
        
        if zscore_results is not None:
            # Load autoencoder weights if available
            weight_files = list(autoencoder_dir.glob("*_weights.csv"))
            
            for weight_file in weight_files:
                try:
                    weights = pd.read_csv(weight_file, index_col=0)
                    sample_name = weight_file.stem.replace('_weights', '')
                    
                    # Merge with Z-scores
                    weights['gene'] = weights.index
                    merged = pd.merge(
                        weights.reset_index(),
                        zscore_results[['gene', 'z_score']],
                        on='gene',
                        how='left'
                    ).set_index('gene')
                    
                    # Save integrated results
                    merged.to_csv(
                        self.results_dir / f"{sample_name}_integrated_importance.csv"
                    )
                    
                except Exception as e:
                    print(f"Error processing {weight_file}: {e}")


# Quick usage function
def quick_zscore_analysis(input_path: str, output_path: str = None):
    """
    Quick function for Z-score analysis.
    
    Args:
        input_path: Path to directory with CSV files
        output_path: Output directory (optional)
    """
    calculator = ZScoreCalculator(Path(output_path) if output_path else None)
    results = calculator.calculate_from_directory(Path(input_path))
    
    if results is not None:
        print(f"Analysis complete. Processed {len(results)} genes.")
        print("\nTop 10 genes:")
        print(results.head(10)[['gene', 'z_score', 'count']].to_string(index=False))
        
        # Save to current directory if no output specified
        if output_path is None:
            results.to_csv("zscore_results.csv", index=False)
            print("\nResults saved to 'zscore_results.csv'")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        quick_zscore_analysis(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("Usage: python zscore_calculator.py <input_directory> [output_directory]")
