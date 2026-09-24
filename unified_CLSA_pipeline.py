#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_CLSA_pipeline.py

Complete unified CLSA pipeline:
1. Load all datasets and filter small datasets (min_samples >= 6).
2. Batch correction (ComBat-style): remove technical differences across datasets.
3. Jointly train the FiLM-DAE-Lin model on all data.
4. Compute the unified Conditional Latent Shift (CLS).
5. Validate correlation between CLS and observed log2FC.
6. Run GO/KEGG enrichment analysis.
7. Export gene lists for STRING PPI analysis.

Usage:
python unified_CLSA_pipeline.py --input_dir /path/to/data --output_dir ./unified_results --device cuda
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

try:
    import gseapy as gp
except ImportError:
    gp = None
    print("gseapy not installed. Enrichment analysis will be skipped.")
    print("Install with: pip install gseapy")

# ============================================================================
# Model definitions
# ============================================================================

class FiLMLayer(nn.Module):
    """FiLM layer: apply an affine transformation conditioned on parameters."""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, h, gamma, beta):
        return gamma * h + beta


class FiLMGenerator(nn.Module):
    """Generate FiLM parameters (gamma, beta)."""
    def __init__(self, cond_dim, hidden_dim, hidden_dims_list):
        super().__init__()
        self.hidden_dims_list = hidden_dims_list
        total_output_dim = sum(hidden_dims_list) * 2
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_output_dim)
        )

    def forward(self, cond):
        params = self.net(cond)
        batch = params.size(0)
        outputs = []
        start = 0
        for hdim in self.hidden_dims_list:
            gamma = params[:, start:start + hdim]
            beta = params[:, start + hdim:start + 2 * hdim]
            outputs.append((gamma, beta))
            start += 2 * hdim
        return outputs


class UnifiedFiLM_DAE_Lin(nn.Module):
    """
    Unified FiLM-DAE-Lin model.
    Condition variables: [dataset_id_embedding, disease_status].
    """
    def __init__(self, input_dim, n_datasets, latent_dim=16,
                 hidden_dims=[64, 32], cond_dim=8, film_hidden=16, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_datasets = n_datasets

        # Condition embedding: dataset_id + disease_status
        self.dataset_embedding = nn.Embedding(n_datasets, cond_dim)
        self.cond_dim = cond_dim + 1

        self.film_gen = FiLMGenerator(self.cond_dim, film_hidden, hidden_dims)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.ModuleList([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                FiLMLayer(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            self.encoder_blocks.append(block)
            prev_dim = h_dim

        self.to_latent = nn.Linear(hidden_dims[-1], latent_dim)

        # Linear decoder: directly interpret genes
        self.decoder_weight = nn.Parameter(torch.Tensor(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.decoder_weight)

        # Classifier for disease_status
        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, x, dataset_id, disease_status):
        """Encode x -> z using FiLM conditioning."""
        dataset_emb = self.dataset_embedding(dataset_id.long())
        status = disease_status.float().unsqueeze(1)
        cond = torch.cat([dataset_emb, status], dim=1)

        film_params = self.film_gen(cond)
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = block[0](h)
            h = block[1](h)
            gamma, beta = film_params[i]
            h = block[2](h, gamma, beta)
            h = block[3](h)
            h = block[4](h)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        """Decode z -> x_hat with a linear decoder."""
        return torch.mm(z, self.decoder_weight.T)

    def forward(self, x, dataset_id, disease_status):
        z = self.encode(x, dataset_id, disease_status)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, logits, z


# ============================================================================
# Data loading and preprocessing
# ============================================================================

def load_single_dataset(file_path, normalize=True):
    """Load one dataset and return data, labels, and gene names."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Clean values
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Ensure samples are rows
    if df.shape[0] > df.shape[1] * 2:
        df = df.T

    # Drop empty rows/columns
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)

    # Drop all-zero rows/columns
    df = df.loc[(df != 0).any(axis=1)]
    df = df.loc[:, (df != 0).any(axis=0)]

    if df.empty:
        raise ValueError(f"Empty dataframe: {file_path}")

    # Merge duplicated gene names
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T

    gene_names = df.columns.tolist()
    sample_names = df.index.tolist()

    # Extract labels
    labels = []
    for name in sample_names:
        name_low = name.lower()
        if any(kw in name_low for kw in ['control', 'ctrl', 'wt', 'wild', 'normal', 'healthy', 'young']):
            labels.append(0)
        elif any(kw in name_low for kw in ['ad', 'alzheimer', 'tg', '3xtg', 'app', 'ps1', 'mutant', 'disease', 'old']):
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels, dtype=np.float32)

    if np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        raise ValueError(f"Only one class: Control={np.sum(labels == 0)}, AD={np.sum(labels == 1)}")

    data = df.values.astype(np.float32)

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = np.nan_to_num(data, nan=0.0)

    return data, labels, gene_names


def batch_correct_combat_style(data_list, dataset_names, epsilon=1e-8):
    """
    Simplified ComBat-style batch correction:
    1. Standardize each dataset to unit variance.
    2. Map each dataset to the global mean and variance.
    """
    print("Applying batch correction (ComBat-style)...")

    # Collect all data
    all_data = np.vstack(data_list)

    # Global mean and standard deviation
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0) + epsilon

    corrected_list = []
    for i, X in enumerate(data_list):
        # Dataset mean and standard deviation
        ds_mean = X.mean(axis=0)
        ds_std = X.std(axis=0) + epsilon

        # Standardize and map to global distribution
        X_norm = (X - ds_mean) / ds_std
        X_corr = X_norm * global_std + global_mean
        corrected_list.append(X_corr)

        print(f"  Batch corrected: {dataset_names[i]} ({len(X)} samples)")

    return np.vstack(corrected_list)


def load_and_align_datasets(input_dir, min_samples=6, batch_correct=True):
    """
    Load all datasets, filter small samples, align genes, and apply batch correction.
    """
    files = glob.glob(os.path.join(input_dir, "*.csv")) + \
            glob.glob(os.path.join(input_dir, "*.xlsx"))

    print(f"Found {len(files)} files. Loading with min_samples={min_samples}...")

    data_list = []
    labels_list = []
    dataset_ids_list = []
    dataset_names = []
    gene_sets = []
    gene_lists = []
    sample_counts = []

    # Note: enumerate(files) keeps the original file index.
    for idx, file_path in enumerate(files):
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            data, labels, gene_names = load_single_dataset(file_path, normalize=False)

            if len(data) < min_samples:
                print(f"  Skipping {dataset_name}: {len(data)} samples < {min_samples}")
                continue

            dataset_names.append(dataset_name)
            data_list.append(data)
            labels_list.append(labels)
            dataset_ids_list.append(np.full(len(data), idx, dtype=np.int64))
            gene_sets.append(set(gene_names))
            gene_lists.append(gene_names)
            sample_counts.append(len(data))

            print(f"  Loaded {dataset_name}: {len(data)} samples, {len(gene_names)} genes")

        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            continue

    if len(data_list) == 0:
        raise ValueError("No valid datasets loaded. Try reducing min_samples.")

    print(f"\nLoaded {len(data_list)} datasets. Total samples: {sum(sample_counts)}")

    # Find common genes
    common_genes = gene_sets[0]
    for gs in gene_sets[1:]:
        common_genes = common_genes.intersection(gs)
    common_genes = sorted(list(common_genes))
    print(f"Common genes across all datasets: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes: {len(common_genes)}")

    # Align all datasets to the common genes
    aligned_data = []
    for data, gene_names in zip(data_list, gene_lists):
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        idxs = [gene_to_idx[g] for g in common_genes]
        aligned_data.append(data[:, idxs])

    # Batch correction
    if batch_correct:
        X = batch_correct_combat_style(aligned_data, dataset_names)
    else:
        X = np.vstack(aligned_data)

    y = np.concatenate(labels_list)
    dataset_ids = np.concatenate(dataset_ids_list)

    # Remap dataset IDs to continuous indices
    unique_ids = np.unique(dataset_ids)
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    dataset_ids = np.array([id_map[old_id] for old_id in dataset_ids])
    n_datasets = len(unique_ids)

    print(f"\nRemapped {n_datasets} datasets to continuous IDs (0 to {n_datasets - 1})")
    print(f"Final: {X.shape[0]} samples, {X.shape[1]} genes, {len(dataset_names)} datasets")
    print(f"  Control: {np.sum(y == 0)}, AD: {np.sum(y == 1)}")

    return X, y, dataset_ids, dataset_names, common_genes


# ============================================================================
# Training
# ============================================================================

def loss_function(recon_x, x, logits, y, decoder_weight, alpha=1.0, beta=1e-4):
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')(logits.view(-1), y)
    l1_loss = torch.sum(torch.abs(decoder_weight))
    return recon_loss + alpha * bce_loss + beta * l1_loss


def train_unified_model(model, train_loader, val_loader, device,
                        epochs=300, lr=1e-3, alpha=1.0, beta=1e-3, patience=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by, bd in train_loader:
            bx, by, bd = bx.to(device), by.to(device), bd.to(device)
            optimizer.zero_grad()
            recon, logits, _ = model(bx, bd, by)
            loss = loss_function(recon, bx, logits, by, model.decoder_weight, alpha, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, bd in val_loader:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                recon, logits, _ = model(bx, bd, by)
                loss = loss_function(recon, bx, logits, by, model.decoder_weight, alpha, beta)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================================
# CLSA
# ============================================================================

def compute_cls_unified(W, Z, dataset_ids, y):
    """
    Compute the unified Conditional Latent Shift.
    Compute CLS per dataset and then take a sample-size-weighted average.
    """
    n_datasets = len(np.unique(dataset_ids))
    cls_list = []
    weights = []
    delta_z_list = []

    for d_id in range(n_datasets):
        mask = dataset_ids == d_id
        Z_d = Z[mask]
        y_d = y[mask]

        if np.sum(y_d == 0) < 3 or np.sum(y_d == 1) < 3:
            print(f"  Dataset {d_id}: insufficient samples (C={np.sum(y_d == 0)}, AD={np.sum(y_d == 1)}), skipping")
            continue

        z0 = Z_d[y_d == 0].mean(axis=0)
        z1 = Z_d[y_d == 1].mean(axis=0)
        delta_z = z1 - z0
        cls = W @ delta_z
        cls_list.append(cls)
        delta_z_list.append(delta_z)
        weights.append(len(Z_d))

        print(f"  Dataset {d_id}: ||delta_z|| = {np.linalg.norm(delta_z):.4f}, ||CLS|| = {np.linalg.norm(cls):.4f}")

    if len(cls_list) == 0:
        raise ValueError("No valid CLS computed. Check class distribution.")

    # Weighted average
    weights = np.array(weights)
    weights = weights / weights.sum()
    cls_weighted = np.zeros_like(cls_list[0])
    for cls, w in zip(cls_list, weights):
        cls_weighted += w * cls

    return cls_weighted, cls_list, delta_z_list, weights


def compute_log2fc(X, y, pseudocount=1e-8):
    """Compute log2 fold change."""
    mean0 = X[y == 0].mean(axis=0)
    mean1 = X[y == 1].mean(axis=0)

    log2fc = np.zeros_like(mean0)
    for i in range(len(mean0)):
        if mean0[i] == 0 and mean1[i] == 0:
            log2fc[i] = 0.0
        else:
            log2fc[i] = np.log2((mean1[i] + pseudocount) / (mean0[i] + pseudocount))

    log2fc = np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0)
    return log2fc


def safe_correlation(x, y):
    """Safely compute correlation coefficients."""
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan, np.nan, np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    try:
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        if np.isnan(rho) or np.isnan(p_rho):
            rho, p_rho = np.nan, np.nan
        if np.isnan(r) or np.isnan(p_r):
            r, p_r = np.nan, np.nan
        return rho, p_rho, r, p_r
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def permutation_test_cls(X, y, W, Z, dataset_ids, genes, output_dir, n_perm=1000):
    """
    Permutation test: check whether CLS is significantly larger than random expectation.
    """
    observed_cls, _, _, _ = compute_cls_unified(W, Z, dataset_ids, y)
    observed_magnitude = np.mean(np.abs(observed_cls))
    observed_log2fc = compute_log2fc(X, y)
    observed_rho, _, _, _ = safe_correlation(observed_cls, observed_log2fc)

    if np.isnan(observed_rho):
        observed_rho = 0.0

    print(f"\n  Observed: mean|CLS| = {observed_magnitude:.4f}, Spearman rho = {observed_rho:.4f}")
    print(f"  Running {n_perm} permutations...")

    perm_magnitudes = []
    perm_rhos = []

    for _ in tqdm(range(n_perm), desc="  Permutation"):
        # Shuffle labels within each dataset
        y_shuffled = y.copy()
        for d_id in np.unique(dataset_ids):
            mask = dataset_ids == d_id
            y_shuffled[mask] = shuffle(y[mask])

        cls_shuff, _, _, _ = compute_cls_unified(W, Z, dataset_ids, y_shuffled)
        perm_magnitudes.append(np.mean(np.abs(cls_shuff)))
        rho, _, _, _ = safe_correlation(cls_shuff, observed_log2fc)
        if np.isnan(rho):
            rho = 0.0
        perm_rhos.append(rho)

    p_mag = np.mean(np.array(perm_magnitudes) >= observed_magnitude)
    p_rho = np.mean(np.array(perm_rhos) >= observed_rho)

    # Plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(perm_magnitudes, bins=50, color='lightgray', edgecolor='black')
        axes[0].axvline(observed_magnitude, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Mean |CLS| (permuted)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'p = {p_mag:.4f}')

        axes[1].hist(perm_rhos, bins=50, color='lightgray', edgecolor='black')
        axes[1].axvline(observed_rho, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Spearman rho (permuted)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'p = {p_rho:.4f}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'permutation_test.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"  Warning: Could not plot permutation results: {e}")

    return {'p_magnitude': p_mag, 'p_correlation': p_rho}


def plot_cls_vs_log2fc(cls, log2fc, genes, output_dir, top_n=30):
    """Scatter plot of CLS vs log2FC."""
    rho, p_rho, r, p_r = safe_correlation(cls, log2fc)

    if np.isnan(rho):
        print("  Warning: Correlation is NaN. Skipping plot.")
        return {'rho': np.nan, 'p_rho': np.nan}

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(cls, log2fc, alpha=0.5, s=10, color='steelblue')

    top_idx = np.argsort(np.abs(cls))[-top_n:]
    for idx in top_idx:
        ax.annotate(genes[idx], (cls[idx], log2fc[idx]), fontsize=8, alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Conditional Latent Shift (CLS)', fontsize=12)
    ax.set_ylabel('Observed log2 Fold Change', fontsize=12)
    ax.set_title(f'Unified Model (All Datasets)\nSpearman rho = {rho:.3f} (p={p_rho:.3e})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'CLS_vs_log2FC.png'), dpi=300)
    plt.close()

    return {'rho': rho, 'p_rho': p_rho}


def run_enrichment(cls_series, genes, output_dir, top_k=500):
    """Run GO/KEGG enrichment analysis."""
    if gp is None:
        return None

    sorted_df = pd.DataFrame({'gene': genes, 'cls': cls_series}).sort_values('cls', ascending=False)

    # Select top and bottom genes
    top_genes = sorted_df.head(top_k // 2)['gene'].tolist()
    bottom_genes = sorted_df.tail(top_k // 2)['gene'].tolist()
    combined = top_genes + bottom_genes

    if len(combined) < 5:
        print(f"  Too few genes for enrichment ({len(combined)})")
        return None

    try:
        enrich_res = gp.enrichr(
            gene_list=combined,
            gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],
            organism='human',
            outdir=None,
            cutoff=0.05
        )
        for gs in ['KEGG_2021_Human', 'GO_Biological_Process_2021']:
            if gs in enrich_res.results:
                enrich_res.results[gs].to_csv(
                    os.path.join(output_dir, f'ORA_{gs}.csv'), index=False
                )
        print("  Enrichment results saved.")
        return enrich_res.results
    except Exception as e:
        print(f"  Enrichment error: {e}")
        return None


def export_string_list(cls, genes, output_dir, top_k=300):
    """Export gene lists for STRING."""
    sorted_df = pd.DataFrame({'gene': genes, 'cls': cls}).sort_values('cls', ascending=False)

    top_genes = sorted_df.head(top_k)['gene'].tolist()
    bottom_genes = sorted_df.tail(top_k)['gene'].tolist()
    abs_sorted = sorted_df.assign(abs_cls=np.abs(sorted_df['cls'])).sort_values('abs_cls', ascending=False)
    top_abs = abs_sorted.head(top_k)['gene'].tolist()

    pd.DataFrame({'gene': top_genes}).to_csv(
        os.path.join(output_dir, f'top_{top_k}_CLS_genes.csv'), index=False
    )
    pd.DataFrame({'gene': bottom_genes}).to_csv(
        os.path.join(output_dir, f'bottom_{top_k}_CLS_genes.csv'), index=False
    )
    pd.DataFrame({'gene': top_abs}).to_csv(
        os.path.join(output_dir, f'top_{top_k}_abs_CLS_genes.csv'), index=False
    )
    print(f"  Exported gene lists for STRING.")
    return top_abs


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified CLSA Pipeline with Batch Correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_CLSA_pipeline.py --input_dir ./data/raw --output_dir ./unified_results --device cuda
  python unified_CLSA_pipeline.py --input_dir ./data/raw --min_samples 30 --latent_dim 8
        """
    )
    parser.add_argument("--input_dir", required=True, help="Directory with raw data files (CSV/XLSX)")
    parser.add_argument("--output_dir", default="./unified_CLSA_results", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--min_samples", type=int, default=6, help="Minimum samples per dataset to include")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension (reduced for stability)")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[64, 32], help="Encoder hidden dimensions")
    parser.add_argument("--alpha", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--beta", type=float, default=1e-3, help="L1 regularization weight")
    parser.add_argument("--n_perm", type=int, default=1000, help="Number of permutations")
    parser.add_argument("--top_k", type=int, default=500, help="Top K genes for enrichment")
    parser.add_argument("--no_batch_correct", action="store_true", help="Skip batch correction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading datasets and batch correction")
    print("=" * 70)

    X, y, dataset_ids, dataset_names, genes = load_and_align_datasets(
        args.input_dir,
        min_samples=args.min_samples,
        batch_correct=not args.no_batch_correct
    )

    n_datasets = len(dataset_names)

    # Save intermediate results
    np.save(os.path.join(args.output_dir, 'X.npy'), X)
    np.save(os.path.join(args.output_dir, 'y.npy'), y)
    np.save(os.path.join(args.output_dir, 'dataset_ids.npy'), dataset_ids)
    with open(os.path.join(args.output_dir, 'genes.txt'), 'w') as f:
        for g in genes:
            f.write(g + '\n')

    # Step 2: Train unified model
    print("\n" + "=" * 70)
    print("STEP 2: Training unified FiLM-DAE-Lin model")
    print("=" * 70)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    d_t = torch.tensor(dataset_ids, dtype=torch.long)

    n_total = len(X_t)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    dataset = TensorDataset(X_t, y_t, d_t)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=min(64, n_train), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(64, n_val), shuffle=False)

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    latent_dim = min(args.latent_dim, X.shape[1] - 1, n_total - 1, 32)
    if latent_dim < 1:
        raise ValueError(f"Latent dimension too small: {latent_dim}")
    print(f"Effective latent dimension: {latent_dim}")

    model = UnifiedFiLM_DAE_Lin(
        input_dim=X.shape[1],
        n_datasets=n_datasets,
        latent_dim=latent_dim,
        hidden_dims=args.hidden_dims,
        cond_dim=8,
        film_hidden=16,
        dropout=0.2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_unified_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        patience=args.patience
    )

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

    # Step 3: Extract W and Z
    print("\n" + "=" * 70)
    print("STEP 3: Extracting W and Z")
    print("=" * 70)

    W = model.decoder_weight.detach().cpu().numpy()
    print(f"W shape: {W.shape}")

    model.eval()
    Z_list = []
    with torch.no_grad():
        for bx, by, bd in DataLoader(dataset, batch_size=128, shuffle=False):
            bx, by, bd = bx.to(device), by.to(device), bd.to(device)
            z = model.encode(bx, bd, by)
            Z_list.append(z.cpu().numpy())
    Z = np.vstack(Z_list)
    print(f"Z shape: {Z.shape}")

    np.save(os.path.join(args.output_dir, 'W.npy'), W)
    np.save(os.path.join(args.output_dir, 'Z.npy'), Z)

    # Step 4: CLSA
    print("\n" + "=" * 70)
    print("STEP 4: Conditional Latent Shift Analysis (CLSA)")
    print("=" * 70)

    cls_weighted, cls_list, delta_z_list, weights = compute_cls_unified(W, Z, dataset_ids, y)
    log2fc = compute_log2fc(X, y)

    # Save CLS results
    result_df = pd.DataFrame({
        'gene': genes,
        'CLS': cls_weighted,
        'log2FC': log2fc
    }).sort_values('CLS', ascending=False)
    result_df.to_csv(os.path.join(args.output_dir, 'CLS_results.csv'), index=False)

    # Correlation analysis
    corr = plot_cls_vs_log2fc(cls_weighted, log2fc, genes, args.output_dir)
    print(f"\nCLS vs log2FC: Spearman rho = {corr['rho']:.4f} (p={corr['p_rho']:.3e})")

    # Permutation test
    perm = permutation_test_cls(X, y, W, Z, dataset_ids, genes, args.output_dir, args.n_perm)
    print(f"\nPermutation test:")
    print(f"  p (magnitude) = {perm['p_magnitude']:.4f}")
    print(f"  p (correlation) = {perm['p_correlation']:.4f}")

    # Step 5: Enrichment analysis
    print("\n" + "=" * 70)
    print("STEP 5: GO/KEGG Enrichment Analysis")
    print("=" * 70)

    run_enrichment(cls_weighted, genes, args.output_dir, args.top_k)

    # Step 6: Export STRING lists
    print("\n" + "=" * 70)
    print("STEP 6: Exporting gene lists for STRING")
    print("=" * 70)

    export_string_list(cls_weighted, genes, args.output_dir, top_k=300)

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        'n_datasets': n_datasets,
        'dataset_names': ', '.join(dataset_names),
        'n_samples': len(X),
        'n_genes': len(genes),
        'latent_dim': latent_dim,
        'spearman_rho': corr['rho'],
        'spearman_p': corr['p_rho'],
        'perm_p_magnitude': perm['p_magnitude'],
        'perm_p_correlation': perm['p_correlation']
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)

    print(f"\nSpearman rho: {corr['rho']:.4f} (p={corr['p_rho']:.3e})")
    print(f"Permutation p (correlation): {perm['p_correlation']:.4f}")

    print("\n" + "=" * 70)
    print("UNIFIED CLSA PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nOutput files:")
    print(f"  - CLS_results.csv        : Gene-level CLS and log2FC")
    print(f"  - CLS_vs_log2FC.png      : Correlation scatter plot")
    print(f"  - permutation_test.png   : Permutation test results")
    print(f"  - ORA_*.csv              : GO/KEGG enrichment")
    print(f"  - top_*_CLS_genes.csv    : Gene lists for STRING")
    print(f"  - summary.csv            : Summary statistics")
    print(f"  - model.pth              : Trained model weights")
    print(f"  - X.npy, y.npy, W.npy, Z.npy : Saved data")


if __name__ == "__main__":
    main()
