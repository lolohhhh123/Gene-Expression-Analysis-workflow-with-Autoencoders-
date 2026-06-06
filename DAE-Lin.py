#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FiLM-DAE-Lin: Condition-Driven Discriminative Autoencoder with Linear Decoder
and Feature-wise Linear Modulation (FiLM) on the encoder.

Input: CSV/XLSX file with genes as rows, samples as columns.
       Column names must contain 'Control' or 'AD' (case-insensitive).
Output: CSV file with gene importance scores.
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
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import argparse

# ------------------------------
# 1. Data loading and preprocessing
# ------------------------------
def load_and_preprocess(file_path, normalize=True):
    """
    Read data, automatically detect orientation, handle duplicate gene names,
    return (data, labels, gene_names).
    Supports two formats:
    1. First row contains sample labels (0/1), first column contains gene names.
    2. Conventional format: rows are samples, columns are genes.
    """
    if file_path.endswith('.csv'):
        # Key change: do not use the first row as header, treat it as data
        df = pd.read_csv(file_path, header=None, index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, header=None, index_col=0)
    else:
        raise ValueError("Unsupported file format")

    # Now df's columns are 0,1,2,...; the first row (original sample labels) is at index 0
    # Extract sample labels (first row of data)
    if df.shape[0] > 0:
        sample_labels_row = df.iloc[0, :].values  # first row, each element is 0 or 1
        # Remove the first row, the rest is expression data
        df = df.iloc[1:, :]
    else:
        raise ValueError("Empty file")

    # Transpose: rows as samples, columns as genes (consistent with DAE code)
    df = df.T
    print(f"Samples as rows: {df.shape[0]} samples, {df.shape[1]} genes")

    # Clean: drop rows/columns that are all NaN, fill remaining NaN with 0
    df = df.dropna(axis=0, how='any')
    df = df.dropna(axis=1, how='any')
    df = df.fillna(0)

    # Handle duplicate gene names (column names)
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
        print(f"Duplicated gene names merged. New shape: {df.shape}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Empty after preprocessing")

    gene_names = df.columns.tolist()
    
    # Sample labels: sample_labels_row length should equal number of samples
    # Note: sample_labels_row corresponds to the original columns before transpose.
    # After transpose, sample order remains unchanged, so use directly.
    labels = sample_labels_row.astype(np.float32)
    
    # Check if label count matches
    if len(labels) != df.shape[0]:
        print(f"Warning: label count {len(labels)} != sample count {df.shape[0]}, adjusting...")
        # If mismatch, attempt to extract from df index (legacy logic)
        labels = []
        for name in df.index:
            name_str = str(name).split('.')[0]
            if name_str in ('0', '0.0'):
                labels.append(0)
            elif name_str in ('1', '1.0'):
                labels.append(1)
            else:
                try:
                    val = int(float(name_str))
                    labels.append(1 if val == 1 else 0)
                except:
                    labels.append(0)
        labels = np.array(labels, dtype=np.float32)

    data = df.values.astype(np.float32)

    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    print(f"Class distribution: Control={np.sum(labels==0)}, AD={np.sum(labels==1)}")
    return data, labels, gene_names

# ------------------------------
# 2. FiLM Layer and FiLM Generator
# ------------------------------
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.
    Applies gamma * h + beta, where gamma and beta are learned per feature channel.
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        # gamma and beta will be provided by the FiLM generator; no parameters here

    def forward(self, h, gamma, beta):
        # h: (batch, num_features)
        # gamma, beta: (batch, num_features) or (num_features,) ; broadcastable
        return gamma * h + beta


class FiLMGenerator(nn.Module):
    def __init__(self, cond_dim, hidden_dim, hidden_dims_list):
        super().__init__()
        self.hidden_dims_list = hidden_dims_list
        num_layers = len(hidden_dims_list)
        total_output_dim = sum(hidden_dims_list) * 2  # gamma and beta for each layer
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_output_dim)
        )

    def forward(self, cond):
        params = self.net(cond)  # (batch, total_output_dim)
        batch = params.size(0)
        outputs = []
        start = 0
        for hdim in self.hidden_dims_list:
            gamma = params[:, start:start+hdim]          # (batch, hdim)
            beta  = params[:, start+hdim:start+2*hdim]   # (batch, hdim)
            outputs.append((gamma, beta))
            start += 2 * hdim
        return outputs  # list of tuples


# ------------------------------
# 3. FiLM-DAE-Lin Model
# ------------------------------
class FiLM_DAE_Lin(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64],
                 cond_dim=2, film_hidden=32, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        # Condition embedding
        self.cond_embedding = nn.Embedding(2, cond_dim)
        # FiLM generator
        self.film_gen = FiLMGenerator(cond_dim, film_hidden, hidden_dims)
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            block = nn.ModuleList([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                FiLMLayer(h_dim),   # will receive gamma/beta from film_gen
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            self.encoder_blocks.append(block)
            prev_dim = h_dim
        self.to_latent = nn.Linear(hidden_dims[-1], latent_dim)
        # Linear decoder
        self.decoder_weight = nn.Parameter(torch.Tensor(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.decoder_weight)
        # Classifier
        self.classifier = nn.Linear(latent_dim, 1)

    def encode(self, x, cond):
        # cond: (batch,) integer tensor (0/1)
        cond_emb = self.cond_embedding(cond.long())          # (batch, cond_dim)
        film_params = self.film_gen(cond_emb)                # list of (gamma, beta)
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = block[0](h)          # Linear
            h = block[1](h)          # BatchNorm
            gamma, beta = film_params[i]
            h = block[2](h, gamma, beta)   # FiLM
            h = block[3](h)          # ReLU
            h = block[4](h)          # Dropout
        z = self.to_latent(h)
        return z

    def decode(self, z):
        return torch.mm(z, self.decoder_weight.T)

    def forward(self, x, cond):
        z = self.encode(x, cond)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, logits, z

    def decode(self, z):
        return torch.mm(z, self.decoder_weight.T)

    def forward(self, x, cond):
        z = self.encode(x, cond)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, logits, z


# ------------------------------
# 4. Loss function (same as DAE-Lin)
# ------------------------------
def loss_function(recon_x, x, logits, y, decoder_weight, alpha=1.0, beta=1e-4):
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')(logits.view(-1), y)
    l1_loss = torch.sum(torch.abs(decoder_weight))
    total_loss = recon_loss + alpha * bce_loss + beta * l1_loss
    return total_loss, recon_loss, bce_loss, l1_loss


# ------------------------------
# 5. Training function (with condition labels)
# ------------------------------
def train_model(model, train_loader, val_loader, device, epochs=200, lr=1e-3,
                alpha=1.0, beta=1e-4, patience=15, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            recon, logits, _ = model(batch_x, batch_y)  # pass condition labels
            loss, rl, cl, l1 = loss_function(recon, batch_x, logits, batch_y,
                                             model.decoder_weight, alpha, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                recon, logits, _ = model(batch_x, batch_y)
                loss, _, _, _ = loss_function(recon, batch_x, logits, batch_y,
                                              model.decoder_weight, alpha, beta)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ------------------------------
# 6. Gene importance computation (same as DAE-Lin)
# ------------------------------
def compute_gene_importance(model, data_tensor, labels, device, n_estimators=100):
    model.eval()
    with torch.no_grad():
        z_list = []
        loader = DataLoader(TensorDataset(data_tensor, labels), batch_size=64, shuffle=False)
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            z = model.encode(batch_x, batch_y)  # need condition labels
            z_list.append(z.cpu().numpy())
        Z = np.vstack(z_list)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(Z, labels.cpu().numpy().ravel())
    latent_importance = rf.feature_importances_

    W = model.decoder_weight.detach().cpu().numpy()
    gene_importance = np.sum(np.abs(W) * latent_importance, axis=1)
    return gene_importance, latent_importance, Z


# ------------------------------
# 7. Single dataset processing
# ------------------------------
def process_dataset(file_path, output_dir, latent_dim=32, hidden_dims=[128,64],
                    epochs=200, lr=1e-3, alpha=1.0, beta=1e-4, patience=15,
                    device='cpu', normalize=True):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    data, labels, gene_names = load_and_preprocess(file_path, normalize=normalize)
    print(f"Data shape: {data.shape}, samples: {data.shape[0]}, genes: {data.shape[1]}")
    print(f"Class distribution: Control={np.sum(labels==0)}, AD={np.sum(labels==1)}")

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    n_total = len(X)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    input_dim = data.shape[1]
    model = FiLM_DAE_Lin(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)

    model, history = train_model(model, train_loader, val_loader, device,
                                 epochs=epochs, lr=lr, alpha=alpha, beta=beta,
                                 patience=patience, verbose=True)

    gene_imp, latent_imp, Z = compute_gene_importance(model, X, y, device)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    imp_df = pd.DataFrame({'gene': gene_names, 'importance': gene_imp})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    out_path = os.path.join(output_dir, f"{base_name}_FiLM_DAE_Lin_importance.csv")
    imp_df.to_csv(out_path, index=False)
    print(f"Saved importance scores to {out_path}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(output_dir, f"{base_name}_training_history.csv")
    hist_df.to_csv(hist_path, index=False)

    return imp_df, hist_df


# ------------------------------
# 8. Batch processing and cross-dataset ranking
# ------------------------------
def batch_process_and_rank(input_dir, output_dir, **kwargs):
    files = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.xlsx"))
    if not files:
        print(f"No data files found in {input_dir}")
        return

    all_imp = {}
    for f in tqdm(files, desc="Processing datasets"):
        try:
            imp_df, _ = process_dataset(f, output_dir, **kwargs)
            base = os.path.splitext(os.path.basename(f))[0]
            all_imp[base] = imp_df.set_index('gene')['importance']
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    imp_all = pd.DataFrame(all_imp).fillna(0)
    imp_log = np.log1p(imp_all)
    z_scores = (imp_log - imp_log.mean()) / imp_log.std(ddof=0)
    z_scores = z_scores.fillna(0)
    final_score = z_scores.mean(axis=1)
    final_rank = final_score.sort_values(ascending=False)

    final_df = pd.DataFrame({
        'gene': final_rank.index,
        'mean_Z_score': final_rank.values,
        'rank': range(1, len(final_rank)+1)
    })
    final_df.to_csv(os.path.join(output_dir, "final_cross_dataset_ranking.csv"), index=False)
    print(f"Cross-dataset ranking saved to {os.path.join(output_dir, 'final_cross_dataset_ranking.csv')}")
    return final_df


# ------------------------------
# 9. Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="FiLM-DAE-Lin model for gene importance learning")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CSV/XLSX files")
    parser.add_argument("--output_dir", type=str, default="./FiLM_DAE_Lin_results", help="Output directory")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension (k)")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128,64], help="Encoder hidden layers")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for classification loss")
    parser.add_argument("--beta", type=float, default=1e-4, help="Weight for L1 sparsity")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--no_normalize", action="store_true", help="Disable gene-wise normalization")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    final_ranking = batch_process_and_rank(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        patience=args.patience,
        device=device,
        normalize=not args.no_normalize
    )

if __name__ == "__main__":
    main()
