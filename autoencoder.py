import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #For debug
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #set the number or CUDA Devices

import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.losses import mse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import gc
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# -------------------- GPU Configuration --------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[:3], 'GPU')
        print(f"Using {len(gpus[:3])} GPUs")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.get_strategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# -------------------- Model Definitions --------------------
def single_gpu_autoencoder_model(embedding_dim, num_genes):
    """Basic autoencoder."""
    input_layer = Input(shape=(num_genes,))
    encoded = Dense(embedding_dim, activation='relu')(input_layer)
    decoded = Dense(num_genes, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def simplified_vae_model(embedding_dim, num_genes):
    """Simplified variational autoencoder with custom loss."""
    inputs = Input(shape=(num_genes,), name='encoder_input')
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(embedding_dim, name='z_mean')(x)
    z_log_var = Dense(embedding_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(embedding_dim,), name='z')([z_mean, z_log_var])
    x = Dense(128, activation='relu')(z)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_genes, activation='sigmoid')(x)
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = mse(inputs, outputs) * num_genes
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(total_loss)
    vae.compile(optimizer='adam')
    encoder = Model(inputs, z_mean, name='encoder')
    return vae, encoder

def simplified_dbn_model(embedding_dim, num_genes):
    """Deep autoencoder (denoising-style without noise)."""
    inputs = Input(shape=(num_genes,))
    x = Dense(512, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(embedding_dim, activation='relu', name='bottleneck')(x)
    x = Dense(256, activation='relu')(encoded)
    x = Dense(512, activation='relu')(x)
    decoded = Dense(num_genes, activation='sigmoid')(x)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

def denoising_autoencoder_model(embedding_dim, num_genes):
    """Denoising autoencoder with dropout noise."""
    inputs = Input(shape=(num_genes,))
    noisy_inputs = Dropout(0.2)(inputs, training=True)
    x = Dense(512, activation='relu')(noisy_inputs)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(embedding_dim, activation='relu')(x)
    x = Dense(256, activation='relu')(encoded)
    x = Dense(512, activation='relu')(x)
    decoded = Dense(num_genes, activation='sigmoid')(x)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

def sparse_autoencoder_model(embedding_dim, num_genes):
    """Sparse autoencoder with L1 activity regularization."""
    inputs = Input(shape=(num_genes,))
    encoded = Dense(embedding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(inputs)
    decoded = Dense(num_genes, activation='sigmoid')(encoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

def build_model(model_type, embedding_dim, num_genes):
    """Unified model builder."""
    if model_type == 'autoencoder':
        model = single_gpu_autoencoder_model(embedding_dim, num_genes)
        encoder = Model(model.input, model.layers[1].output)
        return model, encoder
    elif model_type == 'vae':
        return simplified_vae_model(embedding_dim, num_genes)
    elif model_type == 'dbn':
        return simplified_dbn_model(embedding_dim, num_genes)
    elif model_type == 'denoising':
        return denoising_autoencoder_model(embedding_dim, num_genes)
    elif model_type == 'sparse':
        return sparse_autoencoder_model(embedding_dim, num_genes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# -------------------- Training and Evaluation --------------------
def train_and_evaluate_model(model_type, X_train, X_val, embedding_dim, epochs, num_genes):
    """Train a model and return encoded features, loss, encoder and full model."""
    try:
        print(f"Building {model_type} model with embedding_dim={embedding_dim}")
        model, encoder = build_model(model_type, embedding_dim, num_genes)
        batch_size = min(32, X_train.shape[0] // 4)
        batch_size = max(8, batch_size)
        print(f"Training with batch_size={batch_size}")
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True
        )
        encoded_data = encoder.predict(X_val, batch_size=batch_size)
        loss = model.evaluate(X_val, X_val, batch_size=batch_size, verbose=0)
        if isinstance(loss, list):
            loss = loss[0]
        print(f"{model_type.upper()} training completed with loss: {loss:.4f}")
        return encoded_data, loss, encoder, model
    except Exception as e:
        print(f"Error training {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, float('inf'), None, None

# -------------------- Memory Estimation --------------------
def estimate_memory_usage(num_genes, num_samples, embedding_dim=100, batch_size=32):
    """Estimate GPU memory usage for a model."""
    model_params = (num_genes * embedding_dim + embedding_dim * num_genes) * 4
    activation_memory = (batch_size * (num_genes + embedding_dim + num_genes)) * 4
    gradient_memory = model_params
    optimizer_memory = 2 * model_params
    total_memory_bytes = model_params + activation_memory + gradient_memory + optimizer_memory
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    safety_factor = 3.0
    return total_memory_mb * safety_factor

def get_available_gpu_memory():
    """Get minimum available GPU memory in MB."""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        available_memory = []
        for gpu in gpus:
            memory_info = tf.config.experimental.get_memory_info(gpu.name)
            available = (memory_info['limit'] - memory_info['current']) / (1024 * 1024)
            available_memory.append(available)
        return min(available_memory) if available_memory else 2000
    except:
        return 2000

def memory_safe_grid_search(X_train, X_val, num_genes, embedding_dim_list, epochsnumlist):
    """Grid search over embedding dimension and epochs, skipping combinations that exceed memory."""
    best_score = float('inf')
    best_params = {'embedding_dim': 100, 'epochs': 50}
    tested_combinations = 0
    max_combinations = 8
    for embedding_dim in embedding_dim_list[:4]:
        for epochs in epochsnumlist[:8]:
            if tested_combinations >= max_combinations:
                break
            tested_combinations += 1
            print(f"Testing combination {tested_combinations}: embedding_dim={embedding_dim}, epochs={epochs}")
            estimated_memory = estimate_memory_usage(num_genes, X_train.shape[0], embedding_dim)
            available_memory = get_available_gpu_memory()
            if estimated_memory > available_memory * 0.8:
                print(f"  Skipping - estimated memory {estimated_memory:.2f}MB > available {available_memory:.2f}MB")
                continue
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                model = single_gpu_autoencoder_model(embedding_dim, num_genes)
                batch_size = min(16, X_train.shape[0] // 2)
                history = model.fit(
                    X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                val_loss = min(history.history['val_loss'])
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = {'embedding_dim': embedding_dim, 'epochs': epochs}
                print(f"  Val_loss: {val_loss:.4f}, Best: {best_score:.4f}")
                del model
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception as e:
                print(f"  Failed with error: {e}")
                continue
    return best_params

# -------------------- Distribution Check and Cross-Dataset Ranking --------------------
def check_lognormal_distribution(data, sample_name, model_type, output_dir):
    """
    Check if log-transformed data is normally distributed (i.e., raw data is log-normal).
    Returns a boolean, p-value, and test statistic.
    """
    # Use log1p to handle zeros
    data_transformed = np.log1p(data)
    log_type = "log1p"

    if len(data_transformed) <= 5000:
        stat, p_value = stats.shapiro(data_transformed)
    else:
        z_data = (data_transformed - np.mean(data_transformed)) / np.std(data_transformed)
        stat, p_value = stats.kstest(z_data, 'norm')

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(data_transformed, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot ({log_type} transformed)\n{sample_name} - {model_type}')
    qq_path = os.path.join(output_dir, f'{sample_name}_{model_type}_lognorm_check.png')
    plt.savefig(qq_path, dpi=100)
    plt.close()

    is_normal = p_value > 0.05
    return is_normal, p_value, stat

def compute_cross_dataset_ranking(importance_raw, sample_name, model_type, output_dir):
    """
    Apply log1p transformation, then Z-score standardization to make feature importances
    comparable across datasets.
    Returns a DataFrame with genes, raw importance, log importance, Z-score, and rank.
    """
    log_importance = np.log1p(importance_raw)
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(log_importance.values.reshape(-1, 1)).flatten()

    ranking_df = pd.DataFrame({
        'gene': importance_raw.index,
        'importance_raw': importance_raw.values,
        'log_importance': log_importance.values,
        'z_score': z_scores
    })
    ranking_df = ranking_df.sort_values('z_score', ascending=False)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)

    rank_file = os.path.join(output_dir, f'{sample_name}_{model_type}_zscore_ranking.csv')
    ranking_df.to_csv(rank_file, index=False)
    return ranking_df

# -------------------- Dataset Processing --------------------
def process_dataset(file_path, train_dir, output_dir, models_list, embedding_dim_grid, epochs_grid):
    """Process a single test Excel file: train models, extract features, compute ranking."""
    filename = os.path.basename(file_path)
    sample_name = filename.replace("_test.xlsx", "")
    train_file = os.path.join(train_dir, f"{sample_name}_train.xlsx")

    if not os.path.exists(train_file):
        print(f"Warning: Training file {train_file} not found. Skipping {sample_name}.")
        return

    tf.keras.backend.clear_session()
    gc.collect()

    print(f"\n{'='*50}")
    print(f"Processing {sample_name}")
    print(f"{'='*50}")

    try:
        # Load data
        gene_train_data = pd.read_excel(train_file, index_col=0).dropna(how='any').replace('nan', 0)
        gene_val_data = pd.read_excel(file_path, index_col=0).dropna(how='any').replace('nan', 0)

        gene_ids = gene_train_data.index.values
        train_data = gene_train_data.values
        val_data = gene_val_data.values
        gene_index_dict = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

        def extract_labels_from_df(df):
            labels = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ['control', 'ctrl', 'normal']):
                    labels.append(0)
                elif any(kw in col_lower for kw in ['ad', 'alzheimer', 'disease']):
                    labels.append(1)
                else:
                    labels.append(-1)
            df_clean = df.loc[:, [l != -1 for l in labels]]
            labels_clean = [l for l in labels if l != -1]
            return df_clean, np.array(labels_clean)

        gene_train_data_T = gene_train_data.T   # (samples, genes)
        train_df_clean, y_train = extract_labels_from_df(gene_train_data_T)

        gene_val_data_T = gene_val_data.T
        val_df_clean, y_val = extract_labels_from_df(gene_val_data_T)

        expression_train = train_df_clean.values / (train_df_clean.values.max(axis=1, keepdims=True) + 1e-7)
        expression_val = val_df_clean.values / (val_df_clean.values.max(axis=1, keepdims=True) + 1e-7)

        X_train = expression_train
        X_val = expression_val
        num_samples, num_genes = X_train.shape

        print(f"Data shape: {X_train.shape}, y_train shape: {y_train.shape}")
        estimated_memory = estimate_memory_usage(num_genes, num_samples)
        available_memory = get_available_gpu_memory()
        print(f"Estimated memory: {estimated_memory:.2f}MB, Available: {available_memory:.2f}MB")
        if estimated_memory > available_memory * 0.99:
            print(f"SKIPPING {sample_name} - Memory requirement too high")
            return

        # Hyperparameter search
        print("Starting memory-safe GridSearchCV...")
        best_params = memory_safe_grid_search(X_train, X_val, num_genes, embedding_dim_grid, epochs_grid)
        embedding_dim = best_params['embedding_dim']
        epochs = best_params['epochs']
        print(f"Best parameters: embedding_dim={embedding_dim}, epochs={epochs}")

        # Run each model type
        for model_type in models_list:
            print(f"\n{'='*30}")
            print(f"Training {model_type.upper()} model for {sample_name}")
            print(f"{'='*30}")
            try:
                encoded_data, loss, encoder_model, full_model = train_and_evaluate_model(
                    model_type, X_train, X_val, embedding_dim, epochs, num_genes)
                if encoded_data is None:
                    continue
                    
                if model_type == 'vae':
                    decoder_layers = [layer for layer in full_model.layers if isinstance(layer, tf.keras.layers.Dense)]
                    first_decoder_layer = None
                    for layer in decoder_layers:
                        if layer.input_shape[-1] == embedding_dim + 1:
                            first_decoder_layer = layer
                            break
                    if first_decoder_layer is None:
                        raise ValueError("Could not find decoder first linear layer")
                    W_first = first_decoder_layer.get_weights()[0]   # (hidden, latent_dim+1)
                    W_z = W_first[:, :embedding_dim]                 # (hidden, latent_dim)

                    encoder = Model(full_model.input, full_model.get_layer('z_mean').output)
                    Z_train = encoder.predict(X_train, batch_size=32)

                    from sklearn.ensemble import RandomForestClassifier
                    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    rf_clf.fit(Z_train, y_train)
                    latent_imp = rf_clf.feature_importances_          # (latent_dim,)

                    decoder_linear_layers = []
                    for layer in full_model.layers:
                        if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'z_mean' and layer.name != 'z_log_var':
                            decoder_linear_layers.append(layer)
                            
                    subsequent_layers = decoder_linear_layers[decoder_linear_layers.index(first_decoder_layer)+1:]
                    for layer in subsequent_layers:
                        W_layer = layer.get_weights()[0]  # (next_dim, current_dim)
                        W = W_layer @ W                    # (next_dim, latent_dim)
                    # 最终 W 形状 (num_genes, latent_dim)
                    gene_scores = np.abs(W @ latent_imp)    # (num_genes,)
                    
                else:
                    weights = encoder_model.get_weights()[0]   # (num_genes, embedding_dim)
                    weights_with_names = {gene: weights[idx, :] for gene, idx in gene_index_dict.items()}
                    weight_df = pd.DataFrame.from_dict(weights_with_names, orient='index')
                    df_t = weight_df.T   # (embedding_dim, num_genes)
                    le = LabelEncoder()
                    y = le.fit_transform(df_t.index.tolist())
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    rf.fit(df_t.values, y)
                    importance = rf.feature_importances_
                    gene_scores = importance

                importance_series = pd.Series(gene_scores, index=gene_ids)  # 注意索引应与基因名顺序一致
                imp_raw_path = os.path.join(output_dir, f'{sample_name}_{model_type}_importance_raw.csv')
                importance_series.to_csv(imp_raw_path, header=['importance'])

                importance_arr = gene_scores
                is_norm, p_value, stat = check_lognormal_distribution(
                    importance_arr, sample_name, model_type, output_dir)
                print(f"Log-normal test p-value: {p_value:.4f} (is_normal={is_norm})")

                ranking_df = compute_cross_dataset_ranking(
                    importance_series, sample_name, model_type, output_dir)
                print(f"Top 5 genes by Z-score: {ranking_df.head(5)['gene'].tolist()}")

                loss_df = pd.DataFrame({'Loss': [loss], 'Model_Type': [model_type]})
                loss_df.to_csv(os.path.join(output_dir, f'{sample_name}_{model_type}_loss.csv'), index=False)

                print(f"SUCCESS: Completed {model_type} for {sample_name}, Loss: {loss:.4f}")

            except Exception as e:
                print(f"ERROR with {model_type} for {sample_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if 'encoder_model' in locals(): del encoder_model
                if 'full_model' in locals(): del full_model
                tf.keras.backend.clear_session()
                gc.collect()

        print(f"Completed all models for {sample_name}")
    except Exception as e:
        print(f"ERROR processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()
# -------------------- Feature Extraction --------------------
def decoder_weight_attribution(vae_model, X_train, y_train, gene_names, device='cpu'):
    """
    Use decoder first-layer weights + Random Forest classifier on latent features.
    """
    decoder_layers = [layer for layer in vae_model.layers if 'dense' in layer.name.lower()]
    for layer in decoder_layers:
        if layer.input_shape[-1] == embedding_dim + 1:  # z + condition dim
            first_decoder_layer = layer
            break
    else:
        raise ValueError("Could not locate decoder first linear layer")
        
    weights = first_decoder_layer.get_weights()[0]          # (hidden, latent_dim+1)
    latent_weights = weights[:, :embedding_dim]              # (hidden, latent_dim)

    encoder = Model(vae_model.input, vae_model.get_layer('z_mean').output)
    Z_train = encoder.predict(X_train, batch_size=32)        # (n_samples, latent_dim)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Z_train, y_train)
    latent_importance = rf.feature_importances_              # (latent_dim,)
    
    gene_scores = np.abs(latent_weights.T @ latent_importance)   # (latent_dim, hidden) @ (latent_dim,) -> (hidden,)
    dense_layers = [layer for layer in vae_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    W = latent_weights
    for layer in dense_layers[1:]:
        W = layer.get_weights()[0] @ W   # (next_dim, current_dim) @ (current_dim, latent_dim)
    # 最终W形状为 (num_genes, latent_dim)
    gene_scores = np.abs(W @ latent_importance)   # (num_genes,)
    
    gene_scores = (gene_scores - gene_scores.min()) / (gene_scores.max() - gene_scores.min() + 1e-8)
    importance_series = pd.Series(gene_scores, index=gene_names)
    return importance_series
# -------------------- Main Entry Point --------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dir = args.test_dir
    train_dir = args.train_dir

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} not found.")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory {train_dir} not found.")

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith("_test.xlsx")]
    print(f"Found {len(test_files)} test files.")

    for file_path in test_files:
        process_dataset(
            file_path=file_path,
            train_dir=train_dir,
            output_dir=args.output_dir,
            models_list=args.models,
            embedding_dim_grid=args.embedding_dims,
            epochs_grid=args.epochs
        )

    print("All processing done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train autoencoders, compute gene importances via RF, "
                    "test log-normal distribution, and produce cross-dataset Z-score rankings."
    )
    parser.add_argument("--train_dir", type=str, default="./train",
                        help="Directory containing *_train.xlsx files (default: ./train)")
    parser.add_argument("--test_dir", type=str, default="./test",
                        help="Directory containing *_test.xlsx files (default: ./test)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output files (default: ./output)")
    parser.add_argument("--models", nargs="+", default=['autoencoder','vae','denoising','sparse'],
                        choices=['autoencoder','vae','dbn','denoising','sparse'],
                        help="Model types to run (default: autoencoder vae denoising sparse)")
    parser.add_argument("--embedding_dims", nargs="+", type=int,
                        default=[50,100,150,200,250,300],
                        help="Embedding dimensions for grid search (default: 50 100 150 200 250 300)")
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=[10,20,30,40,50,60],
                        help="Epoch values for grid search (default: 10 20 30 40 50 60)")
    args = parser.parse_args()
    main(args)
