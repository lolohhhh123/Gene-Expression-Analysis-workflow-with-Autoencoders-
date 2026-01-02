"""
Gene Expression Analysis with Autoencoders
Main training script for dimensionality reduction and feature extraction.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Changes should be made accroding to the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gc
from tensorflow.keras import regularizers


def setup_gpus():
    """Configure GPU settings for TensorFlow."""
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
    return strategy


def single_autoencoder_model(embedding_dim, num_genes):
    """Basic autoencoder model."""
    input_layer = Input(shape=(num_genes,))
    encoded = Dense(embedding_dim, activation='relu')(input_layer)
    decoded = Dense(num_genes, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def simplified_vae_model(embedding_dim, num_genes):
    """Variational autoencoder with simplified architecture."""
    inputs = Input(shape=(num_genes,), name='encoder_input')
    
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    
    z_mean = Dense(embedding_dim, name='z_mean')(x)
    z_log_var = Dense(embedding_dim, name='z_log_var')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling, output_shape=(embedding_dim,), name='z')([z_mean, z_log_var])
    
    x = Dense(128, activation='relu')(z)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_genes, activation='sigmoid')(x)
    
    vae = Model(inputs, outputs, name='vae')
    
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= num_genes
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    
    vae.add_loss(total_loss)
    vae.compile(optimizer='adam')
    
    encoder = Model(inputs, z_mean, name='encoder')
    
    return vae, encoder


def denoising_autoencoder_model(embedding_dim, num_genes):
    """Denoising autoencoder model."""
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
    """Sparse autoencoder with L1 regularization."""
    inputs = Input(shape=(num_genes,))
    
    encoded = Dense(embedding_dim, activation='relu', 
                   activity_regularizer=regularizers.l1(10e-5))(inputs)
    
    decoded = Dense(num_genes, activation='sigmoid')(encoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder


def build_model(model_type, embedding_dim, num_genes):
    """Unified model builder function."""
    if model_type == 'autoencoder':
        model = single_autoencoder_model(embedding_dim, num_genes)
        encoder = Model(model.input, model.layers[1].output)
        return model, encoder
    
    elif model_type == 'vae':
        return simplified_vae_model(embedding_dim, num_genes)
    
    elif model_type == 'denoising':
        return denoising_autoencoder_model(embedding_dim, num_genes)
    
    elif model_type == 'sparse':
        return sparse_autoencoder_model(embedding_dim, num_genes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate_model(model_type, X_train, X_val, embedding_dim, epochs, num_genes):
    """Train and evaluate specified model type."""
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
        
        print(f"{model_type.upper()} training completed with loss: {loss:.4f}")
        
        return encoded_data, loss, encoder, model
        
    except Exception as e:
        print(f"Error training {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, float('inf'), None, None


def estimate_memory_usage(num_genes, num_samples, embedding_dim=100, batch_size=32):
    """Estimate memory usage for model training."""
    model_params = (num_genes * embedding_dim + embedding_dim * num_genes) * 4
    activation_memory = (batch_size * (num_genes + embedding_dim + num_genes)) * 4
    gradient_memory = model_params
    optimizer_memory = 2 * model_params
    
    total_memory_bytes = model_params + activation_memory + gradient_memory + optimizer_memory
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    
    safety_factor = 3.0
    estimated_memory_mb = total_memory_mb * safety_factor
    
    print(f"Estimated memory usage: {estimated_memory_mb:.2f} MB")
    return estimated_memory_mb


def get_available_gpu_memory():
    """Get available GPU memory."""
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


def memory_safe_grid_search(X_train, X_val, num_genes, embedding_dim_list, epochsnumlist, cv_num=2):
    """Memory-safe hyperparameter grid search."""
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
                
                model = single_autoencoder_model(embedding_dim, num_genes)
                
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


def main():
    """Main execution function."""
    localpath = './data/'
    MODEL_TYPES = ['autoencoder', 'vae', 'denoising', 'sparse']
    
    processed_files = []
    skipped_files = []
    
    for filename in os.listdir(localpath + 'test/'):
        if filename.endswith(".xlsx"):
            tf.keras.backend.clear_session()
            gc.collect()
            
            sample_name = filename.split("_test.xlsx")[0]
            print(f"\n{'='*50}")
            print(f"Processing {sample_name}")
            print(f"{'='*50}")
            
            try:
                gene_train_data = pd.read_excel(localpath + 'train/' + sample_name + '_train.xlsx', index_col=0)
                gene_train_data = gene_train_data.dropna(axis=0, how='any').replace('nan', 0)
                
                gene_val_data = pd.read_excel(localpath + 'test/' + sample_name + '_test.xlsx', index_col=0)
                gene_val_data = gene_val_data.dropna(axis=0, how='any').replace('nan', 0)
                
                gene_ids = gene_train_data.index.values
                train_data = gene_train_data.values
                val_data = gene_val_data.values
                
                gene_index_dict = {gene_id: index for index, gene_id in enumerate(gene_ids)}
                
                expression_train = train_data / (train_data.max(axis=1)[:, None] + 1e-7)
                expression_val = val_data / (val_data.max(axis=1)[:, None] + 1e-7)
                
                X_train = expression_train.transpose()
                X_val = expression_val.transpose()
                
                num_samples = X_train.shape[0]
                num_genes = X_train.shape[1]
                
                print(f"Data shape: {X_train.shape}")
                
                estimated_memory = estimate_memory_usage(num_genes, num_samples)
                available_memory = get_available_gpu_memory()
                
                print(f"Estimated memory: {estimated_memory:.2f}MB, Available: {available_memory:.2f}MB")
                
                if estimated_memory > available_memory * 0.99:
                    print(f"SKIPPING {sample_name} - Memory requirement too high")
                    skipped_files.append(sample_name)
                    continue
                
                embedding_dim_list = list(range(50, 301, 50))
                epochsnumlist = list(range(10, 61, 10))
                
                print("Starting memory-safe GridSearchCV...")
                best_params = memory_safe_grid_search(X_train, X_val, num_genes, embedding_dim_list, epochsnumlist)
                
                embedding_dim = best_params['embedding_dim']
                epochs = best_params['epochs']
                
                print(f"Best parameters: embedding_dim={embedding_dim}, epochs={epochs}")
                
                for model_type in MODEL_TYPES:
                    print(f"\n{'='*30}")
                    print(f"Training {model_type.upper()} model for {sample_name}")
                    print(f"{'='*30}")
                    
                    try:
                        encoded_data, loss, encoder_model, trained_model = train_and_evaluate_model(
                            model_type, X_train, X_val, embedding_dim, epochs, num_genes
                        )
                        
                        if encoded_data is not None:
                            if model_type == 'vae':
                                weights = encoder_model.get_weights()[0]
                            else:
                                weights = encoder_model.get_weights()[0]
                            
                            weights_with_names = {}
                            for gene, index in gene_index_dict.items():
                                weights_with_names[gene] = weights[index, :]
                            
                            weight_file = localpath + 'results/' + sample_name + f'_{model_type}_weights.csv'
                            pd.DataFrame.from_dict(weights_with_names, orient='index').to_csv(weight_file)
                            
                            df = pd.read_csv(weight_file, index_col=0)
                            gene_names = df.index.tolist()
                            df = df.T
                            
                            le = LabelEncoder()
                            y = le.fit_transform(df.index.tolist())
                            
                            rf = RandomForestRegressor(n_estimators=50, random_state=42)
                            rf.fit(df.values, y=y)
                            
                            importance = rf.feature_importances_
                            result = pd.DataFrame({
                                'gene_names': gene_names,
                                'feature_importance': importance
                            }).sort_values(by='feature_importance', ascending=False)
                            
                            result.to_csv(localpath + 'results/' + sample_name + f'_{model_type}_characteristics.csv', index=False)
                            
                            loss_r = pd.DataFrame({'Loss': [loss], 'Model_Type': [model_type]})
                            loss_r.to_csv(localpath + 'results/' + sample_name + f'_{model_type}_loss.csv', index=False)
                            
                            print(f"SUCCESS: Completed {model_type} for {sample_name}, Loss: {loss:.4f}")
                            processed_files.append(f"{sample_name}_{model_type}")
                        
                    except Exception as e:
                        print(f"ERROR with {model_type} for {sample_name}: {e}")
                        skipped_files.append(f"{sample_name}_{model_type}")
                        
                        with open(localpath + 'logs/error_log.txt', 'a') as f:
                            f.write(f"{sample_name}_{model_type}: {str(e)}\n")
                    
                    finally:
                        if 'encoder_model' in locals():
                            del encoder_model
                        if 'trained_model' in locals():
                            del trained_model
                        tf.keras.backend.clear_session()
                        gc.collect()
                
                print(f"SUCCESS: Completed all models for {sample_name}")
                
            except Exception as e:
                print(f"ERROR processing {sample_name}: {e}")
                skipped_files.append(sample_name)
                
                with open(localpath + 'logs/error_log.txt', 'a') as f:
                    f.write(f"{sample_name}: {str(e)}\n")
            
            tf.keras.backend.clear_session()
            gc.collect()
    
    summary = pd.DataFrame({
        'Processed': processed_files,
        'Skipped': skipped_files
    })
    summary.to_csv(localpath + 'results/processing_summary.csv', index=False)
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {len(processed_files)} files")
    print(f"Skipped: {len(skipped_files)} files")


if __name__ == "__main__":
    setup_gpus()
    main()
