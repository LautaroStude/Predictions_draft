import os
import numpy as np
import pandas as pd
import model as nn
import utils
import config

def build_model(input_dim):

    layers = []
    prev_dim = input_dim
    for size in config.HIDDEN_SIZES:
        init_type = 'he' if 'relu' in config.ACTIVATION else 'xavier'
        layers.append(nn.Linear(prev_dim, size, init_type=init_type))
        if config.ACTIVATION == "relu":
            layers.append(nn.ReLU())
        elif config.ACTIVATION == "leaky_relu":
            layers.append(nn.LeakyReLU(alpha=config.LEAKY_RELU_ALPHA))
        elif config.ACTIVATION == "tanh":
            layers.append(nn.Tanh())
        elif config.ACTIVATION == "sigmoid":
            layers.append(nn.Sigmoid())
        if config.DROPOUT_RATE > 0.0:
            layers.append(nn.Dropout(config.DROPOUT_RATE))
        prev_dim = size
        
    layers.append(nn.Linear(prev_dim, 1, init_type='xavier'))
    return nn.Sequential(layers)

def predict_ensemble(X, ensemble_weights, feature_dim):

    preds_list = []
    for seed in config.ENSEMBLE_SEEDS:
        model = build_model(feature_dim)
        model.set_mode('test')
        
        linear_idx = 0
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                layer.W.val = ensemble_weights[f'W_{linear_idx}_seed_{seed}']
                layer.b.val = ensemble_weights[f'b_{linear_idx}_seed_{seed}']
                linear_idx += 1
                
        preds = model.forward(X).flatten()
        preds_list.append(preds)
        
    avg_preds = np.mean(preds_list, axis=0)
    # Apply sigmoid to convert logits to scores between 0 and 1
    clipped_preds = np.clip(avg_preds, -88.0, 88.0)
    avg_preds = 1.0 / (1.0 + np.exp(-clipped_preds))
    return avg_preds

def train_ensemble(X_train, y_train, train_meta, X_val, y_val, val_meta):
    """
    Trains the 5-seed ensemble for the Siamese RankNet formulation.
    """
    ensemble_weights = {}
    
    for seed in config.ENSEMBLE_SEEDS:
        print(f"\nTraining Model Seed: {seed} (SIAMESE)")
        np.random.seed(seed)
        
        model = build_model(X_train.shape[1])
        
        if config.OPTIMIZER == "adam":
            optimizer = nn.Adam(model.params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        else:
            optimizer = nn.SGD(model.params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            
        best_val_rbo = -1.0
        best_val_loss = float('inf')
        best_epoch = -1
        best_val_rbo_at_checkpoint = -1.0
        best_val_loss_at_checkpoint = float('inf')
        best_w = {}
        
        for epoch in range(1, config.EPOCHS + 1):
            model.set_mode('train')
            
            # Generate training pairs dynamically (includes reciprocal rank weights)
            pairs_A, pairs_B, y_pairs, w_pairs = utils.generate_training_pairs(
                X_train, train_meta, config.MAX_PAIRS_PER_YEAR, lambdarank=True
            )
            
            # Shuffle pairs
            pair_indices = np.arange(len(pairs_A))
            np.random.shuffle(pair_indices)
            pairs_A = pairs_A[pair_indices]
            pairs_B = pairs_B[pair_indices]
            y_pairs = y_pairs[pair_indices]
            w_pairs = w_pairs[pair_indices]
            
            # Train in mini-batches of player pairs
            for start_idx in range(0, len(pairs_A), config.BATCH_SIZE):
                end_idx = min(start_idx + config.BATCH_SIZE, len(pairs_A))
                batch_A = pairs_A[start_idx:end_idx]
                batch_B = pairs_B[start_idx:end_idx]
                y_batch_pair = y_pairs[start_idx:end_idx]
                w_batch_pair = w_pairs[start_idx:end_idx]
                
                X_batch_A = X_train[batch_A]
                X_batch_B = X_train[batch_B]
                
                X_batch = np.vstack([X_batch_A, X_batch_B])
                preds_batch = model.forward(X_batch)
                
                B = len(batch_A)
                preds_A = preds_batch[:B]
                preds_B = preds_batch[B:]
                
                # Pairwise sigmoid probability with input clipping to prevent overflow
                diff = preds_A - preds_B
                diff = np.clip(diff, -88.0, 88.0)
                prob = 1.0 / (1.0 + np.exp(-config.SIAMESE_SCALE * diff))
                
                # Gradient of BCE loss wrt difference (s_A - s_B), weighted by reciprocal rank weights
                dDiff = config.SIAMESE_SCALE * (prob - y_batch_pair) * w_batch_pair / B
                
                dLoss_A = dDiff
                dLoss_B = -dDiff
                dLoss = np.vstack([dLoss_A, dLoss_B])
                
                optimizer.zero_grad()
                model.backward(dLoss)
                optimizer.step()
                    
            # Evaluation
            model.set_mode('test')
            if epoch == 1 or epoch % 5 == 0 or epoch == config.EPOCHS:
                val_preds = model.forward(X_val).flatten()
                
                # Pass logits through sigmoid to get bounded predictions for RBO and Loss evaluation
                val_preds_scaled = 1.0 / (1.0 + np.exp(-np.clip(val_preds, -88.0, 88.0)))
                
                overall_val_rbo, _ = utils.evaluate_model_rbo(val_meta, val_preds_scaled)
                val_loss = np.mean((val_preds_scaled - y_val.flatten()) ** 2)
                
                is_best = False
                if config.CHECKPOINT_METRIC == "loss":
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        is_best = True
                else:  # "rbo"
                    if overall_val_rbo > best_val_rbo:
                        best_val_rbo = overall_val_rbo
                        is_best = True
                        
                if is_best:
                    best_epoch = epoch
                    best_val_rbo_at_checkpoint = overall_val_rbo
                    best_val_loss_at_checkpoint = val_loss
                    
                    # Store best weights
                    best_w = {}
                    linear_idx = 0
                    for layer in model.layers:
                        if isinstance(layer, nn.Linear):
                            best_w[f'W_{linear_idx}'] = layer.W.val.copy()
                            best_w[f'b_{linear_idx}'] = layer.b.val.copy()
                            linear_idx += 1
                            
        print(f"Seed {seed} finished. Best Val RBO: {best_val_rbo_at_checkpoint:.4f} (Val Loss: {best_val_loss_at_checkpoint:.5f}) at Epoch {best_epoch}")
        
        # Save best weights for this seed
        for k, v in best_w.items():
            ensemble_weights[f"{k}_seed_{seed}"] = v
            
    return ensemble_weights

def main():
    print("==================================================")
    print("      NBA Draft Prediction - Ensemble Training    ")
    print("==================================================")
    
    # Load and preprocess data
    print("\nLoading and preprocessing datasets...")
    data = utils.prepare_data()
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    train_meta = data['train_meta']
    val_meta = data['val_meta']
    test_meta = data['test_meta']
    
    print(f"Train samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"Val samples:   {X_val.shape[0]}  (Years: {config.VALIDATION_YEARS})")
    print(f"Test samples:  {X_test.shape[0]}  (Years: {config.TEST_YEARS})")
    
    print(f"\nModel Architecture: {config.HIDDEN_SIZES} with {config.ACTIVATION} activation")
    print(f"Target Score formulation: {config.TARGET_TYPE}")
    print(f"Training Loss Type: {config.LOSS_TYPE}")
    
    # Train Siamese RankNet ensemble
    combined_weights = train_ensemble(X_train, y_train, train_meta, X_val, y_val, val_meta)
        
    # Save ensemble weights
    model_save_path = '/home/lautaro/Draft_prediction/best_model.npz'
    np.savez(model_save_path, **combined_weights)
    print(f"\nSaved all ensemble weights to {model_save_path}")
    
    # Final evaluation of the Ensemble average
    print("\n=== FINAL ENSEMBLE EVALUATION ===")
    
    # 1. Validation Set
    val_preds = predict_ensemble(X_val, combined_weights, X_train.shape[1])
    val_mse = np.mean((val_preds - y_val.flatten()) ** 2)
    val_rbo_raw, val_rbo_by_year_raw = utils.evaluate_model_rbo(val_meta, val_preds)
    
    val_preds_post = utils.apply_postprocessing(val_meta, val_preds)
    val_rbo, val_rbo_by_year = utils.evaluate_model_rbo(val_meta, val_preds_post)
    
    print("\nValidation Set:")
    print(f"  Average Val MSE: {val_mse:.5f}")
    print(f"  Raw Val RBO (No adjustments): {val_rbo_raw:.4f}")
    print(f"  Average Val RBO (With Age adjustments): {val_rbo:.4f} (p={config.P_RBO})")
    for year, rbo_val in val_rbo_by_year.items():
        print(f"    Year {year} RBO: {rbo_val:.4f}")
        
    # 2. Test Set
    test_preds = predict_ensemble(X_test, combined_weights, X_train.shape[1])
    test_mse = np.mean((test_preds - y_test.flatten()) ** 2)
    test_rbo_raw, test_rbo_by_year_raw = utils.evaluate_model_rbo(test_meta, test_preds)
    
    test_preds_post = utils.apply_postprocessing(test_meta, test_preds)
    test_rbo, test_rbo_by_year = utils.evaluate_model_rbo(test_meta, test_preds_post)
    
    print("\nTest Set:")
    print(f"  Average Test MSE: {test_mse:.5f}")
    print(f"  Raw Test RBO (No adjustments): {test_rbo_raw:.4f}")
    print(f"  Average Test RBO (With Age adjustments): {test_rbo:.4f} (p={config.P_RBO})")
    for year, rbo_val in test_rbo_by_year.items():
        print(f"    Year {year} RBO: {rbo_val:.4f}")

if __name__ == '__main__':
    main()
