import numpy as np
import pandas as pd
import model as nn
import utils
import config
import train

def evaluate_params(hidden_sizes, lr, wd, dropout, loss_weight_factor, target_type):
    # Temp config override
    config.HIDDEN_SIZES = hidden_sizes
    config.LEARNING_RATE = lr
    config.WEIGHT_DECAY = wd
    config.DROPOUT_RATE = dropout
    config.LOSS_WEIGHT_FACTOR = loss_weight_factor
    config.TARGET_TYPE = target_type
    
    # Reload data
    data = utils.prepare_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    train_meta = data['train_meta']
    val_meta = data['val_meta']
    test_meta = data['test_meta']
    
    # Build model
    model = train.build_model(X_train.shape[1])
    
    if config.OPTIMIZER == "adam":
        optimizer = nn.Adam(model.params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = nn.SGD(model.params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
    best_val_rbo = -1.0
    best_val_loss = float('inf')
    best_test_rbo = -1.0
    best_epoch = -1
    
    for epoch in range(1, 120 + 1):
        model.set_mode('train')
        
        if config.LOSS_TYPE == "siamese":
            # Generate training pairs dynamically for this epoch
            pairs_A, pairs_B, y_pairs, _ = utils.generate_training_pairs(
                X_train, train_meta, config.MAX_PAIRS_PER_YEAR
            )
            
            # Shuffle pairs
            pair_indices = np.arange(len(pairs_A))
            np.random.shuffle(pair_indices)
            pairs_A = pairs_A[pair_indices]
            pairs_B = pairs_B[pair_indices]
            y_pairs = y_pairs[pair_indices]
            
            # Train in mini-batches of player pairs
            for start_idx in range(0, len(pairs_A), config.BATCH_SIZE):
                end_idx = min(start_idx + config.BATCH_SIZE, len(pairs_A))
                batch_A = pairs_A[start_idx:end_idx]
                batch_B = pairs_B[start_idx:end_idx]
                y_batch_pair = y_pairs[start_idx:end_idx]
                
                X_batch_A = X_train[batch_A]
                X_batch_B = X_train[batch_B]
                
                # Concatenate to perform a single forward pass
                X_batch = np.vstack([X_batch_A, X_batch_B])
                preds_batch = model.forward(X_batch)
                
                B = len(batch_A)
                preds_A = preds_batch[:B]
                preds_B = preds_batch[B:]
                
                # Pairwise sigmoid probability: P(A > B)
                diff = preds_A - preds_B
                prob = 1.0 / (1.0 + np.exp(-config.SIAMESE_SCALE * diff))
                
                # Gradient of BCE loss wrt difference (s_A - s_B) is (prob - y) / B
                # Chain rule with sigmoid(scale * diff) introduces SIAMESE_SCALE
                dDiff = config.SIAMESE_SCALE * (prob - y_batch_pair) / B
                
                dLoss_A = dDiff
                dLoss_B = -dDiff
                
                # Concatenate gradients back
                dLoss = np.vstack([dLoss_A, dLoss_B])
                
                optimizer.zero_grad()
                model.backward(dLoss)
                optimizer.step()
        else:
            # Standard MSE training
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train_sh = X_train[indices]
            y_train_sh = y_train[indices]
            
            for start_idx in range(0, len(X_train), config.BATCH_SIZE):
                end_idx = min(start_idx + config.BATCH_SIZE, len(X_train))
                X_batch = X_train_sh[start_idx:end_idx]
                y_batch = y_train_sh[start_idx:end_idx]
                
                preds = model.forward(X_batch)
                weights = 1.0 + config.LOSS_WEIGHT_FACTOR * y_batch
                dLoss = 2.0 * (preds - y_batch) * weights / len(X_batch)
                
                optimizer.zero_grad()
                model.backward(dLoss)
                optimizer.step()
            
        # Eval
        model.set_mode('test')
        if epoch % 5 == 0 or epoch == 1:
            val_preds = model.forward(X_val)
            val_rbo, _ = utils.evaluate_model_rbo(val_meta, val_preds)
            val_loss = np.mean((val_preds - y_val) ** 2)
            
            is_best = False
            if config.CHECKPOINT_METRIC == "loss":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
            else: # "rbo"
                if val_rbo > best_val_rbo:
                    best_val_rbo = val_rbo
                    is_best = True
                    
            if is_best:
                best_val_rbo = val_rbo
                best_epoch = epoch
                # Evaluate test RBO at this best validation epoch
                test_preds = model.forward(X_test)
                test_rbo, _ = utils.evaluate_model_rbo(test_meta, test_preds)
                best_test_rbo = test_rbo
                
    return best_val_rbo, best_test_rbo, best_epoch

def main():
    print("Starting hyperparameter grid search...")
    
    hidden_options = [
        [64],
        [32, 16],
        [64, 32],
        [32]
    ]
    lr_options = [0.01, 0.001, 0.0005]
    wd_options = [0.0, 0.01, 0.05]
    dropout_options = [0.2, 0.3]
    weight_factor_options = [2.0, 5.0, 10.0]
    target_options = ["linear", "reciprocal"]
    
    results = []
    
    # Randomly sample configurations to search efficiently
    np.random.seed(42)
    
    import itertools
    combinations = list(itertools.product(
        hidden_options, lr_options, wd_options, dropout_options, weight_factor_options, target_options
    ))
    
    # Shuffle and pick 25 random combinations to avoid taking too much time
    np.random.shuffle(combinations)
    sampled_combinations = combinations[:25]
    
    print(f"Testing {len(sampled_combinations)} random parameter configurations out of {len(combinations)} total...")
    
    for i, (hidden, lr, wd, drop, wf, target) in enumerate(sampled_combinations):
        val_rbo, test_rbo, epoch = evaluate_params(hidden, lr, wd, drop, wf, target)
        print(f"Config {i+1:2d}/25 | Hidden={str(hidden):<10} | LR={lr:<6} | WD={wd:<4} | Drop={drop:<3} | WeightFactor={wf:<3} | Val RBO={val_rbo:.4f} | Test RBO={test_rbo:.4f} (Epoch {epoch})")
        results.append({
            'hidden': hidden, 'lr': lr, 'wd': wd, 'drop': drop, 'wf': wf, 'target': target,
            'val_rbo': val_rbo, 'test_rbo': test_rbo, 'epoch': epoch
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='val_rbo', ascending=False)
    
    print("\n=== TOP 5 CONFIGURATIONS (Sorted by Validation RBO) ===")
    for idx, row in df_res.head(5).iterrows():
        print(f"Val RBO: {row['val_rbo']:.4f} | Test RBO: {row['test_rbo']:.4f} | Hidden: {row['hidden']} | LR: {row['lr']} | WD: {row['wd']} | Drop: {row['drop']} | WeightFactor: {row['wf']} | Epoch: {row['epoch']}")
        
    # Save the absolute best config parameters
    best_config = df_res.iloc[0]
    print(f"\nRecommended Optimal Config:")
    print(f"  HIDDEN_SIZES = {best_config['hidden']}")
    print(f"  LEARNING_RATE = {best_config['lr']}")
    print(f"  WEIGHT_DECAY = {best_config['wd']}")
    print(f"  DROPOUT_RATE = {best_config['drop']}")
    print(f"  LOSS_WEIGHT_FACTOR = {best_config['wf']}")
    print(f"  TARGET_TYPE = '{best_config['target']}'")

if __name__ == '__main__':
    main()
