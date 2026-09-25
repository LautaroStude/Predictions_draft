import os
import numpy as np
import pandas as pd
import utils
import config
from train import predict_ensemble

def main():
    print("==================================================")
    print("      NBA Draft Prediction - 2026 Draft Prediction ")
    print("==================================================")
    
    #  Load trained model weights
    model_path = '/home/lautaro/Draft_prediction/best_model.npz'
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Trained model weights not found at {model_path}.")
        print("Please run 'python train.py' first to train the ensemble.")
        return
        
    print(f"Loading ensemble weights from {model_path}...")
    ensemble_weights = np.load(model_path)
    
    # Auto-detect number of features the model was trained with
    trained_features = None
    for key in ensemble_weights.files:
        if key.startswith('W_0_seed_'):
            trained_features = ensemble_weights[key].shape[0]
            break
            
    if trained_features is None:
        print(f"\n[ERROR] Could not determine input feature dimension from weights.")
        return
        
    print(f"Detected model trained with {trained_features} features.")
    
    #  Load data
    print("\nLoading and preprocessing 2026 prospect dataset...")
    data = utils.prepare_data(select_top_features=trained_features)
    X_2026 = data['X_2026']
    meta_2026 = data['2026_meta']
    
    #  Predict using ensemble average
    print("Predicting draft scores for 2026 prospects using Ensemble...")
    preds = predict_ensemble(X_2026, ensemble_weights, X_2026.shape[1])
    
    # Apply age-based post-processing adjustments
    preds_adjusted = utils.apply_postprocessing(meta_2026, preds)
    
    #  Build results DataFrame
    results = meta_2026.copy()
    results['Pred_Score'] = preds_adjusted
    
    raw_2026 = pd.read_excel('/home/lautaro/Draft_prediction/Datasets/Prospects_2026_BDS.xlsx', sheet_name='Datos básicos')
    results_display = pd.merge(raw_2026, results[['Nombre del jugador', 'Pred_Score']], on='Nombre del jugador')
    
    # Calculate actual age at draft for display
    births_display = pd.to_datetime(results_display['Nacimiento'], errors='coerce', format='mixed')
    results_display['Age'] = 2026 - births_display.dt.year
    
    # Sort by score descending
    results_display = results_display.sort_values(by='Pred_Score', ascending=False)
    results_display = results_display.reset_index(drop=True)
    results_display['Predicted_Pick'] = results_display.index + 1
    
    # Save predictions to CSV
    output_csv = '/home/lautaro/Draft_prediction/predicted_2026_draft.csv'
    results_display.to_csv(output_csv, index=False)
    print(f"Saved full predictions to {output_csv}")
    
    # Print Top 30
    print(f"\n=== PREDICTED 2026 NBA DRAFT ORDER (TOP 30) ===")
    print(f"{'Rank':<5} | {'Player Name':<25} | {'Age':<5} | {'Position':<8} | {'League':<15} | {'College/Team':<25} | {'Score':<8}")
    print("-" * 102)
    
    for i in range(min(30, len(results_display))):
        row = results_display.iloc[i]
        age_str = f"{int(row['Age'])}" if not pd.isna(row['Age']) else "-"
        print(f"{row['Predicted_Pick']:<5} | {row['Nombre del jugador']:<25} | {age_str:<5} | {row['Posición']:<8} | {row['Liga']:<15} | {str(row['Equipo']):<25} | {row['Pred_Score']:<8.4f}")
        
    print("-" * 102)
    print("\nTo see the full draft order, open the generated file:")
    print(f"  {output_csv}")

if __name__ == '__main__':
    main()
