import os
import pandas as pd
import numpy as np
import rbo
import config

def load_dataset(file_path, id_col):
    """
    Loads an Excel file containing basics, physics, and stats sheets, and merges them.
    """
    xls = pd.ExcelFile(file_path)
    
    # Read sheets
    basics = pd.read_excel(xls, sheet_name='Datos básicos')
    physics = pd.read_excel(xls, sheet_name='Datos físicos')
    stats = pd.read_excel(xls, sheet_name='Datos estadísticos')
    
    # Merge sheets
    df = pd.merge(basics, physics, on=id_col, how='left')
    df = pd.merge(df, stats, on=id_col, how='left')
    
    # Standardize ID column
    df = df.rename(columns={id_col: 'Player_ID'})
    
    return df

def clean_duplicates(df):
    """
    Cleans duplicates within the same draft year and player name.
    Prefers the drafted row over the undrafted row.
    """
    if 'Pick' not in df.columns:
        return df.drop_duplicates(subset=['Nombre del jugador'])
    
    df = df.copy()
    df['_is_undrafted'] = (df['Pick'] == 'Undrafted').astype(int)
    
    # Sort so that drafted players come first, then drop duplicates
    df = df.sort_values(by=['Año', 'Nombre del jugador', '_is_undrafted'])
    df = df.drop_duplicates(subset=['Año', 'Nombre del jugador'], keep='first')
    df = df.drop(columns=['_is_undrafted'])
    
    return df

def process_age(df, year_col=None, default_year=2026):
    """
    Computes age at draft based on draft year and birthdate.
    """
    df = df.copy()
    births = pd.to_datetime(df['Nacimiento'], errors='coerce', format='mixed')
    
    if year_col and year_col in df.columns:
        years = df[year_col]
    else:
        years = default_year
        
    df['Age'] = years - births.dt.year
    return df

def engineer_features(df):
    """
    Applies advanced feature engineering for basketball statistics and physical measurements.
    """
    df = df.copy()
    
    # Wingspan differential (Wingspan - Height)
    if 'Envergadura (Cm)' in df.columns and 'Altura descalzo (Cm)' in df.columns:
        df['Wingspan_Diff'] = df['Envergadura (Cm)'] - df['Altura descalzo (Cm)']
        
    #  Per-minute stats (standardized to 40 minutes with Bayesian shrinkage C = 0.5)
    per_min_cols = ['PTS', 'REB', 'ORB', 'BLQ', 'AST', 'TOV', 'STL', 'PF', 'TC', '2P', '3P', 'TL']
    shrinkage_c = 0.5
    if 'MP' in df.columns:
        for col in per_min_cols:
            if col in df.columns:
                df[col + '_per40'] = df.apply(
                    lambda r: (r[col] / (r['MP'] + shrinkage_c) * 40.0) if r['MP'] > 0 and not pd.isna(r['MP']) else r[col], 
                    axis=1
                )
                
    #  Assist-to-Turnover ratio
    if 'AST' in df.columns and 'TOV' in df.columns:
        df['AST_TOV_Ratio'] = df.apply(
            lambda r: r['AST'] / (r['TOV'] if r['TOV'] > 0 else 0.5) if not pd.isna(r['AST']) else 0.0,
            axis=1
        )
        
    return df

def encode_categorical(train_df, val_df, test_df, df_2026):
    """
    Handles league threshold grouping and one-hot encodes Position, University Experience, and League.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    df_2026 = df_2026.copy()
    
    domestic_pro_hs = ['high school', 'g league', 'overtime elite', 'ote']
    
    def map_league(l):
        if pd.isna(l) or l == 'Desconocido':
            return 'Desconocido'
        l_clean = str(l).strip().lower().replace('-', ' ')
        if 'college' in l_clean:
            return 'College'
        elif any(x in l_clean for x in domestic_pro_hs):
            return 'Domestic_Pro_HS'
        else:
            return 'International'
            
    for df_temp in [train_df, val_df, test_df, df_2026]:
        df_temp['Liga_Grouped'] = df_temp['Liga'].apply(map_league)
        df_temp['Posición'] = df_temp['Posición'].fillna('Desconocido')
        df_temp['Experiencia universitaria'] = df_temp['Experiencia universitaria'].fillna('No corresponde')
        
    cat_cols = ['Posición', 'Experiencia universitaria', 'Liga_Grouped']
    
    cat_dummies_columns = {}
    for col in cat_cols:
        unique_vals = sorted(train_df[col].unique().tolist())
        cat_dummies_columns[col] = unique_vals
        
    def get_dummies_custom(df):
        df_encoded = df.copy()
        for col, vals in cat_dummies_columns.items():
            for val in vals:
                col_name = f"{col}_{val}"
                df_encoded[col_name] = (df[col] == val).astype(float)
        return df_encoded.drop(columns=cat_cols)
    
    train_encoded = get_dummies_custom(train_df)
    val_encoded = get_dummies_custom(val_df)
    test_encoded = get_dummies_custom(test_df)
    df_2026_encoded = get_dummies_custom(df_2026)
    
    return train_encoded, val_encoded, test_encoded, df_2026_encoded

def impute_missing_values(train_df, val_df, test_df, df_2026, numeric_cols):
    """
    Imputes missing values using position-based medians/means.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    df_2026 = df_2026.copy()
    
    #  Imputation based on Posición (to handle missing heights/wingspans logically)
    for df_temp in [train_df, val_df, test_df, df_2026]:
        df_temp['Posición'] = df_temp['Posición'].fillna('Desconocido')
        
    if config.IMPUTATION_STRATEGY == "mean":
        group_stats = train_df.groupby('Posición')[numeric_cols].mean()
        fallback_stats = train_df[numeric_cols].mean()
    else:
        group_stats = train_df.groupby('Posición')[numeric_cols].median()
        fallback_stats = train_df[numeric_cols].median()
        
    for df_temp in [train_df, val_df, test_df, df_2026]:
        for col in numeric_cols:
            for pos in ['G', 'F', 'C', 'Desconocido']:
                val_fill = group_stats.loc[pos, col] if pos in group_stats.index else fallback_stats[col]
                if pd.isna(val_fill):
                    val_fill = fallback_stats[col]
                if pd.isna(val_fill):
                    val_fill = 0.0
                    
                mask = (df_temp['Posición'] == pos) & (df_temp[col].isna())
                df_temp.loc[mask, col] = val_fill
                
            # Final fallback in case of anything remaining
            df_temp[col] = df_temp[col].fillna(fallback_stats[col]).fillna(0.0)
            
    return train_df, val_df, test_df, df_2026

def get_target_score(pick):
    """
    Transforms actual draft Pick into a target score.
    """
    if pick == 'Undrafted' or pd.isna(pick):
        return 0.0
    
    try:
        pick_num = float(pick)
    except ValueError:
        return 0.0
        
    if config.TARGET_TYPE == "linear":
        return max(0.0, (61.0 - pick_num) / 60.0)
    elif config.TARGET_TYPE == "reciprocal":
        return 1.0 / pick_num
    elif config.TARGET_TYPE == "exponential":
        return config.P_TARGET ** (pick_num - 1)
    else:
        return max(0.0, (61.0 - pick_num) / 60.0)

def prepare_data(select_top_features=None):
    """
    Main pipeline to load, clean, split, preprocess, and format the data.
    """
    # Define paths
    hist_path = '/home/lautaro/Draft_prediction/Datasets/Prospects_Historico_BDS.xlsx'
    prosp_2026_path = '/home/lautaro/Draft_prediction/Datasets/Prospects_2026_BDS.xlsx'
    
    # Load raw datasets
    df_hist = load_dataset(hist_path, 'ID_NBA')
    df_2026 = load_dataset(prosp_2026_path, 'ID_Prospect')
    
    # Drop "Press de banca" since it's not present in 2026
    if 'Press de banca' in df_hist.columns:
        df_hist = df_hist.drop(columns=['Press de banca'])
    if 'Press de banca' in df_2026.columns:
        df_2026 = df_2026.drop(columns=['Press de banca'])
        
    # Clean duplicates
    df_hist = clean_duplicates(df_hist)
    df_2026 = clean_duplicates(df_2026)
    
    # Process age
    df_hist = process_age(df_hist, year_col='Año')
    df_2026 = process_age(df_2026, default_year=2026)
    
    # Apply advanced feature engineering
    df_hist = engineer_features(df_hist)
    df_2026 = engineer_features(df_2026)
    
    # Train / Dev / Test Splits (Temporal)
    val_years = config.VALIDATION_YEARS
    test_years = config.TEST_YEARS
    
    train_df = df_hist[~df_hist['Año'].isin(val_years + test_years)].copy()
    val_df = df_hist[df_hist['Año'].isin(val_years)].copy()
    test_df = df_hist[df_hist['Año'].isin(test_years)].copy()
    df_2026 = df_2026.copy()
    
    # Select numeric columns for scaling (exclude IDs, categoricals, targets)
    exclude_cols = ['Player_ID', 'Nombre del jugador', 'Nacimiento', 'Equipo', 'Liga', 'Año', 
                    'Año (Combine)', 'Numero de ronda', 'Pick', 'Target_Score',
                    'Posición', 'Experiencia universitaria', 'Liga_Grouped']
    numeric_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Impute missing values using position-based medians/means
    train_df, val_df, test_df, df_2026 = impute_missing_values(train_df, val_df, test_df, df_2026, numeric_cols)
    
    # Add relative features (subtract position median from key physical and stat features)
    relative_target_cols = [
        'Altura descalzo (Cm)', 'Envergadura (Cm)', 'Alcance de pie (Cm)', 'Peso (Kg)',
        'Salto vertical - Estático (cm)', 'Salto vertical - En carrera (cm)',
        'PTS_per40', 'REB_per40', 'AST_per40', 'BLQ_per40', 'STL_per40', 'TOV_per40',
        '3P%', 'eFG%', 'AST_TOV_Ratio'
    ]
    relative_target_cols = [c for c in relative_target_cols if c in train_df.columns]
    
    for df_temp in [train_df, val_df, test_df, df_2026]:
        for col in relative_target_cols:
            for pos in ['G', 'F', 'C', 'Desconocido']:
                pos_med = train_df.loc[train_df['Posición'] == pos, col].median()
                if pd.isna(pos_med):
                    pos_med = train_df[col].median()
                if pd.isna(pos_med):
                    pos_med = 0.0
                mask = df_temp['Posición'] == pos
                df_temp.loc[mask, f"{col}_relative_to_pos"] = df_temp.loc[mask, col] - pos_med
                
    # Re-evaluate numeric columns to include new relative features for scaling
    numeric_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Scale numerical features based on train parameters
    scale_params = {}
    for col in numeric_cols:
        if config.SCALING_METHOD == "standard":
            mean = train_df[col].mean()
            std = train_df[col].std()
            if std == 0 or pd.isna(std):
                std = 1.0
            scale_params[col] = (mean, std)
        else: # minmax
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            rng = max_val - min_val
            if rng == 0 or pd.isna(rng):
                rng = 1.0
            scale_params[col] = (min_val, rng)
            
    for df_temp in [train_df, val_df, test_df, df_2026]:
        for col in numeric_cols:
            param1, param2 = scale_params[col]
            df_temp[col] = (df_temp[col] - param1) / param2
            
    # Encode categorical features
    train_df, val_df, test_df, df_2026 = encode_categorical(train_df, val_df, test_df, df_2026)
    
    # List of final features to use
    final_exclude = ['Player_ID', 'Nombre del jugador', 'Nacimiento', 'Equipo', 'Liga', 'Año', 
                     'Año (Combine)', 'Numero de ronda', 'Pick', 'Target_Score']
    feature_cols = [c for c in train_df.columns if c not in final_exclude]
    
    # Create target columns (required for correlation calculation)
    train_df['Target_Score'] = train_df['Pick'].apply(get_target_score)
    val_df['Target_Score'] = val_df['Pick'].apply(get_target_score)
    test_df['Target_Score'] = test_df['Pick'].apply(get_target_score)
    
    # Feature selection based on correlation
    num_features = select_top_features if select_top_features is not None else config.SELECT_TOP_FEATURES
    if num_features is not None and num_features > 0:
        corrs = train_df[feature_cols].corrwith(train_df['Target_Score']).abs().sort_values(ascending=False)
        feature_cols = corrs.head(num_features).index.tolist()
    
    # Output features and targets as numpy arrays
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['Target_Score'].values.astype(np.float32).reshape(-1, 1)
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['Target_Score'].values.astype(np.float32).reshape(-1, 1)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['Target_Score'].values.astype(np.float32).reshape(-1, 1)
    
    X_2026 = df_2026[feature_cols].values.astype(np.float32)
    
    return {
        'X_train': X_train, 'y_train': y_train, 'train_meta': train_df,
        'X_val': X_val, 'y_val': y_val, 'val_meta': val_df,
        'X_test': X_test, 'y_test': y_test, 'test_meta': test_df,
        'X_2026': X_2026, '2026_meta': df_2026,
        'feature_cols': feature_cols
    }

def calculate_rbo_for_year(df_year, predictions, p=None):
    """
    Calculates the Rank Biased Overlap (RBO) between the actual draft order
    and the predicted draft order for a given year.
    """
    if p is None:
        p = config.P_RBO
        
    df = df_year.copy()
    df['Pred_Score'] = predictions
    
    true_drafted = df[df['Pick'] != 'Undrafted'].sort_values(by='Pick')
    true_list = true_drafted['Nombre del jugador'].tolist()
    
    K = len(true_list)
    pred_sorted = df.sort_values(by='Pred_Score', ascending=False)
    pred_list = pred_sorted['Nombre del jugador'].head(K).tolist()
    
    if not true_list or not pred_list:
        return 0.0
        
    sim = rbo.RankingSimilarity(true_list, pred_list)
    rbo_val = sim.rbo_ext(p=p)
    return rbo_val

def evaluate_model_rbo(df_meta, predictions, p=None):
    """
    Computes RBO for each year in the meta dataframe and returns their average.
    """
    years = sorted(df_meta['Año'].unique().tolist())
    rbo_scores = {}
    
    for year in years:
        idx = df_meta['Año'] == year
        df_year = df_meta[idx]
        preds_year = predictions[idx].flatten()
        rbo_score = calculate_rbo_for_year(df_year, preds_year, p)
        rbo_scores[year] = rbo_score
        
    avg_rbo = np.mean(list(rbo_scores.values()))
    return avg_rbo, rbo_scores

_cached_pairs = {}

def generate_training_pairs(X, df_meta, max_pairs_per_year=1500, lambdarank=True):
    """
    Generates pairs of indices for pairwise training within the same draft year.
    Each pair (i, j) compares player i and player j.
    If i is better than j (i.e. pick_i < pick_j or i is drafted and j is undrafted), the base label is 1.0.
    Pairs are shuffled and limited to max_pairs_per_year.
    The order of A and B is randomly swapped to prevent network position bias.
    """
    key = (id(df_meta), lambdarank)
    if key in _cached_pairs:
        all_year_pairs = _cached_pairs[key]
    else:
        years = df_meta['Año'].unique()
        all_year_pairs = {}
        
        for year in years:
            year_idx = np.where(df_meta['Año'] == year)[0]
            picks = df_meta.iloc[year_idx]['Pick'].values
            
            # Separate drafted and undrafted indices within this specific year
            drafted_indices = []
            undrafted_indices = []
            pick_vals = []
            
            for idx_in_year, pick in enumerate(picks):
                global_idx = year_idx[idx_in_year]
                if pick == 'Undrafted' or pd.isna(pick):
                    undrafted_indices.append(global_idx)
                else:
                    drafted_indices.append(global_idx)
                    pick_vals.append(float(pick))
                    
            year_pairs = []
            
            # Compare Drafted vs Drafted
            n_drafted = len(drafted_indices)
            for i in range(n_drafted):
                for j in range(i + 1, n_drafted):
                    idx_i = drafted_indices[i]
                    idx_j = drafted_indices[j]
                    pick_i = pick_vals[i]
                    pick_j = pick_vals[j]
                    
                    if lambdarank:
                        w = 1.0 / pick_i + 1.0 / pick_j
                    else:
                        w = 1.0
                        
                    if pick_i < pick_j:
                        year_pairs.append((idx_i, idx_j, w))
                    elif pick_i > pick_j:
                        year_pairs.append((idx_j, idx_i, w))
            
            # Compare Drafted vs Undrafted
            for idx_i, pick_i in zip(drafted_indices, pick_vals):
                if lambdarank:
                    w = 1.0 / pick_i + 1.0 / 61.0
                else:
                    w = 1.0
                for idx_j in undrafted_indices:
                    year_pairs.append((idx_i, idx_j, w))
            
            all_year_pairs[year] = year_pairs
        _cached_pairs[key] = all_year_pairs

    pairs_A = []
    pairs_B = []
    labels = []
    weights = []
    
    for year, year_pairs in all_year_pairs.items():
        if not year_pairs:
            continue
        n_pairs = len(year_pairs)
        if max_pairs_per_year is not None and n_pairs > max_pairs_per_year:
            # Faster sampling using numpy choices
            sampled_indices = np.random.choice(n_pairs, max_pairs_per_year, replace=False)
            sampled_pairs = [year_pairs[i] for i in sampled_indices]
        else:
            sampled_pairs = year_pairs
            
        for idx_a, idx_b, w in sampled_pairs:
            # Randomly swap A and B to balance label distribution
            if np.random.rand() > 0.5:
                pairs_A.append(idx_a)
                pairs_B.append(idx_b)
                labels.append(1.0)
                weights.append(w)
            else:
                pairs_A.append(idx_b)
                pairs_B.append(idx_a)
                labels.append(0.0)
                weights.append(w)
                
    return (np.array(pairs_A, dtype=np.int32), 
            np.array(pairs_B, dtype=np.int32), 
            np.array(labels, dtype=np.float32).reshape(-1, 1),
            np.array(weights, dtype=np.float32).reshape(-1, 1))

def apply_postprocessing(df_meta, predictions):
    """
    Applies age-based and positional adjustments to model predictions using hyperparameters from config.
    """
    adj = predictions.copy()
    
    df_meta_reset = df_meta.reset_index(drop=True)
    for idx, row in df_meta_reset.iterrows():
        if 'Nacimiento' in row and 'Año' in row:
            birth_year = pd.to_datetime(row['Nacimiento'], errors='coerce').year
            year = row['Año']
            raw_age = year - birth_year
        elif 'Nacimiento' in row:
            birth_year = pd.to_datetime(row['Nacimiento'], errors='coerce').year
            raw_age = 2026 - birth_year
        else:
            raw_age = 20.0
            
        if pd.isna(raw_age):
            raw_age = 20.0
            
        score = predictions[idx]
        
        # Apply Young Prospect Boost
        if raw_age <= config.YOUNG_AGE_THRESHOLD:
            if config.YOUNG_SCORE_EXPONENT > 0.0:
                boost = 1.0 + (config.YOUNG_AGE_MULTIPLIER - 1.0) * (score ** config.YOUNG_SCORE_EXPONENT)
            else:
                boost = config.YOUNG_AGE_MULTIPLIER
            adj[idx] *= boost
            
        # Apply Older Prospect Penalty
        elif raw_age >= config.OLD_AGE_THRESHOLD:
            is_forward = False
            if config.EXEMPT_FORWARDS and 'Posición_F' in row:
                is_forward = (row['Posición_F'] == 1.0)
                
            if is_forward:
                continue
                
            if config.OLD_SCORE_EXPONENT > 0.0:
                penalty = 1.0 - (1.0 - config.OLD_AGE_MULTIPLIER) * ((1.0 - score) ** config.OLD_SCORE_EXPONENT)
            else:
                penalty = config.OLD_AGE_MULTIPLIER
            adj[idx] *= penalty
            
    return adj




