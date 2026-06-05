import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_champion_features(df):
    """
    Creates a sparse-like representation of team compositions.
    +1 if Team 1 picked the champion
    -1 if Team 2 picked the champion
    0 otherwise
    """
    print("Engineering champion features...")
    # Find all unique champion IDs across the dataset to define column space
    champ_cols = [f't1_champ{i}id' for i in range(1, 6)] + [f't2_champ{i}id' for i in range(1, 6)]
    unique_champs = pd.unique(df[champ_cols].values.ravel())
    unique_champs.sort()
    
    # Initialize zero matrix
    champ_features = pd.DataFrame(0, index=df.index, columns=[f'champ_{int(c)}' for c in unique_champs])
    
    for i in range(1, 6):
        # Add 1 for team 1
        t1_champs = df[f't1_champ{i}id']
        for idx, val in t1_champs.items():
            champ_features.at[idx, f'champ_{int(val)}'] = 1
            
        # Subtract 1 for team 2
        t2_champs = df[f't2_champ{i}id']
        for idx, val in t2_champs.items():
            champ_features.at[idx, f'champ_{int(val)}'] = -1

    return champ_features

def prepare_early_game_features(df):
    """
    Prepares features available early in the game to avoid data leakage.
    Includes:
    - Champion compositions
    - First objectives (Blood, Tower, Dragon, Rift Herald)
    """
    print("Preparing Early-Game features...")
    # Base early objectives
    # Values are 0 (none), 1 (Team 1), 2 (Team 2)
    early_cols = ['firstBlood', 'firstTower', 'firstDragon', 'firstRiftHerald']
    X_early = df[early_cols].copy()
    
    # Map 1 -> 1, 2 -> -1, 0 -> 0 to make it symmetric
    for col in early_cols:
        X_early[col] = X_early[col].map({0: 0, 1: 1, 2: -1})
    
    # Add champion features
    champ_feats = get_champion_features(df)
    X = pd.concat([X_early, champ_feats], axis=1)
    
    # Target (1 = Team 1 wins, 0 = Team 2 wins)
    y = df['winner'].map({1: 1, 2: 0})
    
    return X, y

def prepare_full_match_features(df):
    """
    Prepares full match features including end-of-game objective counts.
    (Contains data leakage relative to early prediction).
    """
    print("Preparing Full-Match features...")
    # Get early features first
    X_early, y = prepare_early_game_features(df)
    
    # Add late game objectives
    late_first = ['firstInhibitor', 'firstBaron']
    X_late = df[late_first].copy()
    for col in late_first:
        X_late[col] = X_late[col].map({0: 0, 1: 1, 2: -1})
        
    # Add total counts (Team 1 count - Team 2 count)
    diff_feats = pd.DataFrame(index=df.index)
    diff_feats['tower_diff'] = df['t1_towerKills'] - df['t2_towerKills']
    diff_feats['inhibitor_diff'] = df['t1_inhibitorKills'] - df['t2_inhibitorKills']
    diff_feats['baron_diff'] = df['t1_baronKills'] - df['t2_baronKills']
    diff_feats['dragon_diff'] = df['t1_dragonKills'] - df['t2_dragonKills']
    diff_feats['herald_diff'] = df['t1_riftHeraldKills'] - df['t2_riftHeraldKills']
    
    X = pd.concat([X_early, X_late, diff_feats], axis=1)
    return X, y

def get_train_test_splits(X, y, test_size=0.2, random_state=42):
    """Splits data and standardizes features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
