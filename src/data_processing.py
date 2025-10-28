import os
import ast
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler

def process_data() -> Dict[str, Any]:
    """
    Loads locally saved CSV files from the data directory,
    aggregates and processes them, normalizes feature columns,
    and creates sequences for LSTM.
    Splits data into training and validation sets dynamically.
    Returns a dictionary containing key DataFrames and the LSTM input (X_train, y_train, X_val, y_val).
    """
    base_local_dir = "data"
    teams_path = os.path.join(base_local_dir, "teams.csv")
    fixtures_path = os.path.join(base_local_dir, "fixtures.csv")
    player_idlist_path = os.path.join(base_local_dir, "player_idlist.csv")
    playerraw_path = os.path.join(base_local_dir, "players_raw.csv")
    players_local_dir = os.path.join(base_local_dir, "players")

    # Load key files
    teams_df = pd.read_csv(teams_path) if os.path.exists(teams_path) else None
    fixtures_df = pd.read_csv(fixtures_path) if os.path.exists(fixtures_path) else None
    player_idlist_df = pd.read_csv(player_idlist_path) if os.path.exists(player_idlist_path) else None
    playerraw_df = pd.read_csv(playerraw_path) if os.path.exists(playerraw_path) else None

    # Aggregate player gameweek data from the players folder
    player_gw_dfs = []
    if os.path.exists(players_local_dir):
        for folder in os.listdir(players_local_dir):
            folder_path = os.path.join(players_local_dir, folder)
            if os.path.isdir(folder_path):
                gw_file = os.path.join(folder_path, "gw.csv")
                if os.path.exists(gw_file):
                    df = pd.read_csv(gw_file)
                    player_gw_dfs.append(df)
        if player_gw_dfs:
            player_gw_df = pd.concat(player_gw_dfs, ignore_index=True)
            print("Aggregated player gameweek data shape:", player_gw_df.shape)
        else:
            player_gw_df = pd.DataFrame()
    else:
        player_gw_df = pd.DataFrame()

    # --- Feature Engineering ---
    if not player_gw_df.empty:
        # Rename columns for consistency
        player_gw_df = player_gw_df.rename(columns={'total_points': 'points', 'mins': 'minutes', 'goals_scored': 'goals'})

        # Calculate 'form' (rolling average of points over last 5 gameweeks)
        player_gw_df = player_gw_df.sort_values(by=['player_id', 'gameweek'])
        player_gw_df['form'] = player_gw_df.groupby('player_id')['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(0)

        # Calculate 'fixture_difficulty'
        if teams_df is not None:
            team_strength = teams_df.set_index('id')[['strength_defence_home', 'strength_defence_away']].to_dict('index')
            def get_fixture_difficulty(row):
                opponent_team_id = row['opponent_team']
                was_home = row['was_home']
                if opponent_team_id in team_strength:
                    return team_strength[opponent_team_id]['strength_defence_away'] if was_home else team_strength[opponent_team_id]['strength_defence_home']
                return 1200  # Default average difficulty
            player_gw_df['fixture_difficulty'] = player_gw_df.apply(get_fixture_difficulty, axis=1)
        else:
            player_gw_df['fixture_difficulty'] = 1200

        # Add more features
        player_gw_df['creativity'] = player_gw_df.get('creativity', 0)
        player_gw_df['influence'] = player_gw_df.get('influence', 0)
        player_gw_df['threat'] = player_gw_df.get('threat', 0)

    # Split data into training and validation sets based on gameweek
    if not player_gw_df.empty:
        max_gameweek = player_gw_df['gameweek'].max()
        validation_gameweek = max_gameweek - 2
        if validation_gameweek <= 0:
            train_df = player_gw_df.copy()
            val_df = pd.DataFrame()
        else:
            train_df = player_gw_df[player_gw_df['gameweek'] < validation_gameweek].copy()
            val_df = player_gw_df[player_gw_df['gameweek'] >= validation_gameweek].copy()
    else:
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()


    # Normalize features
    feature_cols = ['minutes', 'goals', 'assists', 'form', 'fixture_difficulty', 'creativity', 'influence', 'threat']
    scaler = StandardScaler()
    
    X_train, y_train, X_val, y_val = None, None, None, None

    if not train_df.empty:
        # Fit scaler on training data only to prevent data leakage
        train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
        X_train, y_train = create_sequences(train_df, seq_length=3, feature_cols=feature_cols)
    
    # Transform validation data using the same scaler
    if not val_df.empty:
        val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
        X_val, y_val = create_sequences(val_df, seq_length=3, feature_cols=feature_cols)

    return {
        "teams_df": teams_df,
        "fixtures_df": fixtures_df,
        "player_idlist_df": player_idlist_df,
        "playerraw_df": playerraw_df,
        "player_gw_df": player_gw_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "scaler": scaler
    }

def create_sequences(df: pd.DataFrame, seq_length: int = 3, 
                     feature_cols: list = ['minutes', 'goals', 'assists', 'form', 'fixture_difficulty', 'creativity', 'influence', 'threat'], 
                     target_col: str = 'points') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    sequences = []
    targets = []
    if df.empty:
        return None, None
    
    df = df.sort_values(['player_id', 'gameweek'])
    for player in df['player_id'].unique():
        player_data = df[df['player_id'] == player].reset_index(drop=True)
        if len(player_data) <= seq_length:
            continue
        
        data_array = player_data[feature_cols].values
        target_array = player_data[target_col].values
        
        for i in range(len(player_data) - seq_length):
            sequences.append(data_array[i:i+seq_length])
            targets.append(target_array[i+seq_length])
            
    return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    process_data()
