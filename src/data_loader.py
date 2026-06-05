import os
import json
import pandas as pd

def load_data(data_dir="."):
    """
    Loads the games.csv dataset and the champion_info.json mapping.
    Performs basic data cleaning (removing duplicates and nulls).
    
    Returns:
        df: pandas DataFrame containing the cleaned matches.
        champion_mapping: dict mapping champion ID (int) to champion name (str).
    """
    csv_path = os.path.join(data_dir, "games.csv")
    json_path = os.path.join(data_dir, "champion_info.json")
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Ensure games.csv and champion_info.json exist in {data_dir}")

    # Load data
    print("Loading games.csv...")
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    initial_len = len(df)
    df = df.drop_duplicates(subset=['gameId'])
    df = df.dropna()
    print(f"Loaded {len(df)} unique games (removed {initial_len - len(df)} duplicates/nulls).")
    
    # Load champion mapping
    print("Loading champion_info.json...")
    with open(json_path, 'r', encoding='utf-8') as f:
        champ_data = json.load(f)
    
    # Create mapping: id -> name
    champion_mapping = {}
    if 'data' in champ_data:
        for key, info in champ_data['data'].items():
            champion_mapping[info['id']] = info['name']
            
    return df, champion_mapping

if __name__ == "__main__":
    # Test load
    df, cmap = load_data()
    print("Sample champion mapping:", list(cmap.items())[:5])
    print(df.head())
