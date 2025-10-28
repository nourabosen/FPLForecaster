import os
import pandas as pd
import requests
from io import StringIO
from urllib.parse import quote
from datetime import datetime

# Base URL for raw files from GitHub
raw_base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/"

def sanitize_name(name):
    """Removes non-ASCII characters from a string to make it a safe filename."""
    return name.encode('ascii', 'ignore').decode('ascii')

def get_last_modified_date(url):
    """Get the last modified date of a remote file."""
    try:
        response = requests.head(url)
        response.raise_for_status()
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            return datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
    except requests.RequestException:
        return None
    return None

def is_data_up_to_date(local_path, remote_url):
    """Check if the local data is up-to-date with the remote source."""
    if not os.path.exists(local_path):
        return False
    
    remote_last_modified = get_last_modified_date(remote_url)
    if not remote_last_modified:
        return True  # Assume up-to-date if we can't get remote date

    local_last_modified = datetime.fromtimestamp(os.path.getmtime(local_path))
    return local_last_modified >= remote_last_modified

def load_csv_from_url(relative_path):
    """
    Given a relative path, construct the raw URL, download the CSV,
    and load it into a DataFrame using UTF-8 encoding.
    """
    url = raw_base_url + relative_path
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8'
        df = pd.read_csv(StringIO(response.text), encoding='utf-8')
        print("Loaded '{}' with shape {}".format(relative_path, df.shape))
        return df
    except Exception as e:
        print("Error loading '{}': {}".format(relative_path, e))
        return None

def save_df_to_local(df, local_path):
    """Save DataFrame to a local CSV file, creating directories if necessary."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False, encoding='utf-8')
    print("Saved file to", local_path)

def ingest_data(force_download=False, check_for_updates=False):
    """
    Downloads required data files from GitHub and saves them locally.
    This includes key files, Understat data, and all players' gameweek data.
    If force_download is False, it will skip downloading if the data already exists.
    """
    base_local_dir = "data"
    os.makedirs(base_local_dir, exist_ok=True)

    # --- Download key files from the root of the data directory ---
    key_files = ["teams.csv", "fixtures.csv", "player_idlist.csv", "players_raw.csv"]
    for file_name in key_files:
        local_path = os.path.join(base_local_dir, file_name)
        remote_url = raw_base_url + file_name
        
        needs_download = not os.path.exists(local_path) or force_download
        if check_for_updates and not needs_download:
            if not is_data_up_to_date(local_path, remote_url):
                needs_download = True
                print(f"'{file_name}' is outdated, re-downloading.")

        if needs_download:
            df = load_csv_from_url(file_name)
            if df is not None:
                save_df_to_local(df, local_path)
        else:
            print(f"Skipping '{file_name}', already exists and is up-to-date.")

    # --- Ingest Understat files using the GitHub API ---
    understat_local_dir = os.path.join(base_local_dir, "understat")
    os.makedirs(understat_local_dir, exist_ok=True)
    if not os.listdir(understat_local_dir) or force_download:
        api_url = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/2025-26/understat"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()
        except Exception as e:
            print(f"Warning: Could not access GitHub API for Understat files, skipping this step: {e}")
            files = [] # Continue without understat data
        for file in files:
            if file.get('type') == 'file' and file.get('name', '').endswith('.csv'):
                name = file.get('name')
                sanitized_name = sanitize_name(name)
                local_file_path = os.path.join(understat_local_dir, sanitized_name)
                if not os.path.exists(local_file_path) or force_download:
                    download_url = file.get('download_url')
                    try:
                        df = pd.read_csv(download_url, encoding='utf-8')
                        save_df_to_local(df, local_file_path)
                    except Exception as e:
                        safe_name = name.encode('utf-8', 'replace').decode('utf-8')
                        print("Error saving Understat file '{}': {}".format(safe_name, e))
                else:
                    print(f"Skipping Understat file '{sanitized_name}', already exists.")
    else:
        print("Skipping Understat files, directory is not empty.")

    # --- Ingest all players' gameweek data ---
    players_local_dir = os.path.join(base_local_dir, "players")
    os.makedirs(players_local_dir, exist_ok=True)
    # Load the local player_idlist file
    player_idlist_path = os.path.join(base_local_dir, "player_idlist.csv")
    if os.path.exists(player_idlist_path):
        player_idlist_df = pd.read_csv(player_idlist_path)
    else:
        print("Local player_idlist.csv not found.")
        return

    for idx, row in player_idlist_df.iterrows():
        # Construct folder name in the format "FirstName_SecondName_ID"
        original_folder_name = f"{row['first_name']}_{row['second_name']}_{int(row['id'])}"
        
        # Sanitize names for local folder creation
        first_name_sanitized = sanitize_name(str(row['first_name']))
        second_name_sanitized = sanitize_name(str(row['second_name']))
        sanitized_folder_name = f"{first_name_sanitized}_{second_name_sanitized}_{int(row['id'])}"
        
        folder_path = os.path.join(players_local_dir, sanitized_folder_name)
        local_file_path = os.path.join(folder_path, "gw.csv")

        if not os.path.exists(local_file_path) or force_download:
            # URL encode the folder name for the request
            relative_path = f"players/{quote(original_folder_name)}/gw.csv"
            
            df = load_csv_from_url(relative_path)
            if df is not None:
                # Save the file preserving folder structure: data/players/<folder_name>/gw.csv
                os.makedirs(folder_path, exist_ok=True)
                df['player_id'] = row['id']
                # If a gameweek column is missing, add one (assumes row order reflects gameweeks)
                if 'gameweek' not in df.columns:
                    df = df.reset_index().rename(columns={'index': 'gameweek'})
                    df['gameweek'] = df['gameweek'] + 1
                save_df_to_local(df, local_file_path)
        else:
            print(f"Skipping player data for '{sanitized_folder_name}', already exists.")

    print("\nData ingestion complete. All files are saved in the 'data' directory.")

if __name__ == "__main__":
    ingest_data()
