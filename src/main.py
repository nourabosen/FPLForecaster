import os
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import streamlit as st
from data_ingestion import ingest_data
from data_processing import process_data
from model import train_model, predict_next_gameweek, save_trained_model, load_trained_model
import json

# Disable GPU to avoid JIT compilation errors on some setups
try:
    tf.config.set_visible_devices([], 'GPU')
    logging.info("GPU disabled for TensorFlow.")
except (ValueError, RuntimeError) as e:
    logging.warning(f"Could not disable GPU: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fpl_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_top100_price_vs_points(predictions: dict, player_gw_df: pd.DataFrame,
                                player_idlist_df: pd.DataFrame, playerraw_df: pd.DataFrame,
                                teams_df: pd.DataFrame):
    pred_df = pd.DataFrame(list(predictions.items()), columns=['player_id', 'predicted_points'])
    pred_df['player_id'] = pred_df['player_id'].astype(int)
    
    latest_player = player_gw_df.sort_values('gameweek').groupby('player_id').tail(1)
    latest_player = latest_player[['player_id', 'value']]
    latest_player['dollar_value'] = latest_player['value'] / 10.0
    
    merged = pd.merge(pred_df, latest_player, on='player_id', how='inner')
    
    # Merge with player_idlist to get full names
    merged = pd.merge(merged, player_idlist_df[['id', 'first_name', 'second_name']],
                      left_on='player_id', right_on='id', how='left')
    merged['full_name'] = merged['first_name'] + " " + merged['second_name']

    if playerraw_df is not None:
        # Combine merges for playerraw_df
        merged = pd.merge(merged, playerraw_df[['id', 'team', 'element_type']], on='id', how='left')
        
        if teams_df is not None:
            merged = pd.merge(merged, teams_df[['id', 'name']],
                              left_on='team', right_on='id', how='left', suffixes=('', '_team'))
            merged['full_name_with_club'] = merged['full_name'] + " (" + merged['name'] + ")"
        else:
            merged['full_name_with_club'] = merged['full_name']
    else:
        merged['full_name_with_club'] = merged['full_name']
        merged['element_type'] = 0
    
    position_colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}
    merged['color'] = merged['element_type'].map(position_colors)
    merged['color'] = merged['color'].fillna('gray')
    
    top100 = merged.sort_values('predicted_points', ascending=False).head(100)
    
    fig = px.scatter(
        top100, x='dollar_value', y='predicted_points',
        color='color', hover_name='full_name_with_club',
        labels={'dollar_value': 'Price ($)', 'predicted_points': 'Predicted Fantasy Points'},
        title='Top 100 Players by Predicted Fantasy Points vs Price'
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    return fig

def main():
    st.title("FPL Point Prediction Dashboard")

    # --- Data Management in Main Panel ---
    st.markdown("### Data Management")
    data_exists = os.path.exists("data")
    if data_exists:
        st.success("Data directory found.")
    else:
        st.warning("Data directory not found.")

    col1, col2 = st.columns(2)
    force_download = col1.checkbox("Force data re-download", help="Force a download even if data files exist.")
    check_for_updates = col2.checkbox("Check for updates", help="Check for new data from the remote source.")

    if 'run_pipeline' not in st.session_state:
        st.session_state.run_pipeline = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'history' not in st.session_state:
        st.session_state.history = None

    if st.button("Run Full Pipeline"):
        st.session_state.run_pipeline = True
        st.session_state.data = None  # Invalidate cached data
        st.session_state.predictions = None

    if st.session_state.run_pipeline:
        if st.session_state.data is None or st.session_state.predictions is None:
            model = load_trained_model()
            predictions = None
            if os.path.exists("predictions.json"):
                try:
                    with open("predictions.json", "r") as f:
                        predictions = json.load(f)
                        predictions = {int(k): v for k, v in predictions.items()}
                except json.JSONDecodeError:
                    logger.warning("Could not decode predictions.json. File might be corrupted.")
                    predictions = None

            if model is None or predictions is None or force_download or check_for_updates:
                logger.info("Starting data ingestion...")
                with st.spinner("Ingesting data..."):
                    ingest_data(force_download=force_download, check_for_updates=check_for_updates)
                st.success("Data ingestion completed.")
                
                logger.info("Starting data processing...")
                with st.spinner("Processing data..."):
                    data = process_data()
                st.success("Data processing completed.")
                
                if data["X_train"] is None or data["y_train"] is None:
                    st.error("No training data available. Exiting.")
                    return
                    
                logger.info("Starting model training...")
                with st.spinner("Training model..."):
                    model, history = train_model(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
                
                if model is None:
                    st.error("Model training failed, not enough data to create sequences. Try a larger dataset.")
                    return
                    
                st.success("Model training completed.")
                save_trained_model(model)
                
                logger.info("Predicting next gameweek fantasy points...")
                with st.spinner("Generating predictions..."):
                    predictions = predict_next_gameweek(model=model, player_gw_df=data["player_gw_df"], scaler=data["scaler"])
                st.success("Prediction completed.")
                # Convert keys to standard int for JSON serialization
                predictions_to_save = {int(k): float(v) for k, v in predictions.items()}
                with open("predictions.json", "w") as f:
                    json.dump(predictions_to_save, f)

                st.session_state.data = data
                st.session_state.predictions = predictions
                st.session_state.history = history
            else:
                logger.info("Loading cached model and predictions.")
                st.session_state.data = process_data()
                st.session_state.predictions = predictions
                st.session_state.history = None # No history for cached model
        
        data = st.session_state.data
        predictions = st.session_state.predictions
        history = st.session_state.history

        pred_df = pd.DataFrame(list(predictions.items()), columns=['player_id', 'predicted_points'])
        pred_df['player_id'] = pred_df['player_id'].astype(int)
        pred_df = pd.merge(pred_df, data["playerraw_df"], left_on='player_id', right_on='id', how='inner')
        pred_df['price'] = pred_df['now_cost'] / 10.0

        # --- Dashboard UI ---
        st.subheader("Dashboard")
        
        # Summary Metrics
        top_scorer = pred_df.loc[pred_df['predicted_points'].idxmax()]
        avg_predicted_points = pred_df['predicted_points'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Top Scorer", f"{top_scorer['first_name']} {top_scorer['second_name']}", f"{top_scorer['predicted_points']:.2f} pts")
        col2.metric("Avg Predicted Points", f"{avg_predicted_points:.2f}")
        if history and 'val_mae' in history.history and history.history['val_mae']:
            col3.metric("Model MAE", f"{history.history['val_mae'][-1]:.2f}")

        # Tabs
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Overview"

        tab1, tab2, tab3 = st.tabs(["Overview", "Predictions", "Team Builder"])

        with tab1:
            st.subheader("Top 100 Players: Price vs Predicted Points")
            fig = plot_top100_price_vs_points(predictions, data["player_gw_df"], data["player_idlist_df"], data["playerraw_df"], data["teams_df"])
            st.plotly_chart(fig)

        with tab2:
            st.subheader("Predicted Players by Position")
            
            # Filters
            teams = ["All"] + sorted(data["teams_df"]['name'].unique())
            team = st.selectbox("Select Team", teams, key="team_select")
            
            positions = {1: 'Goalkeepers', 2: 'Defenders', 3: 'Midfielders', 4: 'Forwards'}
            for pos_id, pos_name in positions.items():
                st.write(f"**{pos_name}**")
                pos_df = pred_df[pred_df['element_type'] == pos_id]
                if not pos_df.empty:
                    max_price = pos_df['price'].max()
                    price_range = st.slider(f"Max Price for {pos_name} (£)", 4.0, max_price, max_price, key=f"price_slider_{pos_id}")
                    
                    filtered_df = pos_df[pos_df['price'] <= price_range]
                    if team != "All":
                        team_id = data["teams_df"][data["teams_df"]["name"] == team]["id"].iloc[0]
                        filtered_df = filtered_df[filtered_df["team"] == team_id]

                    # Merge with teams_df to get team names
                    filtered_df = pd.merge(filtered_df, data["teams_df"], left_on='team', right_on='id', how='left', suffixes=('', '_team'))
                    filtered_df.rename(columns={'name': 'team_name'}, inplace=True)
                    top_players = filtered_df.sort_values('predicted_points', ascending=False).head(20)
                    st.dataframe(top_players[['first_name', 'second_name', 'team_name', 'predicted_points', 'price']])
            
            st.download_button("Download Predictions", pred_df.to_csv().encode('utf-8'), "predictions.csv")

        with tab3:
            st.subheader("Dream Team Builder")
            budget = st.slider("Budget (£m)", 80.0, 100.0, 100.0)
            formation = st.selectbox("Formation", ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"])
            
            if st.button("Build Dream Team"):
                dream_team, total_cost, total_points = build_dream_team(pred_df, budget, formation)
                if dream_team is not None and not dream_team.empty:
                    st.write(f"**Total Cost:** £{total_cost:.1f}m")
                    st.write(f"**Total Predicted Points:** {total_points:.2f}")
                    
                    # Merge with teams_df to get team names
                    dream_team = pd.merge(dream_team, data["teams_df"], left_on='team', right_on='id', how='left', suffixes=('', '_team'))
                    dream_team.rename(columns={'name': 'team_name'}, inplace=True)
                    st.dataframe(dream_team[['first_name', 'second_name', 'team_name', 'predicted_points', 'price']])
                else:
                    st.error("Could not build a dream team with the selected formation. Not enough players available in the data.")

def build_dream_team(pred_df, budget=100.0, formation="3-4-3"):
    formation_map = {
        "3-4-3": {2: 3, 3: 4, 4: 3},
        "3-5-2": {2: 3, 3: 5, 4: 2},
        "4-3-3": {2: 4, 3: 3, 4: 3},
        "4-4-2": {2: 4, 3: 4, 4: 2},
        "4-5-1": {2: 4, 3: 5, 4: 1},
        "5-3-2": {2: 5, 3: 3, 4: 2},
        "5-4-1": {2: 5, 3: 4, 4: 1},
    }
    
    pred_df['value'] = pred_df['predicted_points'] / pred_df['price']
    
    team = []
    
    # Goalkeeper
    gks = pred_df[pred_df['element_type'] == 1].sort_values('predicted_points', ascending=False)
    if gks.empty: return None, 0, 0
    team.append(gks.iloc[0].to_dict())
    
    # Outfield players
    for pos_id, count in formation_map[formation].items():
        players = pred_df[pred_df['element_type'] == pos_id].sort_values('predicted_points', ascending=False)
        if len(players) < count: return None, 0, 0
        team.extend(players.head(count).to_dict('records'))

    team_df = pd.DataFrame(team)
    total_cost = team_df['price'].sum()

    # Adjust for budget
    while total_cost > budget:
        team_df = team_df.sort_values('value', ascending=True)
        player_to_replace = team_df.iloc[0]
        
        replacements = pred_df[(pred_df['element_type'] == player_to_replace['element_type']) & 
                               (pred_df['price'] < player_to_replace['price']) &
                               (~pred_df['id'].isin(team_df['id']))]
        
        if not replacements.empty:
            best_replacement = replacements.sort_values('predicted_points', ascending=False).iloc[0]
            team_df = team_df[team_df['id'] != player_to_replace['id']]
            team_df = pd.concat([team_df, pd.DataFrame([best_replacement])], ignore_index=True)
            total_cost = team_df['price'].sum()
        else:
            # No cheaper replacement found, so we have to remove the player
            team_df = team_df[team_df['id'] != player_to_replace['id']]
            total_cost = team_df['price'].sum()
            if len(team_df) < 11:
                return None, 0, 0

    return team_df, total_cost, team_df['predicted_points'].sum()

if __name__ == "__main__":
    main()
