import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from nba_locations import TEAM_COORDINATES

# -------------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# -------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data # Caches data so it doesn't reload on every click
def load_data():
    try:
        model = joblib.load('nba_model.pkl')
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return model, df
    except FileNotFoundError:
        return None, None

def get_todays_games():
    today = datetime.now().strftime('%Y-%m-%d')
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games = board.game_header.get_data_frame()
    return games

def get_team_stats(team_id, df):
    # Get last game stats for a team
    team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    if team_games.empty: return None
    
    last_game = team_games.iloc[-1]
    return {
        'AVG_10_PTS': last_game['AVG_10_PTS'],
        'AVG_10_FG_PCT': last_game['AVG_10_FG_PCT'],
        'AVG_10_PLUS_MINUS': last_game['AVG_10_PLUS_MINUS'],
        'AVG_10_AST': last_game['AVG_10_AST'],
        'AVG_10_REB': last_game['AVG_10_REB'],
        'LAST_GAME_DATE': last_game['GAME_DATE'],
        'LAST_LAT': last_game['LAT'],
        'LAST_LON': last_game['LON'],
        'TEAM_NAME': last_game['TEAM_NAME']
    }

# -------------------------------------------------------------------
# 2. THE APP LAYOUT
# -------------------------------------------------------------------

st.set_page_config(page_title="NBA AI Oracle", page_icon="🏀")

st.title("🏀 NBA AI Oracle")
st.markdown("### Advanced Analytics & Prediction Engine")

# Load Resources
model, df = load_data()

if model is None:
    st.error("❌ Model or Data not found. Please run the pipeline first!")
    st.stop()

# Sidebar: Model Stats
st.sidebar.header("🧠 Model Stats")
st.sidebar.info(f"Training Data: {len(df)} games")
st.sidebar.info("Model: Gradient Boosting (Scikit-Learn)")

# TAB 1: TODAY'S PREDICTIONS
games = get_todays_games()

if games.empty:
    st.warning("No games scheduled for today.")
else:
    st.subheader(f"📅 Predictions for {datetime.now().strftime('%Y-%m-%d')}")
    
    predictions = []
    
    today_date = pd.to_datetime(datetime.now().date())
    
    # Progress bar for calculation
    progress_bar = st.progress(0)
    
    for i, game in games.iterrows():
        home_id = game['HOME_TEAM_ID']
        visitor_id = game['VISITOR_TEAM_ID']
        
        h_stats = get_team_stats(home_id, df)
        v_stats = get_team_stats(visitor_id, df)
        
        if not h_stats or not v_stats:
            continue
            
        # Calculate Context (Rest/Travel)
        h_rest = (today_date - h_stats['LAST_GAME_DATE']).days
        v_rest = (today_date - v_stats['LAST_GAME_DATE']).days
        
        h_coords = TEAM_COORDINATES.get(home_id, {'lat':0, 'lon':0})
        # Visitor travels TO home arena
        h_travel = haversine_distance(h_stats['LAST_LAT'], h_stats['LAST_LON'], h_coords['lat'], h_coords['lon'])
        v_travel = haversine_distance(v_stats['LAST_LAT'], v_stats['LAST_LON'], h_coords['lat'], h_coords['lon'])
        
        # Build Input (Must match training columns perfectly)
        features = [
            'REST_DAYS', 'IS_HOME', 'IS_B2B', 'TRAVEL_MILES',
            'AVG_10_PTS', 'AVG_10_FG_PCT', 'AVG_10_PLUS_MINUS',
            'AVG_10_AST', 'AVG_10_REB',
            'AVG_10_PTS_OPP', 'AVG_10_PLUS_MINUS_OPP',
            'IS_B2B_OPP', 'TRAVEL_MILES_OPP'
        ]
        
        # Predict Home Score
        h_input = [
            min(h_rest, 7), 1, 1 if h_rest==1 else 0, h_travel,
            h_stats['AVG_10_PTS'], h_stats['AVG_10_FG_PCT'], h_stats['AVG_10_PLUS_MINUS'], h_stats['AVG_10_AST'], h_stats['AVG_10_REB'],
            v_stats['AVG_10_PTS'], v_stats['AVG_10_PLUS_MINUS'], 1 if v_rest==1 else 0, v_travel
        ]
        
        # Predict Visitor Score
        v_input = [
            min(v_rest, 7), 0, 1 if v_rest==1 else 0, v_travel,
            v_stats['AVG_10_PTS'], v_stats['AVG_10_FG_PCT'], v_stats['AVG_10_PLUS_MINUS'], v_stats['AVG_10_AST'], v_stats['AVG_10_REB'],
            h_stats['AVG_10_PTS'], h_stats['AVG_10_PLUS_MINUS'], 1 if h_rest==1 else 0, h_travel
        ]
        
        h_pred = model.predict([h_input])[0]
        v_pred = model.predict([v_input])[0]
        
        spread = abs(h_pred - v_pred)
        winner = h_stats['TEAM_NAME'] if h_pred > v_pred else v_stats['TEAM_NAME']
        confidence = "🔥 High" if spread > 5 else "⚠️ Low"
        
        predictions.append({
            "Matchup": f"{v_stats['TEAM_NAME']} @ {h_stats['TEAM_NAME']}",
            "Predicted Score": f"{int(v_pred)} - {int(h_pred)}",
            "Winner": winner,
            "Spread": f"{spread:.1f}",
            "Confidence": confidence,
            "Visitor Travel": f"{int(v_travel)} mi",
            "Home Rest": f"{h_rest} days"
        })
        
        progress_bar.progress((i + 1) / len(games))

    # Display Dataframe
    st.dataframe(pd.DataFrame(predictions))

# TAB 2: DATA INSPECTION
with st.expander("🔎 Inspect Underlying Data"):
    st.write("This is the raw data your model uses to learn.")
    st.dataframe(df.tail(10))