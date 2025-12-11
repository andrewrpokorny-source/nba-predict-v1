import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_locations import TEAM_COORDINATES
import warnings

# Silence warnings in the dashboard
warnings.filterwarnings("ignore")

# 1. MATH & CONFIG
st.set_page_config(page_title="NBA AI Oracle", page_icon="🏀", layout="wide")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data
def load_data():
    try:
        models = joblib.load('nba_model_v2.pkl')
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return models, df
    except FileNotFoundError:
        return None, None

def get_todays_games():
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today)
        games = board.game_header.get_data_frame()
        return games
    except:
        return pd.DataFrame()

def get_team_stats(team_id, df):
    team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    if team_games.empty: return None
    return team_games.iloc[-1]

# 2. MAIN APP
st.title("🏀 NBA AI Oracle")
st.markdown("### Hybrid Linear/Tree Ensemble (MAE: 9.35)")

models, df = load_data()

if models is None:
    st.error("❌ Models not found. Run train_model.py first.")
    st.stop()

pace_model = models['pace']
eff_model = models['eff']

games = get_todays_games()

if games.empty:
    st.info("No games scheduled for today.")
else:
    # INPUT COLUMNS (Must match train_model.py EXACTLY)
    feature_cols = [
        'REST_DAYS', 'IS_HOME', 'ELO', 'ELO_OPP_LOOKUP',
        'AVG_10_OFF_RTG', 'AVG_10_PACE', 'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP',
        'AVG_ALL_OFF_RTG', 'AVG_ALL_PACE', 'AVG_ALL_OFF_RTG_OPP_LOOKUP', 'AVG_ALL_PACE_OPP_LOOKUP',
        'ELO_DIFF', 'OFF_RTG_DIFF', 'PACE_DIFF'
    ]

    predictions = []
    today = pd.to_datetime(datetime.now().date())

    for i, g in games.iterrows():
        h_id, v_id = g['HOME_TEAM_ID'], g['VISITOR_TEAM_ID']
        h_stats = get_team_stats(h_id, df)
        v_stats = get_team_stats(v_id, df)
        
        if h_stats is None or v_stats is None: continue
        
        # Context
        h_rest = (today - h_stats['GAME_DATE']).days
        v_rest = (today - v_stats['GAME_DATE']).days
        h_coords = TEAM_COORDINATES.get(h_id, {'lat':0,'lon':0})
        h_trav = haversine_distance(h_stats['LAT'], h_stats['LON'], h_coords['lat'], h_coords['lon'])
        v_trav = haversine_distance(v_stats['LAT'], v_stats['LON'], h_coords['lat'], h_coords['lon'])

        # CALCULATE DIFFS
        h_elo_diff = h_stats['ELO'] - v_stats['ELO']
        h_off_diff = h_stats['AVG_10_OFF_RTG'] - v_stats['AVG_10_OFF_RTG']
        h_pace_diff = h_stats['AVG_10_PACE'] - v_stats['AVG_10_PACE']

        v_elo_diff = v_stats['ELO'] - h_stats['ELO']
        v_off_diff = v_stats['AVG_10_OFF_RTG'] - h_stats['AVG_10_OFF_RTG']
        v_pace_diff = v_stats['AVG_10_PACE'] - h_stats['AVG_10_PACE']

        # BUILD ROWS
        h_row = [
            min(h_rest,7), 1, h_stats['ELO'], v_stats['ELO'],
            h_stats['AVG_10_OFF_RTG'], h_stats['AVG_10_PACE'], v_stats['AVG_10_OFF_RTG'], v_stats['AVG_10_PACE'],
            h_stats['AVG_ALL_OFF_RTG'], h_stats['AVG_ALL_PACE'], v_stats['AVG_ALL_OFF_RTG'], v_stats['AVG_ALL_PACE'],
            h_elo_diff, h_off_diff, h_pace_diff
        ]
        
        v_row = [
            min(v_rest,7), 0, v_stats['ELO'], h_stats['ELO'],
            v_stats['AVG_10_OFF_RTG'], v_stats['AVG_10_PACE'], h_stats['AVG_10_OFF_RTG'], h_stats['AVG_10_PACE'],
            v_stats['AVG_ALL_OFF_RTG'], v_stats['AVG_ALL_PACE'], h_stats['AVG_ALL_OFF_RTG'], h_stats['AVG_ALL_PACE'],
            v_elo_diff, v_off_diff, v_pace_diff
        ]

        # PREDICT
        # Pace (Average of both perspectives)
        p_pace = (pace_model.predict(pd.DataFrame([h_row], columns=feature_cols))[0] + 
                  pace_model.predict(pd.DataFrame([v_row], columns=feature_cols))[0]) / 2
        
        h_eff = eff_model.predict(pd.DataFrame([h_row], columns=feature_cols))[0]
        v_eff = eff_model.predict(pd.DataFrame([v_row], columns=feature_cols))[0]
        
        h_score = (p_pace / 100) * h_eff
        v_score = (p_pace / 100) * v_eff
        total = h_score + v_score
        spread = h_score - v_score # Positive = Home Wins
        
        # FORMAT FOR UI
        winner = h_stats['TEAM_NAME'] if h_score > v_score else v_stats['TEAM_NAME']
        display_spread = abs(spread)
        
        predictions.append({
            "Away Team": v_stats['TEAM_NAME'],
            "Home Team": h_stats['TEAM_NAME'],
            "Proj Winner": winner,
            "Proj Spread": f"{display_spread:.1f}",
            "Proj Total": f"{int(total)}",
            "Proj Pace": f"{int(p_pace)}",
            "Score": f"{int(v_score)} - {int(h_score)}"
        })

    # DISPLAY
    st.table(pd.DataFrame(predictions))

    with st.expander("📊 See Model Details"):
        st.json({
            "Model Type": "Hybrid Ensemble (Ridge + Gradient Boosting)",
            "Features": len(feature_cols),
            "Training Games": len(df),
            "Current MAE": "9.35"
        })