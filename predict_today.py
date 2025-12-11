import pandas as pd
import joblib
import numpy as np
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from nba_locations import TEAM_COORDINATES
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 1. MATH HELPER
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# 2. GET SCHEDULE
def get_todays_games():
    print("📅 Fetching today's schedule...")
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today)
        games = board.game_header.get_data_frame()
        return games.to_dict('records')
    except:
        return []

# 3. GET TEAM STATS
def get_team_stats(team_id, df):
    team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    if team_games.empty: return None
    last = team_games.iloc[-1]
    return {
        'ELO': last['ELO'],
        'AVG_10_OFF_RTG': last['AVG_10_OFF_RTG'], 'AVG_10_PACE': last['AVG_10_PACE'],
        'AVG_ALL_OFF_RTG': last['AVG_ALL_OFF_RTG'], 'AVG_ALL_PACE': last['AVG_ALL_PACE'],
        'LAST_LAT': last['LAT'], 'LAST_LON': last['LON'],
        'LAST_GAME_DATE': pd.to_datetime(last['GAME_DATE']),
        'TEAM_NAME': last['TEAM_NAME']
    }

# 4. MAIN PREDICTOR
def predict_games():
    print("🧠 Loading Final Hybrid Model...")
    try:
        models = joblib.load('nba_model_v2.pkl')
        pace_model = models['pace']
        eff_model = models['eff']
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except FileNotFoundError:
        print("❌ Error: Missing files.")
        return

    games = get_todays_games()
    if not games: return

    print(f"\n🏀 Found {len(games)} games today.")
    print(f"{'MATCHUP':<30} | {'PRED SCORE':<15} | {'PACE'} | {'TOTAL'}")
    print("-" * 75)

    today = pd.to_datetime(datetime.now().date())
    
    # Must match train_model.py exactly
    feature_cols = [
        'REST_DAYS', 'IS_HOME', 'ELO', 'ELO_OPP_LOOKUP',
        'AVG_10_OFF_RTG', 'AVG_10_PACE', 'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP',
        'AVG_ALL_OFF_RTG', 'AVG_ALL_PACE', 'AVG_ALL_OFF_RTG_OPP_LOOKUP', 'AVG_ALL_PACE_OPP_LOOKUP',
        'ELO_DIFF', 'OFF_RTG_DIFF', 'PACE_DIFF'
    ]

    for g in games:
        h_id, v_id = g['HOME_TEAM_ID'], g['VISITOR_TEAM_ID']
        h_stats = get_team_stats(h_id, df)
        v_stats = get_team_stats(v_id, df)
        
        if not h_stats or not v_stats: continue
            
        h_rest = (today - h_stats['LAST_GAME_DATE']).days
        v_rest = (today - v_stats['LAST_GAME_DATE']).days
        h_coords = TEAM_COORDINATES.get(h_id, {'lat':0, 'lon':0})
        h_trav = haversine_distance(h_stats['LAST_LAT'], h_stats['LAST_LON'], h_coords['lat'], h_coords['lon'])
        v_trav = haversine_distance(v_stats['LAST_LAT'], v_stats['LAST_LON'], h_coords['lat'], h_coords['lon'])

        # CALCULATE DIFFS
        # Home Perspective
        h_elo_diff = h_stats['ELO'] - v_stats['ELO']
        h_off_diff = h_stats['AVG_10_OFF_RTG'] - v_stats['AVG_10_OFF_RTG'] # My Offense vs Their Offense (proxy)
        h_pace_diff = h_stats['AVG_10_PACE'] - v_stats['AVG_10_PACE']

        # Visitor Perspective
        v_elo_diff = v_stats['ELO'] - h_stats['ELO']
        v_off_diff = v_stats['AVG_10_OFF_RTG'] - h_stats['AVG_10_OFF_RTG']
        v_pace_diff = v_stats['AVG_10_PACE'] - h_stats['AVG_10_PACE']

        h_row = [
            min(h_rest, 7), 1, h_stats['ELO'], v_stats['ELO'],
            h_stats['AVG_10_OFF_RTG'], h_stats['AVG_10_PACE'], v_stats['AVG_10_OFF_RTG'], v_stats['AVG_10_PACE'],
            h_stats['AVG_ALL_OFF_RTG'], h_stats['AVG_ALL_PACE'], v_stats['AVG_ALL_OFF_RTG'], v_stats['AVG_ALL_PACE'],
            h_elo_diff, h_off_diff, h_pace_diff
        ]
        
        v_row = [
            min(v_rest, 7), 0, v_stats['ELO'], h_stats['ELO'],
            v_stats['AVG_10_OFF_RTG'], v_stats['AVG_10_PACE'], h_stats['AVG_10_OFF_RTG'], h_stats['AVG_10_PACE'],
            v_stats['AVG_ALL_OFF_RTG'], v_stats['AVG_ALL_PACE'], h_stats['AVG_ALL_OFF_RTG'], h_stats['AVG_ALL_PACE'],
            v_elo_diff, v_off_diff, v_pace_diff
        ]

        # Predict
        p_pace = (pace_model.predict(pd.DataFrame([h_row], columns=feature_cols))[0] + 
                  pace_model.predict(pd.DataFrame([v_row], columns=feature_cols))[0]) / 2
        
        h_eff = eff_model.predict(pd.DataFrame([h_row], columns=feature_cols))[0]
        v_eff = eff_model.predict(pd.DataFrame([v_row], columns=feature_cols))[0]
        
        h_score = (p_pace / 100) * h_eff
        v_score = (p_pace / 100) * v_eff
        
        total = h_score + v_score
        spread = abs(h_score - v_score)
        winner = h_stats['TEAM_NAME'] if h_score > v_score else v_stats['TEAM_NAME']
        
        matchup = f"{v_stats['TEAM_NAME']} @ {h_stats['TEAM_NAME']}"
        score_str = f"{int(v_score)} - {int(h_score)}"
        
        print(f"{matchup:<30} | {score_str:<15} | {int(p_pace)} | {int(total)} ({winner} +{spread:.1f})")

if __name__ == "__main__":
    predict_games()