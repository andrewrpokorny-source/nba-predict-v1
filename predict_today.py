import pandas as pd
import joblib
import numpy as np
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
from nba_locations import TEAM_COORDINATES  # Import the map we made

def haversine_distance(lat1, lon1, lat2, lon2):
    # Same math as before
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def get_todays_games():
    print("📅 Fetching today's schedule...")
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today)
        games = board.game_header.get_data_frame()
    except Exception as e:
        print(f"❌ Error connecting to NBA API: {e}")
        return []
    
    if games.empty:
        print("❌ No games scheduled for today.")
        return []
    
    return games.to_dict('records')

def get_team_stats_and_location(team_id, df):
    # Find the most recent game
    team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if team_games.empty:
        return None
    
    last_game = team_games.iloc[-1]
    
    # Get stats
    stats = {
        'AVG_10_PTS': last_game['AVG_10_PTS'],
        'AVG_10_FG_PCT': last_game['AVG_10_FG_PCT'],
        'AVG_10_PLUS_MINUS': last_game['AVG_10_PLUS_MINUS'],
        'AVG_10_AST': last_game['AVG_10_AST'],
        'AVG_10_REB': last_game['AVG_10_REB'],
        'LAST_GAME_DATE': pd.to_datetime(last_game['GAME_DATE']),
        # We need to know where they were last game to calc travel
        'LAST_LAT': last_game['LAT'],
        'LAST_LON': last_game['LON']
    }
    return stats

def predict_games():
    print("🧠 Loading model and data...")
    try:
        model = joblib.load('nba_model.pkl')
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except FileNotFoundError:
        print("❌ Error: Files not found. Run the pipeline first.")
        return

    games = get_todays_games()
    if not games:
        return

    print(f"\n🏀 Found {len(games)} games today. Predicting...\n")
    print(f"{'MATCHUP':<30} | {'PRED SCORE':<15} | {'CONFIDENCE'}")
    print("-" * 65)

    today_date = pd.to_datetime(datetime.now().date())

    for game in games:
        home_id = game['HOME_TEAM_ID']
        visitor_id = game['VISITOR_TEAM_ID']
        
        # 1. Get Base Stats
        home_stats = get_team_stats_and_location(home_id, df)
        visitor_stats = get_team_stats_and_location(visitor_id, df)
        
        if not home_stats or not visitor_stats:
            continue

        # 2. Calculate Rest & Fatigue
        home_rest = (today_date - home_stats['LAST_GAME_DATE']).days
        visitor_rest = (today_date - visitor_stats['LAST_GAME_DATE']).days
        
        # 3. Calculate Travel
        # Home Team Location (They are at home)
        home_coords = TEAM_COORDINATES.get(home_id)
        # Visitor Team Location (They are at Home Team's arena)
        # Note: Visitor travels TO the home arena.
        current_arena_coords = home_coords 
        
        # Home Travel: Dist from Last Game -> Home Arena
        home_travel = haversine_distance(
            home_stats['LAST_LAT'], home_stats['LAST_LON'],
            home_coords['lat'], home_coords['lon']
        )
        
        # Visitor Travel: Dist from Last Game -> Home Arena
        visitor_travel = haversine_distance(
            visitor_stats['LAST_LAT'], visitor_stats['LAST_LON'],
            current_arena_coords['lat'], current_arena_coords['lon']
        )

        # 4. Build Input Rows
        # Feature order MUST match train_model.py exactly
        features = [
            'REST_DAYS', 'IS_HOME', 'IS_B2B', 'TRAVEL_MILES',
            'AVG_10_PTS', 'AVG_10_FG_PCT', 'AVG_10_PLUS_MINUS',
            'AVG_10_AST', 'AVG_10_REB',
            'AVG_10_PTS_OPP', 'AVG_10_PLUS_MINUS_OPP',
            'IS_B2B_OPP', 'TRAVEL_MILES_OPP'
        ]

        # Home Row
        home_input = {
            'REST_DAYS': min(home_rest, 7),
            'IS_HOME': 1,
            'IS_B2B': 1 if home_rest == 1 else 0,
            'TRAVEL_MILES': home_travel,
            'AVG_10_PTS': home_stats['AVG_10_PTS'],
            'AVG_10_FG_PCT': home_stats['AVG_10_FG_PCT'],
            'AVG_10_PLUS_MINUS': home_stats['AVG_10_PLUS_MINUS'],
            'AVG_10_AST': home_stats['AVG_10_AST'],
            'AVG_10_REB': home_stats['AVG_10_REB'],
            'AVG_10_PTS_OPP': visitor_stats['AVG_10_PTS'],
            'AVG_10_PLUS_MINUS_OPP': visitor_stats['AVG_10_PLUS_MINUS'],
            'IS_B2B_OPP': 1 if visitor_rest == 1 else 0,
            'TRAVEL_MILES_OPP': visitor_travel
        }
        
        # Visitor Row
        visitor_input = {
            'REST_DAYS': min(visitor_rest, 7),
            'IS_HOME': 0,
            'IS_B2B': 1 if visitor_rest == 1 else 0,
            'TRAVEL_MILES': visitor_travel,
            'AVG_10_PTS': visitor_stats['AVG_10_PTS'],
            'AVG_10_FG_PCT': visitor_stats['AVG_10_FG_PCT'],
            'AVG_10_PLUS_MINUS': visitor_stats['AVG_10_PLUS_MINUS'],
            'AVG_10_AST': visitor_stats['AVG_10_AST'],
            'AVG_10_REB': visitor_stats['AVG_10_REB'],
            'AVG_10_PTS_OPP': home_stats['AVG_10_PTS'],
            'AVG_10_PLUS_MINUS_OPP': home_stats['AVG_10_PLUS_MINUS'],
            'IS_B2B_OPP': 1 if home_rest == 1 else 0,
            'TRAVEL_MILES_OPP': home_travel
        }

        # Predict
        home_pred = model.predict(pd.DataFrame([home_input])[features])[0]
        visitor_pred = model.predict(pd.DataFrame([visitor_input])[features])[0]

        # Display
        # Find names (using a quick ID lookup from the processed df)
        h_name = df[df['TEAM_ID'] == home_id]['TEAM_NAME'].iloc[0]
        v_name = df[df['TEAM_ID'] == visitor_id]['TEAM_NAME'].iloc[0]
        
        matchup = f"{v_name} @ {h_name}"
        score = f"{int(visitor_pred)} - {int(home_pred)}"
        
        # Simple "Confidence" metric (Spread)
        spread = abs(home_pred - visitor_pred)
        winner = h_name if home_pred > visitor_pred else v_name
        
        print(f"{matchup:<30} | {score:<15} | {winner} (+{spread:.1f})")

if __name__ == "__main__":
    predict_games()