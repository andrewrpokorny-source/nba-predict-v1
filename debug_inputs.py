import pandas as pd
import joblib
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime

def inspect_inputs():
    print("🕵️‍♂️ Inspecting Model Inputs for Tonight...")
    
    try:
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except:
        print("❌ Data missing.")
        return

    # 1. Get Tonight's Game IDs
    today = datetime.now().strftime('%Y-%m-%d')
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games = board.game_header.get_data_frame()
    
    # 2. Find Lakers vs Spurs (or just print all)
    print(f"\nStats Entering Tonight ({today}):")
    print(f"{'TEAM':<20} | {'PACE (Last 10)':<15} | {'PACE (Season)':<15} | {'DEF PACE (Opp Allows)'}")
    print("-" * 80)

    for _, game in games.iterrows():
        for team_id in [game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']]:
            # Get latest stats
            team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
            if team_games.empty: continue
            
            last = team_games.iloc[-1]
            
            name = last['TEAM_NAME']
            pace_10 = last['AVG_10_PACE']
            pace_all = last['AVG_ALL_PACE']
            # We don't track "Defensive Pace" explicitly, but we can infer it 
            # from the opponent columns in the processed file if we look carefully.
            # For now, let's just look at their own pace.
            
            print(f"{name:<20} | {pace_10:<15.1f} | {pace_all:<15.1f} | --")

if __name__ == "__main__":
    inspect_inputs()