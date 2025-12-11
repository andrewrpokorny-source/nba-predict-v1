import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

def get_player_logs():
    print("🏀 Fetching PLAYER Data (This extracts every player's performance)...")
    
    # We focus on recent history for the Player Model to save time
    seasons = ['2023-24', '2024-25', '2025-26']
    all_players = []

    for season in seasons:
        print(f"   Downloading Player Logs for {season}...")
        try:
            # 'P' = Player scope
            log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P').get_data_frames()[0]
            all_players.append(log)
            time.sleep(1) # Be polite to API
        except Exception as e:
            print(f"❌ Error: {e}")

    if all_players:
        df = pd.concat(all_players)
        df.to_csv("nba_player_logs.csv", index=False)
        print(f"✅ Success! Saved {len(df)} player performances to 'nba_player_logs.csv'")
    else:
        print("❌ Failed to get data.")

if __name__ == "__main__":
    get_player_logs()