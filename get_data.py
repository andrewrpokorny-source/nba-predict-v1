import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time

def fetch_nba_data():
    # We will fetch the last 3 seasons (including the current one)
    seasons = [
        '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
        '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
    ]
    all_games = []

    print(f"🏀 Starting download for seasons: {seasons}")

    for season in seasons:
        print(f"   Downloading {season}...")
        try:
            # Fetch Regular Season
            log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
            df = log.get_data_frames()[0]
            
            # Add a column to track which season this is
            df['SEASON_ID'] = season
            all_games.append(df)
            
            # Be polite to the NBA servers (pause for 1 second)
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error downloading {season}: {e}")

    # Combine all seasons into one big table
    if all_games:
        final_df = pd.concat(all_games, ignore_index=True)
        
        # Save to the SAME filename we used before (so process_data.py finds it)
        filename = "nba_games_2025-26.csv" 
        final_df.to_csv(filename, index=False)
        
        print("------------------------------------------------")
        print(f"✅ Success! Total games downloaded: {len(final_df)}")
        print(f"💾 Overwrote: {filename}")
        print("------------------------------------------------")
    else:
        print("❌ No data found.")

if __name__ == "__main__":
    fetch_nba_data()