import pandas as pd

def process_health():
    print("🏥 Calculating 'Roster Health' (Availability Score)...")
    
    try:
        # Load Player Logs
        df = pd.read_csv("nba_player_logs.csv")
        # Load Team Game Data (to attach the result to)
        games = pd.read_csv("nba_games_processed.csv")
    except:
        print("❌ Missing CSV files.")
        return

    # 1. Calculate Each Player's "Importance" (Avg Points per Season)
    # We assume a player's value is their season average
    print("   Evaluating player importance...")
    df['SEASON_ID'] = df['SEASON_ID'].astype(str)
    player_averages = df.groupby(['PLAYER_ID', 'SEASON_ID'])['PTS'].mean().reset_index()
    player_averages.rename(columns={'PTS': 'AVG_PTS'}, inplace=True)
    
    # Merge avg points back to the log
    df = pd.merge(df, player_averages, on=['PLAYER_ID', 'SEASON_ID'])
    
    # 2. Calculate "Full Strength" Output for each Team
    # Sum of ALL players' averages (Active or Inactive)
    # This is tricky. A better proxy: What is the sum of the "Top 8" players' averages?
    # Let's approximate: Sum of points of players *who played* in a specific game
    # vs Sum of points of the *Team's Top Scorers*.
    
    # Let's calculate the "Active Scoring Mass" for every game
    # Group by Game and Team, sum the 'AVG_PTS' of players who logged minutes > 0
    active_mass = df.groupby(['GAME_ID', 'TEAM_ID'])['AVG_PTS'].sum().reset_index()
    active_mass.rename(columns={'AVG_PTS': 'ACTIVE_SCORING_MASS'}, inplace=True)
    
    # 3. Calculate "Team Max Strength"
    # The maximum 'Active Mass' the team achieved in any game that season
    team_max = active_mass.groupby('TEAM_ID')['ACTIVE_SCORING_MASS'].quantile(0.95).reset_index()
    team_max.rename(columns={'ACTIVE_SCORING_MASS': 'MAX_SCORING_MASS'}, inplace=True)
    
    # 4. Calculate Availability %
    # (Active Mass / Max Mass)
    # If LeBron (25ppg) sits, the Active Mass drops by 25.
    active_mass = pd.merge(active_mass, team_max, on='TEAM_ID')
    active_mass['AVAILABILITY_PCT'] = active_mass['ACTIVE_SCORING_MASS'] / active_mass['MAX_SCORING_MASS']
    
    # Cap at 1.0 (sometimes a team plays better than 95th percentile)
    active_mass['AVAILABILITY_PCT'] = active_mass['AVAILABILITY_PCT'].clip(upper=1.0)
    
    # 5. Merge into Main Dataset
    print("   Merging health data into game file...")
    final_df = pd.merge(games, active_mass[['GAME_ID', 'TEAM_ID', 'AVAILABILITY_PCT']], on=['GAME_ID', 'TEAM_ID'], how='left')
    
    # Also get Opponent Health
    opp_health = active_mass[['GAME_ID', 'TEAM_ID', 'AVAILABILITY_PCT']].rename(columns={'AVAILABILITY_PCT': 'AVAILABILITY_PCT_OPP_LOOKUP'})
    final_df = pd.merge(final_df, opp_health, left_on=['GAME_ID', 'TEAM_ID_OPP_LOOKUP'], right_on=['GAME_ID', 'TEAM_ID'], how='left')
    
    # Fill missing (older games where we didn't download player data) with 1.0 (Assume healthy)
    final_df['AVAILABILITY_PCT'] = final_df['AVAILABILITY_PCT'].fillna(1.0)
    final_df['AVAILABILITY_PCT_OPP_LOOKUP'] = final_df['AVAILABILITY_PCT_OPP_LOOKUP'].fillna(1.0)
    
    # Clean duplicate columns from merge
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    final_df.to_csv("nba_games_processed.csv", index=False)
    print("✅ Success! Added 'AVAILABILITY_PCT' to dataset.")
    print("   (Ex: 0.85 means the team is missing 15% of its scoring power)")

if __name__ == "__main__":
    process_health()