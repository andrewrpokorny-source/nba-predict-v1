import pandas as pd
import numpy as np
from nba_locations import TEAM_COORDINATES

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_elo_probability(rating1, rating2):
    return 1.0 / (1 + 10 ** ((rating2 - rating1) / 400.0))

def update_elo(winner_elo, loser_elo, margin_of_victory):
    # K-Factor logic
    k = 20 * ((margin_of_victory + 3) ** 0.8) / (7.5 + 0.006 * (winner_elo - loser_elo))
    expected_win = get_elo_probability(winner_elo, loser_elo)
    change = k * (1 - expected_win)
    return winner_elo + change, loser_elo - change

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ---------------------------------------------------------
# MAIN PROCESSOR (STABLE VERSION)
# ---------------------------------------------------------

def process_nba_data(input_file='nba_games_2025-26.csv'):
    print(f"⚙️ Processing {input_file} (Reverting to Stable Elo/Travel)...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("❌ Error: CSV file not found.")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')

    # 1. MERGE OPPONENT STATS
    # We grab basic stats: Points, Possessions triggers
    game_opponents = df[['GAME_ID', 'TEAM_ID', 'PTS']]
    df_merged = pd.merge(df, game_opponents, on='GAME_ID', suffixes=('', '_OPP'))
    df_merged = df_merged[df_merged['TEAM_ID'] != df_merged['TEAM_ID_OPP']]
    df_merged = df_merged.sort_values(by='GAME_DATE')

    # 2. CALCULATE ELO
    elo_dict = {team: 1500 for team in df_merged['TEAM_ID'].unique()}
    elo_list = []
    
    for idx, row in df_merged.iterrows():
        t = row['TEAM_ID']
        opp = row['TEAM_ID_OPP']
        
        # Record Elo BEFORE the game
        elo_list.append(elo_dict[t])
        
        # Update Elo AFTER the game
        margin = row['PTS'] - row['PTS_OPP']
        if margin > 0:
            new_elo, _ = update_elo(elo_dict[t], elo_dict[opp], margin)
            elo_dict[t] = new_elo
        else:
            _, new_elo = update_elo(elo_dict[opp], elo_dict[t], abs(margin))
            elo_dict[t] = new_elo

    df_merged['ELO'] = elo_list

    # 3. ADVANCED METRICS (Basic OffRtg & Pace)
    # We approximate Possessions for stability
    df_merged['POSS'] = df_merged['FGA'] + (0.44 * df_merged['FTA']) - df_merged['OREB'] + df_merged['TOV']
    df_merged['OFF_RTG'] = (df_merged['PTS'] / df_merged['POSS']) * 100
    df_merged['PACE'] = df_merged['POSS']

    # 4. LOCATION & TRAVEL
    def get_lat_lon(row):
        is_home = 0 if '@' in row['MATCHUP'] else 1
        arena_id = row['TEAM_ID'] if is_home else row['TEAM_ID_OPP']
        coords = TEAM_COORDINATES.get(arena_id, {'lat': 0, 'lon': 0})
        return pd.Series([coords['lat'], coords['lon']])

    df_merged[['LAT', 'LON']] = df_merged.apply(get_lat_lon, axis=1)
    df_merged['PREV_LAT'] = df_merged.groupby('TEAM_ID')['LAT'].shift(1)
    df_merged['PREV_LON'] = df_merged.groupby('TEAM_ID')['LON'].shift(1)
    df_merged['TRAVEL_MILES'] = haversine_distance(
        df_merged['PREV_LAT'], df_merged['PREV_LON'], 
        df_merged['LAT'], df_merged['LON']
    ).fillna(0)

    # 5. ROLLING STATS
    df_merged['LAST_GAME_DATE'] = df_merged.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df_merged['REST_DAYS'] = (df_merged['GAME_DATE'] - df_merged['LAST_GAME_DATE']).dt.days.fillna(7).clip(upper=7)
    df_merged['IS_B2B'] = (df_merged['REST_DAYS'] == 1).astype(int)
    df_merged['IS_HOME'] = df_merged['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

    features_to_roll = ['OFF_RTG', 'PACE', 'FG_PCT', 'PLUS_MINUS']
    
    for col in features_to_roll:
        df_merged[f'AVG_10_{col}'] = df_merged.groupby('TEAM_ID')[col].transform(
            lambda x: x.shift(1).rolling(window=10).mean()
        )

    # 6. FINAL CONTEXT MERGE
    # We grab the Opponent's Elo and Rolling Stats
    cols_to_share = ['GAME_ID', 'TEAM_ID', 'ELO', 'IS_B2B', 'TRAVEL_MILES']
    for col in features_to_roll:
        cols_to_share.append(f'AVG_10_{col}')
        
    opp_stats = df_merged[cols_to_share]
    
    final_df = pd.merge(
        df_merged, 
        opp_stats, 
        left_on=['GAME_ID', 'TEAM_ID_OPP'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        suffixes=('', '_OPP_LOOKUP')
    )
    
    # Save
    final_df = final_df.dropna(subset=['AVG_10_OFF_RTG', 'AVG_10_OFF_RTG_OPP_LOOKUP'])
    final_df.to_csv("nba_games_processed.csv", index=False)
    print("✅ Successfully Reverted to Stable Version.")
    return final_df

if __name__ == "__main__":
    process_nba_data()