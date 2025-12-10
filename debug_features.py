import pandas as pd

def inspect_travel():
    print("🕵️‍♂️ Inspecting Travel & Features...")
    
    # Load the processed data
    try:
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except FileNotFoundError:
        print("❌ Data not found.")
        return

    # Filter for Golden State Warriors (ID: 1610612744)
    gsw = df[df['TEAM_ID'] == 1610612744].sort_values('GAME_DATE')
    
    # Grab a random slice of 10 games
    # We look at columns that should change based on location
    columns_to_check = [
        'GAME_DATE', 'MATCHUP', 'IS_HOME', 
        'LAT', 'LON', 'TRAVEL_MILES', 'REST_DAYS', 'IS_B2B'
    ]
    
    print("\n--- TRAVEL AUDIT: GOLDEN STATE WARRIORS ---")
    print(gsw[columns_to_check].tail(10).to_string(index=False))
    
    # Check for "Impossible" Zeros
    # If a team is AWAY, Travel Miles should usually be > 0 (unless playing LAC vs LAL)
    errors = df[(df['IS_HOME'] == 0) & (df['TRAVEL_MILES'] == 0) & (~df['MATCHUP'].str.contains('LAL|LAC|BKN|NYK'))]
    
    if len(errors) > 0:
        print(f"\n⚠️ WARNING: Found {len(errors)} games where an Away team traveled 0 miles.")
        print(errors[['GAME_DATE', 'MATCHUP', 'TRAVEL_MILES']].head())
    else:
        print("\n✅ Logic Check Passed: Away teams are traveling.")

if __name__ == "__main__":
    inspect_travel()