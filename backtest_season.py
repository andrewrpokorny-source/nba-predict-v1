import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

def backtest_season():
    print("📉 Starting Season Simulation (Backtest)...")
    
    # 1. Load Resources
    try:
        model = joblib.load('nba_model.pkl')
        df = pd.read_csv('nba_games_processed.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except FileNotFoundError:
        print("❌ Data missing.")
        return

    # 2. Filter for THIS Season (2025-26)
    # We assume games after Oct 1, 2025 are this season
    current_season = df[df['GAME_DATE'] > '2025-10-01'].copy()
    
    if current_season.empty:
        print("⚠️ No games found for 2025-26 season to backtest.")
        return

    print(f"🎮 Simulating {len(current_season)} games...")

    # 3. Define Features (MUST MATCH train_model.py EXACTLY)
    features = [
        'REST_DAYS', 'IS_HOME',
        'ELO', 'ELO_OPP_LOOKUP',
        
        # FORM (Last 10)
        'AVG_10_OFF_RTG', 'AVG_10_PACE',
        'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP',
        
        # CLASS (All Season - NEW ADDITION)
        'AVG_ALL_OFF_RTG', 'AVG_ALL_PACE',
        'AVG_ALL_OFF_RTG_OPP_LOOKUP', 'AVG_ALL_PACE_OPP_LOOKUP'
    ]
    target = 'PTS'
    
    # 4. Clean Data
    # Drop rows where any of the required features are missing (NaN)
    current_season = current_season.dropna(subset=features)
    
    X_test = current_season[features]
    
    # 5. Run Predictions
    predictions = model.predict(X_test)
    
    # 6. Calculate Errors
    current_season['PREDICTED_PTS'] = predictions
    current_season['ERROR'] = abs(current_season['PTS'] - current_season['PREDICTED_PTS'])
    
    mae = current_season['ERROR'].mean()
    print(f"🏆 Season MAE: {mae:.2f}")

    # 7. Visualize
    current_season = current_season.sort_values('GAME_DATE')
    current_season['CUMULATIVE_MAE'] = current_season['ERROR'].expanding().mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(current_season['GAME_DATE'], current_season['CUMULATIVE_MAE'], label='Model Error (Lower is Better)', color='blue')
    plt.title(f'Model Accuracy Over 2025-26 Season (Final MAE: {mae:.2f})')
    plt.ylabel('Mean Absolute Error (Points)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('backtest_chart.png')
    print("📊 Chart saved to 'backtest_chart.png'.")

    # 8. Show Best/Worst
    print("\n✅ Best Predictions (Spot On):")
    print(current_season[['GAME_DATE', 'MATCHUP', 'PTS', 'PREDICTED_PTS', 'ERROR']].sort_values('ERROR').head(5))

if __name__ == "__main__":
    backtest_season()