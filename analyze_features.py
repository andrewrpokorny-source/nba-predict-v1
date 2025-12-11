import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
import numpy as np

def analyze_model():
    print("🕵️‍♂️ Auditing Model Brain...")
    
    # 1. Load Resources
    try:
        model = joblib.load('nba_model.pkl')
        df = pd.read_csv('nba_games_processed.csv')
    except FileNotFoundError:
        print("❌ Error: Model or Data not found. Run train_model.py first!")
        return

    # 2. Prepare Data (Must match train_model.py list exactly)
    features = [
        'REST_DAYS', 'IS_HOME', 'IS_B2B', 'TRAVEL_MILES',
        'ELO', 'ELO_OPP_LOOKUP',
        'AVG_10_OFF_RTG', 'AVG_10_PACE', 'AVG_10_FG_PCT',
        'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP'
    ]
    target = 'PTS'
    
    # Clean Data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    # Test on the most recent 20% of data (The "Current Season" effectively)
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"🧪 Testing on {len(test_df)} recent games...")

    # 3. Run Permutation Importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42, 
        scoring='neg_mean_absolute_error'
    )
    
    # 4. Display Results
    sorted_idx = result.importances_mean.argsort()
    
    print("\n📊 FEATURE IMPORTANCE (Higher = More Valuable)")
    print("---------------------------------------")
    for idx in sorted_idx:
        print(f"{features[idx]:<30} | {result.importances_mean[idx]:.4f}")
    print("---------------------------------------")

if __name__ == "__main__":
    analyze_model()