import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def tune_brain():
    print("🔧 Initializing Hyperparameter Tuning (The Tournament)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('nba_games_processed.csv')
    except FileNotFoundError:
        print("❌ Data missing.")
        return

    # 2. Define Features (Must match your current best set)
    features = [
        'REST_DAYS', 'IS_HOME',
        'ELO', 'ELO_OPP_LOOKUP',
        'AVG_10_OFF_RTG', 'AVG_10_PACE',
        'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP',
        'AVG_ALL_OFF_RTG', 'AVG_ALL_PACE',
        'AVG_ALL_OFF_RTG_OPP_LOOKUP', 'AVG_ALL_PACE_OPP_LOOKUP'
    ]
    target = 'PTS'
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    # Use recent data for tuning (last 3 seasons) to ensure it fits modern NBA
    # Assuming roughly 1200 games per season
    tune_df = df.tail(4000)
    
    X = tune_df[features]
    y = tune_df[target]
    
    print(f"🎮 Tuning on {len(tune_df)} games...")

    # 3. Define the Competitors
    # We want to tune the Boosting Model specifically
    
    model = HistGradientBoostingRegressor(random_state=42)
    
    # The Grid: These are the settings we will test
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [100, 200, 500],
        'max_depth': [3, 5, 7],         # How complex the trees can be
        'l2_regularization': [0.0, 0.1, 1.0], # Prevents overfitting
        'min_samples_leaf': [20, 50]    # Requires x games to make a rule
    }
    
    # 4. Run the Tournament (Randomized Search)
    # n_iter=20 means we try 20 random combinations
    search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=20, 
        scoring='neg_mean_absolute_error', 
        cv=5, # 5-Fold Cross Validation (Robust testing)
        verbose=1,
        random_state=42,
        n_jobs=-1 # Use all computer cores
    )
    
    search.fit(X, y)
    
    # 5. The Winner
    print("\n🏆 BEST SETTINGS FOUND:")
    print(search.best_params_)
    print(f"Best Validation Score (MAE): {-search.best_score_:.4f}")
    
    return search.best_params_

if __name__ == "__main__":
    tune_brain()