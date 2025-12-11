import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import joblib

def train_nba_model(input_file='nba_games_processed.csv'):
    print("🧠 Training Final Hybrid Model (with Mismatch Logic)...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}.")
        return

    # 1. DEFINE FEATURES (The Full Arsenal)
    features = [
        # Physical Context
        'REST_DAYS', 'IS_HOME', 
        
        # Skill / Elo
        'ELO', 'ELO_OPP_LOOKUP',
        
        # Form (Last 10 Games - Recency Bias)
        'AVG_10_OFF_RTG', 'AVG_10_PACE', 
        'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP',
        
        # Class (Season Long - Stability)
        'AVG_ALL_OFF_RTG', 'AVG_ALL_PACE', 
        'AVG_ALL_OFF_RTG_OPP_LOOKUP', 'AVG_ALL_PACE_OPP_LOOKUP',
        
        # Mismatches (Interaction Features - NEW!)
        'ELO_DIFF',       # Am I historically better?
        'OFF_RTG_DIFF',   # Is my offense better than yours?
        'PACE_DIFF'       # Do I play faster than you?
    ]
    
    # 2. Clean Data
    # Drop rows where any feature is missing (NaN) or Infinite
    cols_to_check = features + ['PTS', 'PACE', 'OFF_RTG']
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check)

    # 3. Train/Test Split (85% Train, 15% Test)
    # We split chronologically (past predicts future)
    split_index = int(len(df) * 0.85)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    print(f"📚 Training on {len(train_df)} games.")

    # ---------------------------------------------------------
    # MODEL 1: THE PACE PREDICTOR (LINEAR / RIDGE)
    # Strategy: Linear models handle "additive" speed better.
    # If Team A (+3 pace) plays Team B (+3 pace), Linear correctly predicts +6.
    # ---------------------------------------------------------
    print("   🏃 Training Pace Model (Ridge/Linear)...")
    y_train_pace = train_df['PACE']
    
    pace_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    pace_model.fit(X_train, y_train_pace)
    
    # ---------------------------------------------------------
    # MODEL 2: THE EFFICIENCY PREDICTOR (BOOSTING / TREES)
    # Strategy: Efficiency is non-linear (rock-paper-scissors matchups).
    # Gradient Boosting finds complex patterns in Offense vs Defense.
    # ---------------------------------------------------------
    print("   🎯 Training Efficiency Model (Gradient Boosting)...")
    y_train_eff = train_df['OFF_RTG']
    
    eff_model = HistGradientBoostingRegressor(
        learning_rate=0.05, 
        max_depth=4, 
        max_iter=150, 
        min_samples_leaf=20, 
        l2_regularization=0.1, 
        random_state=42
    )
    eff_model.fit(X_train, y_train_eff)

    # ---------------------------------------------------------
    # EVALUATION
    # Formula: Predicted Score = (Predicted Pace / 100) * Predicted Efficiency
    # ---------------------------------------------------------
    pred_pace = pace_model.predict(X_test)
    pred_eff = eff_model.predict(X_test)
    pred_pts = (pred_pace / 100) * pred_eff
    
    actual_pts = test_df['PTS']
    mae = mean_absolute_error(actual_pts, pred_pts)
    
    print("------------------------------------------------")
    print(f"🏆 FINAL MODEL MAE: {mae:.2f}")
    print("------------------------------------------------")
    
    # Save the dictionary of models
    joblib.dump({'pace': pace_model, 'eff': eff_model}, 'nba_model_v2.pkl')
    print("💾 Saved to nba_model_v2.pkl")

if __name__ == "__main__":
    train_nba_model()