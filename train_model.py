import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import joblib

def train_nba_model(input_file='nba_games_processed.csv'):
    print("🧠 Initializing Ensemble Training (Stable Version)...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}.")
        return

    # THE STABLE FEATURE LIST
    features = [
        # Physical
        'REST_DAYS', 'IS_HOME', 'IS_B2B', 'TRAVEL_MILES',
        # Elo
        'ELO', 'ELO_OPP_LOOKUP', # Note: Process data creates _OPP_LOOKUP now
        # Efficiency
        'AVG_10_OFF_RTG', 'AVG_10_PACE', 'AVG_10_FG_PCT',
        'AVG_10_OFF_RTG_OPP_LOOKUP', 'AVG_10_PACE_OPP_LOOKUP'
    ]
    target = 'PTS'
    
    # Clean Data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    # Train/Test Split
    split_index = int(len(df) * 0.85)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"📚 Training on {len(train_df)} games.")
    
    # Define Models
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    boost_model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    
    ensemble = VotingRegressor(estimators=[('linear', ridge_model), ('boost', boost_model)])
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    predictions = ensemble.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print("------------------------------------------------")
    print(f"🎯 Reverted Model MAE: {mae:.2f}")
    print("------------------------------------------------")
    
    joblib.dump(ensemble, 'nba_model.pkl')

if __name__ == "__main__":
    train_nba_model()