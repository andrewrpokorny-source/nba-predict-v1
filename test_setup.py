import pandas as pd
from nba_api.stats.static import teams

# Get list of teams
nba_teams = teams.get_teams()
# Select the first team (Atlanta Hawks)
hawks = nba_teams[0]

print("------------------------------------------------")
print("SUCCESS! Python is speaking to the NBA.")
print(f"First Team Found: {hawks['full_name']}")
print("------------------------------------------------")