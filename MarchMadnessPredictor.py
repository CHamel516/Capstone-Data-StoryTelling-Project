import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define paths to the datasets
base_path = r"D:\Code Projects\Capstone Data StoryTelling Project\DataSet"
team_results_path = f"{base_path}\\Team Results.csv"
power_ratings_path = f"{base_path}\\538 Ratings.csv"
seed_results_path = f"{base_path}\\Seed Results.csv"

# Load datasets
team_results = pd.read_csv(team_results_path)
power_ratings = pd.read_csv(power_ratings_path)
seed_results = pd.read_csv(seed_results_path)

# Check columns to ensure the key for merging exists
print(team_results.columns)
print(power_ratings.columns)
print(seed_results.columns)

# Merging datasets
combined_data = team_results.merge(power_ratings, on='TEAM').merge(seed_results, on='SEED')

# Data cleaning
combined_data.dropna(inplace=True)
combined_data['SEED'] = pd.Categorical(combined_data['SEED']).codes

# Feature selection
X = combined_data[['POWER RATING', 'SEED', 'PAKE', 'PAKE RANK', 'PASE', 'PASE RANK', 'GAMES', 'W', 'L', 'WIN%', 'R64', 'R32', 'S16', 'E8', 'F4', 'F2', 'CHAMP', 'TOP2', 'F4%', 'CHAMP%']]  # Replace 'OTHER_RELEVANT_METRICS' with actual metrics from your data
y = combined_data['WINNER']  # WINNER column should indicate if the team won the game

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Optional: Save model for future predictions
import joblib
joblib.dump(model, 'ncaa_model.pkl')

# Example of how you might predict new data for an upcoming tournament
# new_data = pd.read_csv(r"path_to_new_tournament_data.csv")
# new_data_processed = preprocess_new_data(new_data)  # Define preprocess_new_data to fit your dataset
# predictions = model.predict(new_data_processed)
# print(predictions)

# Note: You need to define preprocess_new_data or ensure the new_data has the same format and preprocessing as your training data