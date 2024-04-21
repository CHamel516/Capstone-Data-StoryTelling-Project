import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define paths to the datasets
base_path = r"/Users/jeremyhamel/Documents/GitHub/Capstone-Data-StoryTelling-Project/DataSet/"
team_results_path = f"{base_path}TeamResults.csv"
power_ratings_path = f"{base_path}538 Ratings.csv"
seed_results_path = f"{base_path}Seed Results.csv"

# Load datasets
team_results = pd.read_csv(team_results_path)
power_ratings = pd.read_csv(power_ratings_path)
seed_results = pd.read_csv(seed_results_path)

# Merging datasets
combined_data = team_results.merge(power_ratings, on='TEAM').merge(seed_results, on='SEED')

# Check the column names in combined_data
print(combined_data.columns)

# Data cleaning
combined_data.dropna(inplace=True)
combined_data['SEED'] = pd.Categorical(combined_data['SEED']).codes

# Feature selection
# Adjust feature selection based on the actual column names in combined_data with suffixes (_x and _y)
X = combined_data[['POWER RATING', 'SEED', 'PAKE_x', 'PAKE RANK_x', 'PASE_x', 'PASE RANK_x', 'GAMES_x', 'W_x', 'L_x', 'WIN%_x', 'R64_x', 'R32_x', 'S16_x', 'E8_x', 'F4_x', 'F2_x', 'CHAMP_x', 'TOP2_x', 'F4%', 'CHAMP%_x']]

# Identify the correct target variable column name
# Example: If the target variable indicates whether a team won or lost, replace 'WINNER_COLUMN_NAME' with the actual column name
target_variable_name = 'CHAMP_x'  # Replace 'WINNER' with the actual column name representing the target variable

y = combined_data[target_variable_name]

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
