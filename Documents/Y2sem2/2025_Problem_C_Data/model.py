import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
medal_data = pd.read_csv("summerOly_medal_counts.csv")
hosts_data = pd.read_csv("summerOly_hosts.csv")
sports_data = pd.read_csv("summerOly_programs.csv")

# Prepare hosting status
hosts_data['Hosting'] = 1
medal_data = medal_data.merge(hosts_data[['Year', 'Country', 'Hosting']], on=['Year', 'Country'], how='left')
medal_data['Hosting'] = medal_data['Hosting'].fillna(0)

# Aggregate sport-specific medals
# Assuming sports_data has 'Country', 'Year', 'Sport', and 'Medals' columns
sports_medals = sports_data.groupby(['Country', 'Year', 'Sport'])['Medals'].sum().unstack(fill_value=0)
sports_medals = sports_medals.reset_index()

# Merge sport-specific medals with medal data
medal_data = medal_data.merge(sports_medals, on=['Country', 'Year'], how='left')
medal_data = medal_data.fillna(0)  # Fill missing values for sports with 0 medals

# Add historical performance (lag features)
medal_data = medal_data.sort_values(by=['Country', 'Year'])
for lag in range(1, 6):  # Include the last 5 Olympic games
    medal_data[f'Lag_{lag}_Medals'] = medal_data.groupby('Country')['Gold'].shift(lag)

# Drop rows with missing historical data
medal_data = medal_data.dropna()

# Features and target variable
features = medal_data.drop(columns=['Gold', 'Silver', 'Bronze', 'Total', 'Country'])
target = medal_data['Gold']  # Predicting Gold medals (can extend to Total)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Predict for 2024
future_year = 2024
future_data = medal_data[medal_data['Year'] == 2020].copy()  # Use 2020 as the base
future_data['Year'] = future_year
future_data['Hosting'] = (future_data['Country'] == 'France').astype(int)  # Paris hosting
future_predictions = model.predict(future_data.drop(columns=['Gold', 'Silver', 'Bronze', 'Total', 'Country']))

# Combine predictions with countries
future_data['Predicted_Gold'] = future_predictions
print(future_data[['Country', 'Predicted_Gold']].sort_values(by='Predicted_Gold', ascending=False))
