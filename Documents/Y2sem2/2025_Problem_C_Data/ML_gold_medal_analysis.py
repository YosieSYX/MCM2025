import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset (replace with your actual CSV file path)
data = pd.read_csv("summerOly_medal_counts.csv")

# Filter data for a specific country (e.g., 'China') and years before 2020
country_data = data[(data['NOC'] == 'China') & (data['Year'] < 2020)]

# Select relevant features (Year and Gold Medals)
X = country_data[['Year']]  # Using Year as the feature
y = country_data['Gold']    # Target variable: Number of Gold medals

# Add constant to the features for intercept in the model
X = sm.add_constant(X)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Negative Binomial model
model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(), link=sm.families.links.log())
result = model.fit()

# Print model summary
print(result.summary())

# Predict the values for both training and testing data
y_pred_train = result.predict(X_train)
y_pred_test = result.predict(X_test)

# Evaluate the model (Mean Squared Error)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Negative Binomial MSE (Training): {mse_train}")
print(f"Negative Binomial MSE (Testing): {mse_test}")

# Plot the original data vs. model predictions
plt.scatter(X['Year'], y, label="Original Data", color='blue')
plt.plot(X_train['Year'], y_pred_train, label="Negative Binomial Prediction (Training)", color='red')

plt.xlabel("Year")
plt.ylabel("Gold Medals")
plt.legend()
plt.title("Negative Binomial Regression Predictions (Before 2020)")
plt.show()

# Predict for 2024 using the trained model
future_years = pd.DataFrame({'Year': [2024]})
future_years = sm.add_constant(future_years)
future_predictions = result.predict(future_years)

print(f"Predicted Gold Medals for 2024: {future_predictions[0]}")
