import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the results dataset (gold_medals_by_sport_and_country.csv)
results_df = pd.read_csv('gold_medals_by_sport_and_country.csv', encoding='ISO-8859-1')

# Load the summerOly_programs dataset (summerOly_programs.csv)
programs_df = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

# Filter for 'Swimming' from summerOly_programs dataset
swimming_data = programs_df[programs_df['Sport'] == 'Aquatics']

# Extract the total number of gold medals awarded in 'Swimming' for each year
gold_medals_per_year = swimming_data.iloc[:, 4:].sum(axis=0)  # Summing across all countries for each year

# Filter for 'Swimming' events in results_df
swimming_results = results_df[results_df['Sport'] == 'Aquatics']

# Create a list to store China's gold medal percentage for each year
china_gold_percentage_medals = []

# Loop through each year and calculate the percentage of gold medals won by China
for year in swimming_results['Year'].unique():
    year_data = swimming_results[swimming_results['Year'] == year]
    
    # Get the number of gold medals won by China in this year
    china_gold_medals = year_data['CHN'].sum()  # 'CHN' column represents China
    
    # Get the total number of gold medals awarded in that year from summerOly_programs dataset
    total_gold_medals_in_year = gold_medals_per_year.get(year, 0)  # Get the total from the new dataset
    
    # Calculate the percentage of total gold medals won by China
    if total_gold_medals_in_year > 0:  # Prevent division by zero
        china_gold_percentage = (china_gold_medals / total_gold_medals_in_year) * 100
    else:
        china_gold_percentage = 0  # If no gold medals were awarded in that year
    
    # Print out the total number of gold medals for that year and China's gold medals
    print(f"Year: {year} | Total Gold Medals: {total_gold_medals_in_year} | China Gold Medals: {china_gold_medals} | China Gold Percentage: {china_gold_percentage:.2f}%")
    
    # Store the result with the year
    china_gold_percentage_medals.append({
        'Year': year,
        'China_Gold_Percentage': china_gold_percentage
    })

# Create a DataFrame with the results
china_gold_percentage_df = pd.DataFrame(china_gold_percentage_medals)

# Polynomial Regression Model
X_poly = china_gold_percentage_df['Year'].values.reshape(-1, 1)
y = china_gold_percentage_df['China_Gold_Percentage'].values

# Create polynomial features (degree 2 for quadratic)
poly = PolynomialFeatures(degree=2)
X_poly_transformed = poly.fit_transform(X_poly)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly_transformed, y)

# Predict the next year's percentage (e.g., for year 2024)
next_year = 2024
next_year_poly = poly.transform(np.array([[next_year]]))
predicted_percentage = model.predict(next_year_poly)

print(f"\nPredicted gold medal percentage for China in {next_year}: {predicted_percentage[0]:.2f}%")

# Plotting the trend of China's gold medal percentage over the years
plt.figure(figsize=(10, 6))
plt.scatter(china_gold_percentage_df['Year'], china_gold_percentage_df['China_Gold_Percentage'], color='gold', label='Actual Data')
plt.plot(china_gold_percentage_df['Year'], model.predict(X_poly_transformed), color='red', linestyle='-', label='Polynomial Regression')
plt.scatter(next_year, predicted_percentage, color='blue', marker='x', label=f"Predicted for {next_year}")

# Adding labels and title
plt.title("China's Gold Medal Percentage in Swimming (Yearly)", fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Gold Medal Percentage', fontsize=12)
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
