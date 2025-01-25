import pandas as pd

# Load the dataset (replace with your actual CSV file path)
data = pd.read_csv("summerOly_athletes.csv")

# Filter data to include only medal winners
medal_data = data[data['Medal'].isin(['Gold', 'Silver', 'Bronze'])]

# Group by Country and Sport, and count the number of medals
grouped_data = medal_data.groupby(['NOC', 'Sport']).size().reset_index(name='Medal Count')

# Find the strongest country for each sport
strongest_countries = grouped_data.loc[grouped_data.groupby('Sport')['Medal Count'].idxmax()]

# Sort by Sport for better readability
strongest_countries = strongest_countries.sort_values('Sport')

# Print the results
print("Strongest countries in each sport:")
print(strongest_countries)

# Save the results to a CSV file (optional)
strongest_countries.to_csv("strongest_countries_by_sport.csv", index=False)
