import pandas as pd

# Load the dataset
athletes_data = pd.read_csv("summerOly_athletes.csv")

# Filter only gold medal winners
gold_medals = athletes_data[athletes_data['Medal'] == 'Gold']

# Group by Year, Sport, and Country to count gold medals
gold_counts = gold_medals.groupby(['Year', 'Sport', 'NOC']).size().reset_index(name='Gold_Medals')

# Pivot the table to have countries as columns
pivot_table = gold_counts.pivot_table(index=['Year', 'Sport'], 
                                      columns='NOC', 
                                      values='Gold_Medals', 
                                      fill_value=0).reset_index()

# Save the new table to a CSV file
pivot_table.to_csv("gold_medals_by_sport_and_country.csv", index=False)

print("File 'gold_medals_by_sport_and_country.csv' created successfully!")
