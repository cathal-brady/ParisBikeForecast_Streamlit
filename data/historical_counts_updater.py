# WARNING - This script can be slow as we have to download the entire dataset which is ~1.3GB

import pandas as pd

# Define the dataset URL
dataset_url = 'https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B'

# Try to download the data
try:
    raw_df = pd.read_csv(dataset_url, sep=';')
except Exception as e:
    print(f"Error downloading the dataset: {e}")
    exit()

# Select specific columns
selected_columns = raw_df[['Nom du site de comptage', 
                            'Comptage horaire', 
                            'Date et heure de comptage', 
                            'Coordonnées géographiques']]

# Rename columns to English
df = selected_columns.rename(columns={
    'Nom du site de comptage': 'Site Name',
    'Date et heure de comptage': 'Date and Time',
    'Coordonnées géographiques': 'Geographical Coordinates',
    'Comptage horaire': 'Hourly Count',
})

# Create a summary to count entries per site
site_counts = df['Site Name'].value_counts()

# Filter for sites with 100 or more counts
valid_sites = site_counts[site_counts >= 100].index

# Keep only the entries in df for these valid sites
df = df[df['Site Name'].isin(valid_sites)]

# Create unique sites DataFrame
unique_sites = df[['Site Name', 'Geographical Coordinates']].drop_duplicates()

# Drop the original Geographical Coordinates column from the main DataFrame - this is for efficiency 
df = df.drop(columns=['Geographical Coordinates'])

# Reset index for clarity
df.reset_index(drop=True, inplace=True)
unique_sites.reset_index(drop=True, inplace=True)

# Save the cleaned data to CSV files
df.to_csv('clean_count_data.csv', index=False)
unique_sites.to_csv('site_coordinates.csv', index=False)

print("Data processing complete. Files saved as 'clean_count_data.csv' and 'site_coordinates.csv'.")
