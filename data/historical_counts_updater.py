# WARNING: This script can be slow, as it downloads the entire dataset (~1.3GB).

import pandas as pd

# Define the dataset URL
dataset_url = (
    'https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/'
    'comptage-velo-donnees-compteurs/exports/csv?lang=fr&timezone='
    'Europe%2FParis&use_labels=true&delimiter=%3B'
)

# Try to download the dataset and handle potential errors
try:
    raw_df = pd.read_csv(dataset_url, sep=';')
except Exception as e:
    print(f"Error downloading the dataset: {e}")
    exit()

# Select relevant columns from the raw DataFrame
selected_columns = raw_df[['Nom du site de comptage',
                            'Comptage horaire',
                            'Date et heure de comptage',
                            'Coordonnées géographiques']]

# Rename columns to English for easier understanding
df = selected_columns.rename(columns={
    'Nom du site de comptage': 'Site Name',
    'Date et heure de comptage': 'Date and Time',
    'Coordonnées géographiques': 'Geographical Coordinates',
    'Comptage horaire': 'Hourly Count',
})

df['Date and Time'] = pd.to_datetime(df['Date and Time'], utc=True)

# Create a summary of the number of entries for each site
site_counts = df['Site Name'].value_counts()

# Filter for sites that have 100 or more counts
valid_sites = site_counts[site_counts >= 100].index

# Retain only the entries for the valid sites in the DataFrame
df = df[df['Site Name'].isin(valid_sites)]

# Create a DataFrame of unique sites with their coordinates
unique_sites = df[['Site Name', 'Geographical Coordinates']].drop_duplicates()

# Drop the Geographical Coordinates column from the main DataFrame for efficiency
df = df.drop(columns=['Geographical Coordinates'])

# Reset the index of both DataFrames for clarity
df.reset_index(drop=True, inplace=True)
unique_sites.reset_index(drop=True, inplace=True)

# Save the cleaned data to CSV files
df.to_csv('clean_count_data.csv', index=False)
unique_sites.to_csv('site_coordinates.csv', index=False)

print("Data processing complete. Files saved as 'clean_count_data.csv' and 'site_coordinates.csv'.")
