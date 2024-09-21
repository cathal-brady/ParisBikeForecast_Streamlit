import pandas as pd
import streamlit as st
import joblib
import numpy as np
import holidays
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Load site names
site_data = pd.read_csv('data/site_coordinates.csv')

# Calculate average latitude and longitude for map centering
avg_lat = site_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[0])).mean()
avg_lon = site_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[1])).mean()

# Initialize a folium map centered around the average coordinates
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

# Add markers to the map for each site
for index, row in site_data.iterrows():
    lat, lon = map(float, row['Geographical Coordinates'].split(','))
    folium.Marker(
        location=[lat, lon],
        popup=row['Site Name'],
        icon=folium.Icon(color='blue')
    ).add_to(m)

# Streamlit app title and description
st.title("Predicting Bicycle Count Around Paris")
st.write("""
    **Welcome to the Paris Bicycle Count Prediction App!**

    This application allows you to predict bike counts for any date and time period you specify. 
    You can select a site from the map below, which highlights areas where the Paris council has installed bike counters to monitor bike traffic.
    
    Historical data from these counters is utilized to forecast future bike traffic based on past patterns. 
    A unique model has been fitted for each site to account for the spatial dependence of the data. 
    More information is available regarding the bike counters at the
    [Paris open Data website.](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name)
""")

# Display the map with all sites
st.subheader("Map of all sites you can choose from")
st_folium(m, width=700, height=500)

# Section to choose parameters for the model
st.subheader("Choose your parameters for the model")

# Function to determine restriction level
def get_restriction_level(date):
    date_ranges = [
        ('2020-10-16', '2020-10-17', 3),
        ('2020-10-17', '2020-11-28', 5),
        # Add all other ranges as needed...
    ]
    for start_date, end_date, level in date_ranges:
        if start_date <= date.strftime('%Y-%m-%d') < end_date:
            return level
    return 0  # Default level

# Function to prepare features
def prepare_features(selected_site, selected_date, selected_time):
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_time}")
    month = selected_datetime.month
    weekday = selected_datetime.weekday() + 1  # Monday=1, Sunday=7
    hr = selected_datetime.hour
    hr_sin = np.sin(hr * (2 * np.pi / 24))
    hr_cos = np.cos(hr * (2 * np.pi / 24))
    restriction_level = get_restriction_level(selected_datetime)
    fr_holidays = holidays.FR()
    is_holiday = 1 if selected_datetime in fr_holidays else 0
    is_weekend = 1 if weekday in [6, 7] else 0
    rush_hour = int((weekday <= 5) and ((hr in range(7, 10)) or (hr in range(17, 21))) and (is_holiday == 0))

    return pd.DataFrame({
        'month': [month],
        'weekday': [weekday],
        'hr': [hr],
        'hr_sin': [hr_sin],
        'hr_cos': [hr_cos],
        'restriction_level': [restriction_level],
        'is_holiday': [is_holiday],
        'is_weekend': [is_weekend],
        'rush_hour': [rush_hour],
    })

# Streamlit UI
selected_site = st.selectbox("Select a site", site_data['Site Name'])
selected_date = st.date_input("Select a date", value=pd.to_datetime("today"))
hours = [f"{hour:02d}:00" for hour in range(24)]
selected_time = st.selectbox("Select a time", hours)

# Add a button for prediction
if st.button("Predict Bike Count"):
    features = prepare_features(selected_site, selected_date, selected_time)

    # Load the model
    model_filename = f"pretrained_catboost_models/{selected_site}_catboost_model.joblib"
    model = joblib.load(model_filename)

    # Make prediction
    prediction_log = model.predict(features)
    prediction_count = max(0, round(np.exp(prediction_log[0]) - 1))  # Undo log transformation

    st.write(f"Predicted Bike Count: {prediction_count}")

    # Load historical data and filter for the selected site
    historical_data = pd.read_csv('data/clean_count_data.csv')
    site_historical_data = historical_data[historical_data['Site Name'] == selected_site]

    # Convert 'Date and Time' to datetime and extract hour
    site_historical_data['Date and Time'] = pd.to_datetime(site_historical_data['Date and Time'], utc=True, errors='coerce')
    site_historical_data = site_historical_data.dropna(subset=['Date and Time'])
    site_historical_data['Hour'] = site_historical_data['Date and Time'].dt.hour

    # Calculate average counts per hour
    average_hourly_counts = site_historical_data.groupby('Hour')['Hourly Count'].mean().reset_index()

    # Create a DataFrame for the prediction
    selected_hour = pd.to_datetime(f"{selected_date} {selected_time}").hour
    prediction_df = pd.DataFrame({'Hour': [selected_hour], 'Hourly Count': [prediction_count]})

    # Combine average counts and prediction
    combined_counts = pd.concat([average_hourly_counts, prediction_df], ignore_index=True)

    # Sort by hour for correct plotting
    combined_counts.sort_values('Hour', inplace=True)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(combined_counts['Hour'], combined_counts['Hourly Count'], label='Average Hourly Counts', marker='o')
    plt.axvline(x=selected_hour, color='red', linestyle='--', label='Predicted Hour')
    plt.scatter(selected_hour, prediction_count, color='orange', label='Predicted Count', zorder=5)
    plt.title(f'Average Hourly Bike Count at {selected_site}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Bike Count')
    plt.xticks(range(24))  # Show hours from 0 to 23
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)
