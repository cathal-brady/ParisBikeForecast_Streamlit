import pandas as pd
import streamlit as st
import joblib
import numpy as np
import holidays
import folium
from streamlit_folium import folium_static
import plotly.express as px
import matplotlib.pyplot as plt

# Load site coordinates and bike count data
site_coordinates = pd.read_csv('data/site_coordinates.csv')
bike_count_data = pd.read_csv('data/clean_count_data.csv')

# Merge site coordinates with bike count data using "Site Name" 
merged_data = pd.merge(bike_count_data, site_coordinates, on='Site Name', how='left')

# Calculate average latitude and longitude for map centering
avg_lat = merged_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[0])).mean()
avg_lon = merged_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[1])).mean()

# Initialize a folium map centered around the average coordinates
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

# Add markers to the map for each site
for index, row in site_coordinates.iterrows():
    lat, lon = map(float, row['Geographical Coordinates'].split(','))
    folium.Marker(
        location=[lat, lon],
        popup=row['Site Name'],
        icon=folium.Icon(color='blue')
    ).add_to(m)

# Streamlit app title and description
st.title("Predicting Bicycle Count Around Paris")

# GIF Embedding - for aesthetics
gif_url = "https://giphy.com/embed/5b3EelVhourTnxFkSF"
try:
    st.markdown(f'''
        <div style="display: flex; justify-content: center;">
            <iframe src="{gif_url}" width="240" height="240" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
        </div>
        ''', unsafe_allow_html=True)
except:
    st.write("")

st.subheader("Welcome to the Paris Bicycle Count Prediction App!")

st.write("""
    This application allows you to predict bike counts for any date and time period you specify. 
    You can select a site from the map below, which highlights areas where the Paris council has installed bike counters to monitor bike traffic.
    
    Historical data from these counters is utilized to forecast future bike traffic based on past patterns. 
    A unique model has been fitted for each site to account for the spatial dependence of the data. 
    More information is available regarding the bike counters at the
    [Paris open Data website.](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name)
""")

# Display the map of all sites
st.subheader("Map of all sites")
folium_static(m, width=600)

st.header("Looking at average historical traffic across time")
# Hour selector slider
hour = st.slider("Select hour of interest", 0, 23, 0)

# Date and Time in datetime format
merged_data['Date and Time'] = pd.to_datetime(merged_data['Date and Time'], utc=True, errors='coerce')
merged_data['Hour'] = merged_data['Date and Time'].dt.hour  # Add an hour column

# Group by Hour and Site Name, then calculate the mean of Hourly Count
grouped_data = merged_data.groupby(['Hour', 'Site Name']).agg({'Hourly Count': 'mean'}).reset_index()

# Filter the data for the selected hour
hourly_counts = grouped_data[grouped_data['Hour'] == hour]

# Merge back with site coordinates to get latitudes and longitudes
hourly_counts = pd.merge(hourly_counts, site_coordinates, on='Site Name', how='left')

# Parse latitudes and longitudes
latitudes, longitudes = zip(*hourly_counts['Geographical Coordinates'].apply(lambda x: (float(x.split(',')[0]), float(x.split(',')[1]))))

# Plotly scatter map with bike counts
fig = px.scatter_mapbox(
    hourly_counts,
    lat=latitudes,
    lon=longitudes,
    size=hourly_counts['Hourly Count'],
    color=hourly_counts['Hourly Count'],
    color_continuous_scale=px.colors.sequential.Reds,
    size_max=15,
    zoom=12,
    mapbox_style="open-street-map"
)

# Update plot layout
fig.update_layout(
    title=f"Average Bike Count Across Paris for {hour}:00",
    margin={"r":0, "t":40, "l":0, "b":0}
)

# Display plotly chart
st.plotly_chart(fig)

# Moving onto the prediction part
st.header("🚴‍♂️ Predict Bicycle Count with Pretrained Model")
st.subheader("📅 Choose Your Parameters for Prediction")

st.write("""
Select a site and time of interest to get predictions on bicycle counts using our pretrained CatBoost regressor model.
""")

# Function to determine restriction level based on dates (for Covid lockdown)
def get_restriction_level(date):
    date_ranges = [
        ('2020-10-17', '2020-11-28', 5),
        ('2020-11-28', '2020-12-15', 4),
        ('2020-12-15', '2021-01-16', 2),
        ('2021-01-16', '2021-03-19', 1),
        ('2021-03-19', '2021-06-09', 5),
        ]

    for start_date, end_date, level in date_ranges:
        if start_date <= date.strftime('%Y-%m-%d') < end_date:
            return level
    return 0  # Default level if no match

# Function to prepare model features
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

# UI for selecting site, date, and time
selected_site = st.selectbox("Select a site", merged_data['Site Name'].unique())
selected_date = st.date_input("Select a date", value=pd.to_datetime("today"))
hours = [f"{hour:02d}:00" for hour in range(24)]
selected_time = st.selectbox("Select a time", hours)

# Button to predict bike count
if st.button("Predict Bike Count"):
    features = prepare_features(selected_site, selected_date, selected_time)

    # Load the model for the selected site
    model_filename = f"pretrained_catboost_models/{selected_site}_catboost_model.joblib"
    model = joblib.load(model_filename)

    # Make prediction using the model
    prediction_log = model.predict(features)
    prediction_count = max(0, round(np.exp(prediction_log[0]) - 1))  # Undo log transformation

    st.subheader(f"Predicted Bike Count: {prediction_count}")

    # Load historical data and filter by site
    historical_data = pd.read_csv('data/clean_count_data.csv')
    site_historical_data = historical_data[historical_data['Site Name'] == selected_site].copy()

    # Ensure proper datetime conversion
    site_historical_data['Date and Time'] = pd.to_datetime(site_historical_data['Date and Time'], utc=True, errors='coerce')
    site_historical_data = site_historical_data.dropna(subset=['Date and Time'])

    # Extract hour for plotting
    site_historical_data['Hour'] = site_historical_data['Date and Time'].dt.hour

    # Calculate average hourly counts
    average_hourly_counts = site_historical_data.groupby('Hour')['Hourly Count'].mean().reset_index()

    # Create DataFrame for prediction
    selected_hour = pd.to_datetime(f"{selected_date} {selected_time}").hour
    prediction_df = pd.DataFrame({'Hour': [selected_hour], 'Hourly Count': [prediction_count]})

    # Plot historical data and prediction
    plt.figure(figsize=(10, 5))
    plt.plot(average_hourly_counts['Hour'], average_hourly_counts['Hourly Count'], label='Average Hourly Counts', marker='o')
    plt.axvline(x=selected_hour, color='red', linestyle='--', label='Predicted Hour')
    plt.scatter(prediction_df['Hour'], prediction_df['Hourly Count'], color='orange', label='Predicted Count')
    plt.title(f'Average Hourly Bike Count at {selected_site}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Bike Count')
    plt.xticks(range(24))
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Display the plot
    st.pyplot(plt)

    # Some info for the user to read
    st.write("""
        You may notice that the predictions often deviate from historical hourly averages for a site, even though only the date and time are used as inputs. 
        This deviation is intentional and results from the advanced feature engineering performed by the CatBoost regressor model, designed for higher accuracy.

        ### Key Aspects of Feature Engineering:
        - **Hour of the Day** (hr):
         The model captures the hour (1-23) and applies sine and cosine transformations. This ensures that 23:00 and 0:00 are treated as close values, reflecting the cyclical nature of time (i.e., the end of the day is not drastically different from the start).
        
        - **Public Holidays**:
         The model uses the Python holidays package to detect if the date is a French public holiday, adjusting the prediction to account for the typically lower or different traffic on such days.
        
        - **Weekend Adjustment**:
         Predictions are adjusted based on whether the date falls on a weekend, recognizing that weekend traffic patterns differ significantly from weekdays.
        
        - **Rush Hour Detection**:
         The model identifies if the given time is within typical morning or evening rush hours and adjusts the predictions to account for higher traffic during these periods.

        These engineered features were carefully selected and are the result of deliberate model development. You can find more details about the modeling process in the [GitHub repository](https://github.com/cathal-brady/Cathal_Nadiy_Bike_Prediction).

        The goal of these predictions is to provide a better understanding of bike traffic, both spatially and temporally, around Paris. For city planners and urbanists, these insights can aid in making informed decisions about where and when to implement traffic alleviation measures.
        """)
