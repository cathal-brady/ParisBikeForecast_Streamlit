import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# Load site coordinates and site names dataframe
site_data = pd.read_csv('data/site_coordinates.csv')

# Create a map centered around the average coordinates
avg_lat = site_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[0])).mean()
avg_lon = site_data['Geographical Coordinates'].apply(lambda x: float(x.split(',')[1])).mean()

# Initialize a folium map
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

for index, row in site_data.iterrows():
    site_name = row['Site Name']
    lat, lon = map(float, row['Geographical Coordinates'].split(','))
    
    # Use HTML to format the popup
    popup_content = f"<strong>{site_name}</strong>"
    
    folium.Marker(
        location=[lat, lon],
        popup=popup_content,  # Use formatted HTML content for the popup
        icon=folium.Icon(color='blue')
    ).add_to(m)

# Streamlit app title
st.title("Predicting Bicycle Count Around Paris")

st.write(
    """
    **Welcome to the Paris Bicycle Count Prediction App!**

    This application allows you to predict bike counts for any date and time period you specify. 
    You can select a site from the map below, which highlights areas where the Paris council has installed bike counters to monitor bike traffic.

    Historical data from these counters is utilised to forecast future bike traffic based on past patterns. 
    A unique model has been fitted for each site to account for the spatial dependence of the data. 
    The parameters for these models were fine-tuned in a previous project, whose repo is available [here](https://github.com/cathal-brady/Cathal_Nadiy_Bike_Prediction).
    
    This model achieved 0.55 RMSE at predicting log bike count and finished 3rd out of 30 teams in a Kaggle competition. 
    
    The background models predict log bike count as it addressed outliers and improved the distribution but predictions shown here have been reconverted back into standard units of bikes.
    
    More information is available regarding the bike counters at on the 
    [Paris open Data website.](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name)
    """
)

# Display the map
st.subheader("Map of all sites you can choose from")
st_folium(m, width=750)

st.subheader("Choose your parameters for the model")
# Dropdown for selecting a site
selected_site = st.selectbox("Select a site", site_data['Site Name'])
# Date input for selecting a date
selected_date = st.date_input("Select a date", value=pd.to_datetime("today"))
# Dropdown for selecting time in 24-hour format
hours = [f"{hour:02d}:00" for hour in range(24)]  # Generates time in HH:00 format
selected_time = st.selectbox("Select a time", hours)

# Display the selected site
st.write(f"You selected: {selected_site}")

# Optional: Show information about the selected site
if selected_site:
    site_info = site_data[site_data['Site Name'] == selected_site]
    st.write("Selected Site Information:")
    st.write(site_info)
