# 🚴‍♀️ Bike Counters in Paris: Visualizing and Predicting Counts Through a Containerized Streamlit App

**Welcome to the repo!** 

This repository pertains to a **containerized Streamlit app** based on the bike counters installed by the Parisian council to monitor bike traffic. These counters use computer vision to detect bikes and send a total count each hour. They were installed to identify key cycling routes and track bike traffic flow throughout the city. The results help guide the development of cycling infrastructure as part of the [**Paris 2021-2026 Cycling Plan**](https://www.paris.fr/en/pages/a-new-cycling-plan-for-a-100-bikeable-city-28350).

The data from these counters is publicly available and updated frequently, making it ideal for a **dynamic Streamlit app**. This app can be used by policymakers, planners, or anyone interested in identifying busy cycling hotspots and times in Paris. The underlying models come from a previous project, which is available [here](https://github.com/cathal-brady/Cathal_Nadiy_Bike_Prediction). This project was part of my first year in the Data Science master's program, and the CatBoost regressor used in it performed well, finishing **3rd out of 30** in a Kaggle competition.

---

## 📁 The Repo Layout

### **1. Streamlit App Script**
A Python script that builds the app using both historical bike counter data and pre-trained models to predict bike counts based on user-defined inputs. The app displays historical data in an informative way and can be run in a containerized manner through the Docker image. The script, Dockerfile, and `requirements.txt` are included in this repo.

### **2. Data Folder**
This folder contains historical bike count data for all site IDs. To limit storage and remain within free data limits, the data is split into two files:  
- `clean_count_data.csv`: Contains bike counts.  
- `site_coordinates.csv`: Contains the location of all sites.  

The data includes nearly **1 million rows** and is frequently updated. Since downloading the full 1.3 GB dataset each time isn't feasible, users can update the data occasionally by running the `historical_counts_updater.py` script. This approach strikes a balance between keeping the data static and downloading it constantly.

### **3. Pre-trained CatBoost Models**
This folder contains pre-trained **CatBoost regressor models** for each site, trained on historical bike count data. If the data is updated, it's recommended to retrain the models by running the `model_trainer.py` script in the home directory.

The models use the following features:
- **Hour**: Sine and cosine transformations account for the cyclical nature of time.
- **Month**: 1-12.
- **Covid Restriction Level**: Indicates lockdown periods.
- **Holiday Variable**: Uses the `holidays` package to identify French holidays.
- **Weekend**: A binary variable to flag weekends.
- **Rush Hour**: A binary variable for French workday rush hours.

The models perform well because they account for the **spatial dependence** of the data by training separate models for each site.

---

## 🚀 Running the Streamlit Script

You can run the app from the **Dockerfile** using Docker Desktop. Here's how to get it up and running:

1. **Open Docker Desktop** (Install if you don't have it already).
2. From the repo's home directory, build the Docker image:
    ```bash
    docker build -t paris_bike_forecast .
    ```
3. Run the app:
    ```bash
    docker run -p 8501:8501 paris_bike_forecast
    ```
4. The app will be available at **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

Enjoy exploring the bike traffic data in Paris! 🚲🌍
