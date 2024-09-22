# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /ParisBikeForecast_Streamlit

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/cathal-brady/ParisBikeForecast_Streamlit .

RUN pip3 install -r requirements.txt

COPY data/ data/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_bike_count_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
