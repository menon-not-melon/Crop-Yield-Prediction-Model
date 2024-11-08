# Crop-Yield-Prediction-Model
This project is a machine learning application that predicts crop yield based on several input factors, such as the area, item (crop), rainfall, pesticides usage, and temperature. The application preprocesses input data, builds a model, and provides a prediction through a Flask web app interface.
## Project Overview
This project utilizes a machine learning model to predict crop yields using a dataset of agricultural data. The model uses features like average rainfall, pesticides usage, and average temperature to estimate the yield in hg/ha for various crops across different regions.

## Dataset
The dataset file is named yield_df.csv, and it should include the following columns:

- Area: Geographical area (state or region).
- Item: Type of crop.
- Year: Year of the data entry.
- average_rain_fall_mm_per_year: Average rainfall (mm).
- pesticides_tonnes: Pesticide usage (tonnes).
- avg_temp: Average temperature (°C).
- hg/ha_yield: Yield in hg/ha (target variable).

## Requirements
This project requires the following libraries:
- Python (>=3.6)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Flask
- pickle

To install the required packages, run:
```
pip install -r requirements.txt
```
