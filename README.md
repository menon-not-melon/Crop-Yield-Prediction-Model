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

## Installation
1. Clone the repository:
```
git clone <repository_url>
```
2. Navigate to the project directory:
```
cd crop-yield-prediction
```
3. Place the dataset file ```(yield_df.csv)``` in the project directory.

## Usage
1. Preprocess the Data: 
Run the preprocessing script to clean and prepare data for training. This script also handles missing values, outliers, and normalizes data.
2. Train the Model: 
Run the ```model.py``` script to train the ExtraTreesRegressor model on the dataset. This will save a trained model as a pickle file ```(model.pkl)```.
```
python model.py
```
3. Run the Web Application: Start the Flask app to use the web interface for prediction.
```
python app.py
```

## Model Training
The model uses an ExtraTreesRegressor from the scikit-learn library. 
Key preprocessing steps:
1. Label Encoding: Applied to categorical columns.
2. OneHot Encoding: Used for certain categorical features to improve model interpretability.
3. Min-Max Scaling: Normalizes numerical features to a scale of 0 to 5.
4. Outlier Removal: Using quantiles for specified features.

### Application Structure
- app.py: Main Flask application that provides web interface and API.
- model.py: Contains code for data preprocessing, model training, and saving the model.
- yield_df.csv: Dataset for training and prediction.
- model.pkl: Saved model for predictions.
- requirements.txt: List of required packages.

## License
This project is licensed under the MIT License.
