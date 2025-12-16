# Train Delay Prediction Project

## Project Overview
This project aims to predict historical train delays based on various factors such as distance, weather conditions, day of the week, time of day, train type, and route congestion. The goal is to develop a machine learning model that can accurately estimate train delays, which can be useful for operational planning and passenger information.

## Data
The dataset `data.csv` contains information about train journeys, including:
- `Distance Between Stations (km)`: The distance covered by the train.
- `Weather Conditions`: Environmental conditions during the journey (e.g., Clear, Rainy, Foggy).
- `Day of the Week`: The day on which the journey took place.
- `Time of Day`: The part of the day (e.g., Morning, Afternoon, Evening, Night).
- `Train Type`: The classification of the train (e.g., Express, Superfast, Local).
- `Historical Delay (min)`: The target variable, representing the historical delay in minutes.
- `Route Congestion`: The level of congestion on the route (e.g., Low, Medium, High).

## Exploratory Data Analysis (EDA)
- **Null Value Check**: Confirmed zero null values in the dataset.
- **Data Types**: Verified that all data types are correctly fitted to their respective features.
- **Outlier Detection**: Identified and removed outliers in `Distance Between Stations (km)` and `Historical Delay (min)` using the IQR method, improving data quality for modeling.
- **Correlation Analysis**: A heatmap was generated to show the correlation between numerical features. 'Distance Between Stations (km)' and 'Historical Delay (min)' showed a negative correlation.
- **Categorical Distribution**: Count plots were used to visualize the distribution of each categorical feature, ensuring the data is balanced across categories.

## Preprocessing
- **Outlier Removal**: Outliers in numerical columns were removed using the IQR (Interquartile Range) method.
- **Label Encoding**: All categorical features (`Weather Conditions`, `Day of the Week`, `Time of Day`, `Train Type`, `Route Congestion`) were converted into numerical representations using `LabelEncoder`.

## Model Building and Evaluation
Several regression models were trained and evaluated:
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**

The models were evaluated using `Mean Absolute Error (MAE)`, `Mean Squared Error (MSE)`, `Root Mean Squared Error (RMSE)`, and `R² Score`.

### Model Performance (After initial training):
- **Linear Regression**: MAE: 21.664, MSE: 692.666, RMSE: 26.319, R² Score: 0.161
- **Random Forest**: MAE: 18.586, MSE: 643.166, RMSE: 25.361, R² Score: 0.221
- **XGBoost**: MAE: 20.026, MSE: 846.902, RMSE: 29.102, R² Score: -0.025
- **CatBoost**: MAE: 18.735, MSE: 655.834, RMSE: 25.609, R² Score: 0.206

### Hyperparameter Tuning
Hyperparameter tuning was performed on Random Forest and XGBoost:
- **Random Forest Tuned**: MAE: 18.678, MSE: 613.725, RMSE: 24.773, R² Score: 0.257
- **XGBoost Tuned**: MAE: 19.959, MSE: 820.926, RMSE: 28.652, R² Score: 0.006

## Final Model Selection
The **RandomForestRegressor** model (with tuning) was selected as the final model due to its best performance (lowest MAE/RMSE and highest R² Score) among the evaluated models.

## Model Saving
The final Random Forest model has been saved as `model.pkl` using `joblib` for future use and deployment.

## Streamlit Application
A Streamlit application (`app.py`) has been provided to interact with the trained model. This application allows users to input various parameters and receive a predicted train delay.

Live Demo: https://train-delay-prediction-databyab.streamlit.app/

### How to Run the Streamlit App:
1.  Save the generated Streamlit code into a file named `app.py`.
2.  Install Streamlit: `pip install streamlit`
3.  Run the application from your terminal in the directory where `app.py` is saved: `streamlit run app.py`

## Setup and Installation
To replicate this project, you need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost joblib streamlit
```

## Usage
1.  Run through the Jupyter Notebook cells to preprocess the data, train the models, and save the best model.

2.  Use the `app.py` Streamlit script to interact with the deployed model for delay predictions.
