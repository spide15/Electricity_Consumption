# Electricity Demand Prediction Dashboard

This project provides an interactive Streamlit dashboard for predicting electricity demand using a trained XGBoost model. Users can upload a CSV file or manually enter data to get predictions.

## Files
- `streamlit_app.py`: The Streamlit dashboard application.
- `electicity_xgb_prediction_model.pkl`: Trained XGBoost model.
- `Electricity.csv`: Example dataset.

## Setup Instructions

1. **Install Python Packages**
   Open a terminal in the project directory and run:
   ```
   pip install --user streamlit pandas numpy joblib scikit-learn xgboost
   ```

2. **Run the Dashboard**
   In the terminal, run:
   ```
   python -m streamlit run streamlit_app.py
   ```

3. **Usage**
   - **Upload CSV**: Upload a file with columns like `Timestamp`, `Temperature`, `Humidity`, etc. The app will preprocess and predict demand for each row.
   - **Manual Entry**: Enter a single data point (timestamp, temperature, humidity) and get an instant prediction.
   - Units are shown for all inputs and outputs (e.g., Temperature in °C, Humidity in %, Demand in MW).

4. **Notes**
   - The uploaded data should match the format and features used in model training.
   - If you encounter missing package errors, install them using the `pip install --user <package>` command.

## Author
By Deep gada
