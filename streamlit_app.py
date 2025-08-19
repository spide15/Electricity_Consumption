import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('electicity_xgb_prediction_model.pkl')

model = load_model()

st.title('Electricity Demand Prediction Dashboard')
st.write('Upload your data or enter a single data point to predict electricity demand using the trained XGBoost model.')

# Helper: Feature engineering (must match training pipeline)
FEATURE_ORDER = [
    'hour', 'dayofweek', 'month', 'year', 'dayofyear', 'weekofyear', 'quarter', 'is_weekend',
    'Temperature', 'Humidity', 'Demand_lag_24hr', 'demand_lag_168hr', 'demand_rolling_mean_24hr', 'demand_rolling_std_24hr'
]

def preprocess(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5,6]).astype(int)
    df['Demand_lag_24hr'] = np.nan
    df['demand_lag_168hr'] = np.nan
    df['demand_rolling_mean_24hr'] = np.nan
    df['demand_rolling_std_24hr'] = np.nan
    # Ensure Temperature and Humidity columns exist for manual entry
    if 'Temperature' not in df.columns:
        df['Temperature'] = np.nan
    if 'Humidity' not in df.columns:
        df['Humidity'] = np.nan
    # Reindex to the correct order
    return df.reindex(columns=FEATURE_ORDER)

# File uploader
data_file = st.file_uploader('Upload CSV file', type=['csv'])

if data_file:
    user_df = pd.read_csv(data_file)
    st.write('Preview of uploaded data:')
    st.dataframe(user_df.head())
    # Preprocess
    X_user = preprocess(user_df)
    # Predict
    preds = model.predict(X_user)
    st.write('Predictions:')
    st.dataframe(pd.DataFrame({'Timestamp': user_df['Timestamp'], 'Predicted Demand': preds}))
    # If actual demand is present, show accuracy
    if 'Demand' in user_df.columns:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mask = ~user_df['Demand'].isna()
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(user_df.loc[mask, 'Demand'], preds[mask]))
            mae = mean_absolute_error(user_df.loc[mask, 'Demand'], preds[mask])
            st.write(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')

st.markdown('---')
st.header('Manual Single Data Point Entry')

with st.form('manual_entry'):
    timestamp = st.text_input('Timestamp (YYYY-MM-DD HH:MM:SS)', value=datetime.now().strftime('%Y-%m-%d %H:00:00'))
    temperature = st.number_input('Temperature (°C)', value=25.0)
    humidity = st.number_input('Humidity (%)', value=50.0)
    # Add more fields as needed for your model
    submitted = st.form_submit_button('Predict')
    if submitted:
        try:
            dt = pd.to_datetime(timestamp)
            row = {
                'Timestamp': dt,
                'Temperature': temperature,
                'Humidity': humidity,
                # Fill other columns with default or NaN as needed
            }
            # Add all required columns with NaN/defaults
            for col in ['Demand']:
                row[col] = np.nan
            df_single = pd.DataFrame([row])
            X_single = preprocess(df_single)
            pred = model.predict(X_single)[0]
            st.success(f'Predicted Demand: {pred:.2f} MW')
        except Exception as e:
            st.error(f'Error: {e}')

st.markdown('---')
st.write('Note: For best results, uploaded data should match the format and features used in model training.')
st.write('By Deep Gada')