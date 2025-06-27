import numpy as np
import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import os

def load_prophet_pickle(filepath):
    """
    Load Prophet model from pickle file
    
    Args:
        filepath: Path to the saved model file
    
    Returns:
        Loaded Prophet model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

def prepare_new_data(df_new, behavior_type='Buy'):
    """
    Prepare new data for Prophet model
    
    Args:
        new_data_path: Path to new CSV file
        behavior_type: Type of behavior to analyze ('buy', 'cart', 'fav', 'pv')
    
    Returns:
        DataFrame with 'ds' and 'y' columns for Prophet
    """
    
    # Convert timestamp to datetime
    df_new['datetime'] = pd.to_datetime(df_new['Timestamp'], unit='s')
    
    # Filter by behavior type and group by hour
    clicks_new = df_new[df_new['Behavior'] == behavior_type].groupby(
        pd.Grouper(key='datetime', freq='h')
    ).size().reset_index(name='y')
    
    # Rename columns for Prophet
    clicks_new = clicks_new.rename(columns={'datetime': 'ds'})
    
    print(f"Prepared {len(clicks_new)} hourly data points for '{behavior_type}' behavior")
    return clicks_new

def update_model_with_new_data(model, new_data):
    """
    Update existing Prophet model with new data
    
    Args:
        model: Existing Prophet model
        new_data: New data with 'ds' and 'y' columns
    
    Returns:
        Updated Prophet model
    """
    # Combine existing training data with new data
    # Note: Prophet doesn't have incremental learning, so we need to retrain
    print("Retraining model with new data...")
    
    # Create a new model with same parameters
    new_model = Prophet(
        daily_seasonality=model.daily_seasonality,
        weekly_seasonality=model.weekly_seasonality,
        yearly_seasonality=model.yearly_seasonality
    )
    
    # Fit with new data
    new_model.fit(new_data)
    
    return new_model

def convert_plot_to_base64(fig):
    """
    Convert matplotlib figure to base64 encoded image string
    
    Args:
        fig: Matplotlib figure object
    
    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # Close the figure to free memory
    return image_base64

def predict_future_hours(model_path, hours, behavior_type='Buy'):
    """
    First function: Predict the future after specified hours using existing model
    
    Args:
        model_path: Path to the saved Prophet model
        hours: Number of hours to predict into the future
        behavior_type: Type of behavior to analyze ('buy', 'cart', 'fav', 'pv')
    
    Returns:
        Dictionary with predictions and plot images (base64 encoded)
    """
    # Load the existing model
    model = load_prophet_pickle(model_path)
    
    # Make future predictions
    print(f"Making predictions for the next {hours} hours...")
    future = model.make_future_dataframe(periods=hours, freq='h')
    forecast = model.predict(future)
    
    # Extract only future predictions
    future_predictions = forecast.tail(hours)
    
    # Display prediction summary
    print(f"\nNext {hours} hours predictions:")
    print(future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Generate forecast plot
    forecast_fig = model.plot(forecast)
    forecast_fig.suptitle(f"Prophet Forecast - Next {hours} Hours")
    forecast_fig.axes[0].set_xlabel('Date')
    forecast_fig.axes[0].set_ylabel('Count')
    plt.tight_layout()
    
    forecast_image = convert_plot_to_base64(forecast_fig)
    
    # Generate components plot
    components_fig = model.plot_components(forecast)
    plt.tight_layout()
    
    components_image = convert_plot_to_base64(components_fig)
    
    return {
        'predictions': future_predictions,
        'forecast_plot': forecast_image,
        'components_plot': components_image
    }

def update_and_predict(model_path, df, hours, behavior_type='Buy'):
    """
    Second function: Update model with new CSV data and predict future hours
    
    Args:
        model_path: Path to the saved Prophet model
        new_csv_path: Path to new CSV file for updating the model
        hours: Number of hours to predict into the future
        behavior_type: Type of behavior to analyze ('buy', 'cart', 'fav', 'pv')
    
    Returns:
        Dictionary with updated model, predictions, and plot images (base64 encoded)
    """
    # Load the existing model
    model = load_prophet_pickle(model_path)
    
    # Prepare new data
    new_data = prepare_new_data(df, behavior_type)
    
    # Update model with new data
    updated_model = update_model_with_new_data(model, new_data)
    
    # Make future predictions with updated model
    print(f"Making predictions for the next {hours} hours with updated model...")
    future = updated_model.make_future_dataframe(periods=hours, freq='h')
    forecast = updated_model.predict(future)
    
    # Extract only future predictions
    future_predictions = forecast.tail(hours)
    
    # Display prediction summary
    print(f"\nNext {hours} hours predictions (updated model):")
    print(future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Generate updated forecast plot
    forecast_fig = updated_model.plot(forecast)
    forecast_fig.suptitle(f"Updated Prophet Forecast - Next {hours} Hours")
    forecast_fig.axes[0].set_xlabel('Date')
    forecast_fig.axes[0].set_ylabel('Count')
    plt.tight_layout()
    
    forecast_image = convert_plot_to_base64(forecast_fig)
    
    # Generate updated components plot
    components_fig = updated_model.plot_components(forecast)
    plt.tight_layout()
    
    components_image = convert_plot_to_base64(components_fig)
    
    return {
        'updated_model': updated_model,
        'predictions': future_predictions,
        'forecast_plot': forecast_image,
        'components_plot': components_image
    }


# Example usage functions
def example_usage():
    """
    Example of how to use the two functions with image output
    """
    model_path = "Trained_Model/prophet_model.pkl"
    
    # Example 1: Predict future 24 hours using existing model
    print("=== Function 1: Predict Future Hours (returns images) ===")
    result_1 = predict_future_hours(
        model_path=model_path,
        hours=24,
        behavior_type='Buy'
    )
    
    # Example 2: Update model with new data and predict future 48 hours
    print("\n=== Function 2: Update and Predict (returns images) ===")
    result_2 = update_and_predict(
        model_path=model_path,
        new_csv_path="csv_chunks/chunk_20.csv",
        hours=48,
        behavior_type='Buy'
    )
    
    # Optional: Display images in Jupyter notebook
    # Uncomment the following lines if running in Jupyter
    """
    display_image_from_base64(result_1['forecast_plot'], "Forecast Plot 1")
    display_image_from_base64(result_1['components_plot'], "Components Plot 1")
    display_image_from_base64(result_2['forecast_plot'], "Updated Forecast Plot")
    display_image_from_base64(result_2['components_plot'], "Updated Components Plot")
    """
    
    return result_1, result_2