import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings.pkl')
rf_alightings = joblib.load('models/rf_alightings.pkl')
encodings = joblib.load('models/encodings.pkl')

# Load dataset
df = pd.read_csv('dataset/stop_10637_data.csv')

# Constants for dropdowns
SCHEDULE_PERIODS = ['', 'Summer 2025', 'Fall 2025', 'Spring 2025', 'Winter 2026']
DAY_TYPES = ['', 'Weekday', 'Saturday', 'Sunday']
TIME_PERIODS = ['', 'Morning', 'Mid-Day', 'PM Peak', 'Evening', 'Night']

@app.route('/')
def home():
    """
    Render the main page with dropdowns preloaded with options.
    """
    route_numbers = [''] + df['route_number'].unique().tolist()
    return render_template(
        'index.html',
        schedule_periods=SCHEDULE_PERIODS,
        route_numbers=route_numbers,
        day_types=DAY_TYPES,
        time_periods=TIME_PERIODS
    )

@app.route('/get_route_names', methods=['POST'])
def get_route_names():
    """
    Get route names based on the selected route number.
    """
    try:
        # Retrieve the route_number from the request
        route_number = request.json.get('route_number', None)
        app.logger.info(f"Fetching route names for route number: {route_number}")
        
        # Handle empty or missing route_number
        if not route_number:
            return jsonify([''])  # Return an empty list for empty input
        
        # Filter route names based on route_number
        route_number = int(route_number)
        route_names = [''] + df[df['route_number'] == route_number]['route_name'].unique().tolist()
        app.logger.info(f"Found route names: {route_names}")
        
        return jsonify(route_names)
    except Exception as e:
        app.logger.error(f"Error in get_route_names: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict average boardings and alightings based on user inputs.
    """
    try:
        # Retrieve form data
        data = request.form
        schedule_period_name = data.get('schedule_period_name', '')
        route_number = data.get('route_number', '')
        route_name = data.get('route_name', '')
        day_type = data.get('day_type', '')
        time_period = data.get('time_period', '')

        # Validate inputs
        if not (schedule_period_name and route_number and route_name and day_type and time_period):
            app.logger.warning("Missing input fields for prediction.")
            return render_template(
                'index.html',
                error="All fields are required for prediction.",
                schedule_periods=SCHEDULE_PERIODS,
                route_numbers=[''] + df['route_number'].unique().tolist(),
                day_types=DAY_TYPES,
                time_periods=TIME_PERIODS
            )

        # Convert inputs
        route_number = int(route_number)
        year = 2025 if '2025' in schedule_period_name else 2026
        month = {'Spring': 4, 'Summer': 7, 'Fall': 10, 'Winter': 1}[schedule_period_name.split()[0]]

        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'year': year,
            'month': month,
            'schedule_period_name': encodings['schedule_period_name'][schedule_period_name],
            'route_number': encodings['route_number'][route_number],
            'route_name': encodings['route_name'][route_name],
            'day_type': encodings['day_type'][day_type],
            'time_period': encodings['time_period'][time_period]
        }])

        # Perform predictions
        boardings_prediction = rf_boardings.predict(input_data)[0]
        alightings_prediction = rf_alightings.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction={
                "boardings": f"{boardings_prediction:.2f}",
                "alightings": f"{alightings_prediction:.2f}"
            },
            schedule_periods=SCHEDULE_PERIODS,
            route_numbers=[''] + df['route_number'].unique().tolist(),
            day_types=DAY_TYPES,
            time_periods=TIME_PERIODS
        )
    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}")
        return render_template(
            'index.html',
            error="An error occurred during prediction. Please check your inputs.",
            schedule_periods=SCHEDULE_PERIODS,
            route_numbers=[''] + df['route_number'].unique().tolist(),
            day_types=DAY_TYPES,
            time_periods=TIME_PERIODS
        )

if __name__ == '__main__':
    # Use port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
