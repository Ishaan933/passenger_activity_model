from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings_model.joblib')
rf_alightings = joblib.load('models/rf_alightings_model.joblib')
encodings = joblib.load('models/encodings.joblib')

# Load dataset
df = pd.read_csv('/path/to/stop_10637_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = request.form
        schedule_period_name = data['schedule_period_name']
        route_number = int(data['route_number'])
        route_name = data['route_name']
        day_type = data['day_type']
        time_period = data['time_period']
        
        # Determine year and month
        year = 2025 if '2025' in schedule_period_name else 2026
        month = {'Spring': 4, 'Summer': 7, 'Fall': 10, 'Winter': 1}[schedule_period_name.split()[0]]
        
        # Handle new schedule period names
        if schedule_period_name not in encodings['schedule_period_name']:
            max_encoding = max(encodings['schedule_period_name'].values())
            encodings['schedule_period_name'][schedule_period_name] = max_encoding + 1

        # Prepare input data
        input_data = pd.DataFrame({
            'year': [year],
            'month': [month],
            'schedule_period_name': [encodings['schedule_period_name'][schedule_period_name]],
            'route_number': [encodings['route_number'][route_number]],
            'route_name': [encodings['route_name'][route_name]],
            'day_type': [encodings['day_type'][day_type]],
            'time_period': [encodings['time_period'][time_period]]
        })

        # Make predictions
        boardings_prediction = rf_boardings.predict(input_data)[0]
        alightings_prediction = rf_alightings.predict(input_data)[0]

        return jsonify({
            "boardings": boardings_prediction,
            "alightings": alightings_prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Debug and Run
if __name__ == '__main__':
    app.run(debug=True)
