from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataset
file_path = 'dataset/stop_10637_data.csv'
df = pd.read_csv(file_path)

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings.pkl')
rf_alightings = joblib.load('models/rf_alightings.pkl')
encodings = joblib.load('models/encodings.pkl')

@app.route('/')
def home():
    # Serve the index.html page
    return render_template('index.html')
    
@app.route('/get_route_numbers', methods=['POST'])
def get_route_numbers():
    data = request.get_json()
    schedule_period_name = data.get('schedule_period_name')
    stop_number = data.get('stop_number', 10637)

    filtered_data = df[df['stop_number'] == stop_number]
    unique_route_numbers = filtered_data['route_number'].unique().tolist()

    return jsonify({'route_numbers': unique_route_numbers})

@app.route('/get_route_names', methods=['POST'])
def get_route_names():
    data = request.get_json()
    schedule_period_name = data.get('schedule_period_name')
    stop_number = data.get('stop_number', 10637)
    route_number = data.get('route_number')

    filtered_data = df[(df['stop_number'] == stop_number) & 
                       (df['route_number'] == route_number)]
    unique_route_names = filtered_data['route_name'].unique().tolist()

    return jsonify({'route_names': unique_route_names})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract inputs
    schedule_period_name = data['schedule_period_name']
    route_number = data['route_number']
    route_name = data['route_name']
    day_type = data['day_type']
    time_period = data['time_period']

    # Derive year and month
    year = 2025 if '2025' in schedule_period_name else 2026
    month = {'Spring': 4, 'Summer': 7, 'Fall': 10, 'Winter': 1}[schedule_period_name.split()[0]]

    # Handle unseen schedule period names
    if schedule_period_name not in encodings['schedule_period_name']:
        max_encoding = max(encodings['schedule_period_name'].values())
        encodings['schedule_period_name'][schedule_period_name] = max_encoding + 1

    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'schedule_period_name': [encodings['schedule_period_name'].get(schedule_period_name, -1)],
        'route_number': [encodings['route_number'].get(route_number, -1)],
        'route_name': [encodings['route_name'].get(route_name, -1)],
        'day_type': [encodings['day_type'].get(day_type, -1)],
        'time_period': [encodings['time_period'].get(time_period, -1)]
    })

    # Predict boardings and alightings
    boardings_prediction = rf_boardings.predict(input_data)[0]
    alightings_prediction = rf_alightings.predict(input_data)[0]

    return jsonify({
        'boardings_prediction': boardings_prediction,
        'alightings_prediction': alightings_prediction
    })

if __name__ == '__main__':
    # Use port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
