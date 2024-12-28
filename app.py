from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings_model.joblib')
rf_alightings = joblib.load('models/rf_alightings_model.joblib')
encodings = joblib.load('models/encodings.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request (JSON or form)
        data = request.json or request.form
        
        # Process inputs
        year = int(data['year'])
        month = int(data['month'])
        schedule_period_name = data['schedule_period_name']
        route_number = int(data['route_number'])
        route_name = data['route_name']
        day_type = data['day_type']
        time_period = data['time_period']
        
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([{
            "year": year,
            "month": month,
            "schedule_period_name": encodings['schedule_period_name'][schedule_period_name],
            "route_number": encodings['route_number'][route_number],
            "route_name": encodings['route_name'][route_name],
            "day_type": encodings['day_type'][day_type],
            "time_period": encodings['time_period'][time_period]
        }])

        # Perform predictions
        boardings = rf_boardings.predict(input_data)[0]
        alightings = rf_alightings.predict(input_data)[0]
        
        # Return results as JSON
        return jsonify({
            "boardings": boardings,
            "alightings": alightings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
