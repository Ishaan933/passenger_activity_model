from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the trained model and column information
model = joblib.load('model_files/passenger_activity_model.pkl')
model_columns = joblib.load('model_files/model_columns.pkl')

# Define Flask app
app = Flask(__name__)

# Define categorical options
DAY_TYPE_OPTIONS = ['Weekday', 'Saturday', 'Sunday']
TIME_PERIOD_OPTIONS = ['Morning', 'Mid-Day', 'Evening', 'Night']

@app.route('/')
def home():
    return render_template('index.html', day_types=DAY_TYPE_OPTIONS, time_periods=TIME_PERIOD_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        future_date = request.form['future_date']
        day_type = request.form['day_type']
        time_period = request.form['time_period']

        # Create the input DataFrame
        input_data = pd.DataFrame([{**{f'Day_Type_{dt}': int(day_type == dt) for dt in DAY_TYPE_OPTIONS},
                                    **{f'Time_Period_{tp}': int(time_period == tp) for tp in TIME_PERIOD_OPTIONS}}])

        # Ensure all required columns are present
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_columns]

        # Predict using the model
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
