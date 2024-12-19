from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('passenger_activity_model.pkl')

# Categorical options
DAY_TYPE_OPTIONS = ['Weekday', 'Saturday', 'Sunday']
TIME_PERIOD_OPTIONS = ['Morning', 'Mid-Day', 'Evening', 'Night']

# HTML template for the prediction interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Passenger Activity Prediction</title>
</head>
<body>
    <h1>Predict Passenger Activity</h1>
    <form action="/predict" method="post">
        <label for="future_date">Future Date:</label>
        <input type="date" id="future_date" name="future_date" required><br><br>
        
        <label for="day_type">Day Type:</label>
        <select id="day_type" name="day_type" required>
            {% for day in day_types %}
                <option value="{{ day }}">{{ day }}</option>
            {% endfor %}
        </select><br><br>
        
        <label for="time_period">Time Period:</label>
        <select id="time_period" name="time_period" required>
            {% for period in time_periods %}
                <option value="{{ period }}">{{ period }}</option>
            {% endfor %}
        </select><br><br>
        
        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, day_types=DAY_TYPE_OPTIONS, time_periods=TIME_PERIOD_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        future_date = request.form['future_date']
        day_type = request.form['day_type']
        time_period = request.form['time_period']

        # Process inputs
        input_data = pd.DataFrame([{**{f'Day_Type_{dt}': int(day_type == dt) for dt in DAY_TYPE_OPTIONS},
                                    **{f'Time_Period_{tp}': int(time_period == tp) for tp in TIME_PERIOD_OPTIONS}}])

        # Load model columns
        model_columns = joblib.load('model_columns.pkl')
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_columns]

        # Predict
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
