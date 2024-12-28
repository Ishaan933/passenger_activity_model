from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings_model.joblib')
rf_alightings = joblib.load('models/rf_alightings_model.joblib')
encodings = joblib.load('models/encodings.joblib')

@app.route('/')
def home():
    return "Transit Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process input data
    # Example input: {"year": 2025, "month": 7, ...}
    input_data = pd.DataFrame([data])
    boardings = rf_boardings.predict(input_data)[0]
    alightings = rf_alightings.predict(input_data)[0]
    return jsonify({"boardings": boardings, "alightings": alightings})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
