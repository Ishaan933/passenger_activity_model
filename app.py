from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and encodings
rf_boardings = joblib.load('models/rf_boardings.pkl')
rf_alightings = joblib.load('models/rf_alightings.pkl')
encodings = joblib.load('models/encodings.pkl')

@app.route('/')
def home():
    return "Transit Prediction Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        
        # Perform predictions
        boardings = rf_boardings.predict(input_data)[0]
        alightings = rf_alightings.predict(input_data)[0]
        
        return jsonify({
            "boardings": boardings,
            "alightings": alightings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
