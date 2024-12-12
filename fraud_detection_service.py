from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Expanded training data with more diverse cases
training_data = pd.DataFrame({
    "amount": [
        # Normal retail transactions
        50, 20, 30, 100, 200, 75, 125, 175, 225, 275,
        # Grocery/daily purchases
        15, 25, 35, 45, 65, 85, 95, 115, 145, 185,
        # Large purchases
        500, 750, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 7500,
        # Very small transactions
        5, 8, 12, 17, 22, 27, 33, 38, 42, 47,
        # Medium range purchases
        150, 250, 350, 450, 550, 650, 750, 850, 950, 1050
    ],
    "timestamp": [
        int(datetime.strptime(f"2024-01-{day:02d} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
        for day in range(1, 6)
        for hour in [9, 11, 13, 15, 17, 19, 21, 23, 14, 16]  # Common transaction hours
    ],
    "location": [
        # Major cities with repeated patterns
        "PRISHTINE", "PRIZREN", "PEJE", "GJAKOVE", "GJILAN",
        "PRISHTINE", "FERIZAJ", "MITROVICE", "PEJE", "GJAKOVE",
        # Include some less common locations
        "VUSHTRRI", "PODUJEVE", "LIPJAN", "RAHOVEC", "MALISHEVE",
        "PRISHTINE", "PRIZREN", "PEJE", "GJAKOVE", "GJILAN",
        # Mix of locations
        "PRISHTINE", "FERIZAJ", "DRAGASH", "ISTOG", "KLINE",
        "DECAN", "JUNIK", "KACANIK", "SHTIME", "OBILIQ",
        # More major city transactions
        "PRISHTINE", "PRIZREN", "PEJE", "GJAKOVE", "GJILAN",
        "PRISHTINE", "FERIZAJ", "MITROVICE", "PEJE", "GJAKOVE",
        # Some border towns
        "HANI I ELEZIT", "VITIA", "FUSHE KOSOVE", "SKENDERAJ", "SUHAREKE",
        "PRISHTINE", "PRIZREN", "PEJE", "GJAKOVE", "DRENAS"
    ],
    "device": [
        # Common device mix
        "POS", "ATM", "POS", "MOBILE", "WEB",
        "POS", "ATM", "MOBILE", "POS", "ATM",
        # More varied device usage
        "MOBILE", "WEB", "POS", "ATM", "MOBILE",
        "POS", "POS", "ATM", "MOBILE", "WEB",
        # Include all device types
        "POS", "ATM", "MOBILE", "WEB", "POS",
        "ATM", "MOBILE", "POS", "WEB", "ATM",
        # Representative distribution
        "POS", "POS", "POS", "ATM", "ATM",
        "MOBILE", "MOBILE", "WEB", "WEB", "POS",
        # More common device patterns
        "POS", "ATM", "POS", "MOBILE", "WEB",
        "POS", "ATM", "MOBILE", "POS", "ATM"
    ]
})

# One-hot encode categorical variables
training_data_encoded = pd.get_dummies(training_data, columns=['location', 'device'])

# Initialize and train the Isolation Forest model with higher contamination
model = IsolationForest(
    n_estimators=100, 
    contamination=0.2,  # Increased contamination factor
    random_state=42     # For reproducibility
)
model.fit(training_data_encoded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.json
        amount = float(data['amount'])
        
        # Add explicit rules for obvious fraud cases
        if amount > 1000:  # High amount threshold
            return jsonify({
                "is_fraud": True,
                "anomaly_score": -0.9
            })
            
        if data['location'].upper() == 'UNKNOWN' or data['device'].upper() == 'UNKNOWN':
            return jsonify({
                "is_fraud": True,
                "anomaly_score": -0.8
            })
        
        # Create DataFrame with same structure as training data
        transaction = pd.DataFrame([{
            'amount': amount,
            'timestamp': int(datetime.strptime(data['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()),
            'location': data['location'].upper(),
            'device': data['device'].upper()
        }])
        
        # One-hot encode using same columns as training data
        transaction_encoded = pd.get_dummies(transaction, columns=['location', 'device'])
        
        # Align columns with training data
        for col in training_data_encoded.columns:
            if col not in transaction_encoded.columns:
                transaction_encoded[col] = 0

        # Ensure columns are in same order as training data
        transaction_encoded = transaction_encoded[training_data_encoded.columns]

        # Predict anomaly score and fraud status
        score = model.decision_function(transaction_encoded)[0]
        is_fraud = model.predict(transaction_encoded)[0] == -1

        # Add additional scoring logic
        if amount < 20 or amount > 500:  # Amount outside normal range
            score -= 0.3  # Make score more negative (more anomalous)
            is_fraud = True

        return jsonify({
            "is_fraud": bool(is_fraud),
            "anomaly_score": float(score)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": str(type(e).__name__)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)