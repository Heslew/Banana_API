from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# ═══════════════════════════════════════════════════════
# LOAD MODEL (PIPELINE VERSION)
# ═══════════════════════════════════════════════════════
try:
    model = joblib.load('banana_model_pipeline.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    print("✓ Model loaded successfully")
    print(f"✓ Classes: {label_encoder.classes_}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoder = None


# ═══════════════════════════════════════════════════════
# HOME ROUTE
# ═══════════════════════════════════════════════════════
@app.route('/')
def home():
    return "🍌 Banana Disease Prediction API is running!"


# ═══════════════════════════════════════════════════════
# PREDICTION ROUTE
# ═══════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json

        # Get inputs safely
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        moisture = float(data.get('moisture', 0))
        rainfall = float(data.get('rainfall', 0))

        print(f"[INPUT] temp={temperature}, hum={humidity}, moist={moisture}, rain={rainfall}")

        # ═══════════════════════════════════════════════
        # FEATURE ENGINEERING (MUST MATCH TRAINING)
        # ═══════════════════════════════════════════════
        temp_hum = temperature * humidity
        moist_rain = moisture * rainfall
        stress = (temperature + humidity + moisture) / 3

        features = np.array([[
            temperature,
            humidity,
            moisture,
            rainfall,
            temp_hum,
            moist_rain,
            stress
        ]])

        # ═══════════════════════════════════════════════
        # PREDICTION
        # ═══════════════════════════════════════════════
        pred_class = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = float(np.max(probabilities))
        disease_name = label_encoder.inverse_transform([pred_class])[0].lower()

        print(f"[DEBUG] Probabilities: {probabilities}")
        print(f"[DEBUG] Prediction: {disease_name} ({confidence:.2f})")

        # ═══════════════════════════════════════════════
        # RISK LEVEL LOGIC
        # ═══════════════════════════════════════════════
        if disease_name == "healthy":
            risk_level = "Healthy"

        elif disease_name == "panama":
            if moisture > 70 and temperature > 25:
                risk_level = "High Risk"
            elif moisture > 60:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"

        elif disease_name == "sigatoka":
            if humidity > 85 and 24 <= temperature <= 28:
                risk_level = "High Risk"
            elif humidity > 80:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"

        else:
            risk_level = "Low Risk"

        # ═══════════════════════════════════════════════
        # RESPONSE
        # ═══════════════════════════════════════════════
        return jsonify({
            "prediction": disease_name,
            "risk_level": risk_level,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "error": str(e),
            "prediction": "healthy",
            "risk_level": "Low Risk",
            "confidence": 0.5
        }), 400


# ═══════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
