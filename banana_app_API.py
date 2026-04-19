from flask import Flask, request, jsonify
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
# Load your trained ML model
# ═══════════════════════════════════════════════════════════════
model = None
try:
    # Load the trained banana_model.pkl
    with open('banana_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ ML model loaded successfully")
    print(f"  Model classes: {model.classes_}")
except FileNotFoundError:
    print("⚠ Warning: banana_model.pkl not found in current directory")
except Exception as e:
    print(f"⚠ Warning: Error loading model: {e}")

@app.route('/')
def home():
    return "Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get all sensor inputs
        temperature = data.get('temperature', 0)
        humidity = data.get('humidity', 0)
        moisture = data.get('moisture', 0)
        rainfall = data.get('rainfall', 0)
        
        # Prepare input for ML model
        features = np.array([[temperature, humidity, moisture, rainfall]])
        
        # Make prediction using ML model
        if model is not None:
            try:
                # Predict class
                prediction_class = int(model.predict(features)[0])
                
                # Get probabilities and confidence
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                
                # Map numeric class to disease name
                # Classes: 0=healthy, 1=panama, 2=sigatoka
                # (Adjust if needed based on your training labels)
                disease_map = {
                    0: "healthy",
                    1: "panama",
                    2: "sigatoka"
                }
                disease_name = disease_map.get(prediction_class, "healthy")
                
                # Determine risk level based on disease and environmental factors
                if disease_name == "healthy":
                    risk_level = "Low Risk"
                elif disease_name == "panama":
                    # Panama prefers high moisture and warmth
                    if moisture > 70 and temperature > 25:
                        risk_level = "High Risk"
                    elif moisture > 60 and temperature > 23:
                        risk_level = "Medium Risk"
                    else:
                        risk_level = "Low Risk"
                elif disease_name == "sigatoka":
                    # Sigatoka prefers high humidity and moderate temp
                    if humidity > 85 and 24 <= temperature <= 28:
                        risk_level = "High Risk"
                    elif humidity > 80 and 23 <= temperature <= 29:
                        risk_level = "Medium Risk"
                    else:
                        risk_level = "Low Risk"
                else:
                    risk_level = "Low Risk"
                
                return jsonify({
                    "prediction": disease_name,
                    "risk_level": risk_level,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fall back to threshold logic
                return _fallback_prediction(temperature, humidity, moisture, rainfall)
        else:
            # Use fallback logic if model not loaded
            return _fallback_prediction(temperature, humidity, moisture, rainfall)
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "prediction": "healthy",
            "risk_level": "Low Risk"
        }), 400

def _fallback_prediction(temperature, humidity, moisture, rainfall):
    """Fallback prediction logic based on environmental thresholds"""
    
    # Determine primary risk based on environmental conditions
    panama_score = 0
    sigatoka_score = 0
    
    # Panama Risk Factors (soil moisture + warmth)
    if moisture > 80 and temperature > 26:
        panama_score += 3
    elif moisture > 70 and temperature > 25:
        panama_score += 2
    elif moisture > 60:
        panama_score += 1
    
    # Sigatoka Risk Factors (humidity + moderate temps)
    if humidity > 85 and 24 <= temperature <= 28:
        sigatoka_score += 3
    elif humidity > 80 and 23 <= temperature <= 29:
        sigatoka_score += 2
    elif humidity > 75:
        sigatoka_score += 1
    
    # Determine disease
    if panama_score > sigatoka_score and panama_score > 0:
        disease_name = "panama"
        if panama_score >= 3:
            risk_level = "High Risk"
        elif panama_score >= 2:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
    elif sigatoka_score > panama_score and sigatoka_score > 0:
        disease_name = "sigatoka"
        if sigatoka_score >= 3:
            risk_level = "High Risk"
        elif sigatoka_score >= 2:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
    else:
        disease_name = "healthy"
        risk_level = "Low Risk"
    
    return jsonify({
        "prediction": disease_name,
        "risk_level": risk_level,
        "confidence": 0.70
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)